import torch
import torch.nn as nn
import torch.nn.functional as F
import pufferlib
import pufferlib.models
from pufferlib.emulation import unpack_batched_obs
from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]

class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
        
def topk_neighbors(positions, k=8):
    dists = torch.cdist(positions, positions)
    idx = torch.topk(-dists, k=k, dim=-1).indices
    return idx

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1,
                 concat=True, use_residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.use_residual = use_residual

        self.W = nn.Parameter(torch.empty(size=(in_features, num_heads * out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_src = nn.Parameter(torch.empty(size=(num_heads, out_features)))
        self.a_dst = nn.Parameter(torch.empty(size=(num_heads, out_features)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.layernorm = nn.LayerNorm(num_heads * out_features if concat else out_features)

    def forward(self, x, mask=None, positions=None, topk=None):
        B, N, _ = x.shape

        h = torch.matmul(x, self.W).view(B, N, self.num_heads, self.out_features)

        e_i = (h * self.a_src).sum(dim=-1)
        e_j = (h * self.a_dst).sum(dim=-1)

        if positions is not None and topk is not None:
            idx = topk_neighbors(positions, k=topk) 
            k = idx.shape[-1]

            idx_nodes = idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, k, self.num_heads, self.out_features)
            h_neighbors = h.unsqueeze(2).expand(B, N, k, self.num_heads, self.out_features)
            h_neighbors = h_neighbors.gather(1, idx_nodes)

            idx_heads = idx.unsqueeze(-1).expand(B, N, k, self.num_heads)
            e_i_neighbors = e_i.unsqueeze(2).expand(B, N, k, self.num_heads).gather(1, idx_heads)
            e_j_neighbors = e_j.unsqueeze(2).expand(B, N, k, self.num_heads).gather(1, idx_heads)

            scores = self.leakyrelu(e_i_neighbors + e_j_neighbors)

            if mask is not None:
                neighbor_mask = mask.gather(1, idx)  
                scores = scores.masked_fill(neighbor_mask.unsqueeze(-1) == 0, -1e9)

            attn = F.softmax(scores, dim=2)
            attn = self.dropout(attn)

            h_prime = (attn.unsqueeze(-1) * h_neighbors).sum(dim=2)

        else:
            e = self.leakyrelu(e_i.unsqueeze(2) + e_j.unsqueeze(1))

            if mask is not None:
                mask_matrix = mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(2).unsqueeze(-1)
                e = e.masked_fill(mask_matrix == 0, -1e9)

            attn = F.softmax(e, dim=2)
            attn = self.dropout(attn)
            h_prime = torch.einsum('bnjh,bjhd->bnhd', attn, h)

        if self.concat:
            h_prime = h_prime.reshape(B, N, -1)
        else:
            h_prime = h_prime.mean(dim=2)

        if self.use_residual:
            if h_prime.shape[-1] == x.shape[-1]:
                h_prime = self.layernorm(h_prime + x)
            else:
                h_prime = self.layernorm(h_prime)

        return h_prime


class SpatialGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, topk=8):
        super().__init__()
        self.gat = GATLayer(in_features, out_features, num_heads, dropout)
        self.distance_embed = nn.Linear(1, num_heads)
        self.topk = topk

    def forward(self, x, positions, mask=None):
        gnn_features = self.gat(x, mask, positions=positions, topk=self.topk)
        return gnn_features, None


class TileEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.register_buffer("tile_offset", torch.tensor([i * 256 for i in range(3)], dtype=torch.long))
        self.embedding = nn.Embedding(3 * 256, 32)
        self.spatial_gat = SpatialGATLayer(96, 64, num_heads=4)
        self.tile_conv_1 = nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = nn.Conv2d(32, 8, 3)
        self.tile_fc = nn.Linear(8 * 11 * 11 + 256, input_size)

    def forward(self, tile):
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7

        embedded = self.embedding(tile.long().clip(0, 255) + self.tile_offset)
        agents, tiles, features, embed_dim = embedded.shape

        tile_flat = embedded.view(agents, tiles, features * embed_dim)
        positions = tile[:, :, :2].float()
        gnn_features, _ = self.spatial_gat(tile_flat, positions)
        gnn_pool = gnn_features.mean(dim=1)

        conv_input = tile_flat.transpose(1, 2).view(agents, features * embed_dim, 15, 15)
        conv = F.relu(self.tile_conv_1(conv_input))
        conv = F.relu(self.tile_conv_2(conv))
        conv = conv.contiguous().view(agents, -1)

        combined = torch.cat([conv, gnn_pool], dim=-1)
        return F.relu(self.tile_fc(combined))


class PlayerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.entity_dim = 31
        self.num_classes_npc_type = 5
        self.feature_proj = nn.Linear(self.entity_dim + 4, hidden_size)
        self.gat1 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads)
        self.gat2 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads, concat=False)
        self.agent_fc = nn.Linear(hidden_size // num_heads, hidden_size)
        self.my_agent_fc = nn.Linear(hidden_size // num_heads, input_size)

    def forward(self, agents, my_id):
        npc_type = agents[:, :, 1]
        one_hot_npc = F.one_hot(npc_type.long(), num_classes=self.num_classes_npc_type).float()
        merged = torch.cat([agents[:, :, :1], one_hot_npc, agents[:, :, 2:]], dim=-1)

        mask = (merged[:, :, EntityId] != 0).float()

        proj = self.feature_proj(merged.float())
        h1 = F.relu(self.gat1(proj, mask))
        h2 = F.relu(self.gat2(h1, mask))

        agent_embeddings = self.agent_fc(h2)

        self_mask = merged[:, :, EntityId] == my_id.unsqueeze(1)
        idx = self_mask.int().argmax(dim=1)

        my_emb = h2[torch.arange(h2.size(0)), idx]
        my_emb = F.relu(self.my_agent_fc(my_emb))

        return agent_embeddings, my_emb


class ItemEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.discrete_idxs = [1, 14]
        self.continuous_idxs = [3,4,5,6,7,8,9,10,11,12,13,15]
        self.register_buffer(
            "continuous_scale",
            torch.tensor([1/100.0]*12, dtype=torch.float32)
        )

        self.feature_proj = nn.Linear(18 + 2 + 12, hidden_size)

        self.gat1 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads)
        self.gat2 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads, concat=False)

        self.fc = nn.Linear(hidden_size // num_heads, hidden_size)

    def forward(self, items):
        items = items.float()  

        one_hot_equip = F.one_hot(items[:, :, 14].long(), num_classes=2).float()
        one_hot_type = F.one_hot(items[:, :, 1].long(), num_classes=18).float()
        one_hot = torch.cat([one_hot_type, one_hot_equip], dim=-1)

        continuous = items[:, :, self.continuous_idxs] * self.continuous_scale
        merged = torch.cat([one_hot, continuous], dim=-1).float()  

        mask = (items[:, :, 1] != 0).float()
        proj = self.feature_proj(merged)
        h1 = F.relu(self.gat1(proj, mask))
        h2 = F.relu(self.gat2(h1, mask))

        return self.fc(h2)


class InventoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(12 * hidden_size, input_size)

    def forward(self, inventory):
        b, items, hidden = inventory.shape
        inventory = inventory.reshape(b, items * hidden)
        return self.fc(inventory)


class MarketEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.gat = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads, concat=False)
        self.fc = nn.Linear(hidden_size // num_heads, input_size)

    def forward(self, market):
        h = self.gat(market)
        h = F.relu(self.fc(h))
        return h.mean(-2)

class TaskEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, task_size):
        super().__init__()
        self.fc = nn.Linear(task_size, input_size)

    def forward(self, task):
        return self.fc(task.clone().float())

class ActionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 12 heads – matches baseline neurips23_start_kit policy
        self.layers = nn.ModuleDict({
            "attack_style":    nn.Linear(hidden_size, 3),          # categorical (3 styles)
            "attack_target":   nn.Linear(hidden_size, hidden_size),# pointer → players
            "market_buy":      nn.Linear(hidden_size, hidden_size),# pointer → market items
            "inventory_destroy": nn.Linear(hidden_size, hidden_size),
            "inventory_give_item": nn.Linear(hidden_size, hidden_size),
            "inventory_give_player": nn.Linear(hidden_size, hidden_size),
            "gold_quantity":   nn.Linear(hidden_size, 99),         # price / quantity
            "gold_target":     nn.Linear(hidden_size, hidden_size),
            "move":            nn.Linear(hidden_size, 5),          # 5 move directions
            "inventory_sell":  nn.Linear(hidden_size, hidden_size),
            "inventory_price": nn.Linear(hidden_size, 99),
            "inventory_use":   nn.Linear(hidden_size, hidden_size),
        })

    def apply_layer(self, layer, embeddings, mask, hidden):
        # Hidden → logits or key vector
        hidden = layer(hidden)

        # For pointer-type heads: project hidden onto embedding set
        if hidden.dim() == 2 and embeddings is not None:
            # embeddings: [B, N, D], hidden: [B, D] → [B, N]
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

        # Mask out invalid targets
        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)

        return hidden

    def forward(self, hidden, lookup):
        (
            player_embeddings,
            inventory_embeddings,
            market_embeddings,
            action_targets,
        ) = lookup

        # Which embedding table each head should use (or None for pure categorical)
        embeddings = {
            "attack_target":     player_embeddings,
            "market_buy":        market_embeddings,
            "inventory_destroy": inventory_embeddings,
            "inventory_give_item": inventory_embeddings,
            "inventory_give_player": player_embeddings,
            "gold_target":       player_embeddings,
            "inventory_sell":    inventory_embeddings,
            "inventory_use":     inventory_embeddings,
        }

        # Map flattened head names → env["ActionTargets"] structure
        at = {
            "attack_style":      action_targets["Attack"]["Style"],
            "attack_target":     action_targets["Attack"]["Target"],
            "market_buy":        action_targets["Buy"]["MarketItem"],
            "inventory_destroy": action_targets["Destroy"]["InventoryItem"],
            "inventory_give_item": action_targets["Give"]["InventoryItem"],
            "inventory_give_player": action_targets["Give"]["Target"],
            "gold_quantity":     action_targets["GiveGold"]["Price"],
            "gold_target":       action_targets["GiveGold"]["Target"],
            "move":              action_targets["Move"]["Direction"],
            "inventory_sell":    action_targets["Sell"]["InventoryItem"],
            "inventory_price":   action_targets["Sell"]["Price"],
            "inventory_use":     action_targets["Use"]["InventoryItem"],
        }

        actions = []
        for key, layer in self.layers.items():
            mask = at[key]
            embs = embeddings.get(key, None)
            action_logits = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action_logits)

        return actions

class Policy(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256, task_size=2048):
        super().__init__(env)
        self.unflatten_context = env.unflatten_context

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)

        self.proj_fc = nn.Linear(6 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_obs):
        env = unpack_batched_obs(flat_obs, self.unflatten_context)

        tile = self.tile_encoder(env["Tile"])
        p_emb, my_agent = self.player_encoder(env["Entity"], env["AgentId"][:, 0])
        items = self.item_encoder(env["Inventory"])
        market_items = self.item_encoder(env["Market"])
        market = self.market_encoder(market_items)
        task = self.task_encoder(env["Task"])

        pooled_items = items.mean(1)
        pooled_players = p_emb.mean(1)

        obs = torch.cat([tile, my_agent, pooled_players, pooled_items, market, task], dim=-1)
        obs = self.proj_fc(obs)

        padded = []
        for emb in [p_emb, items, market_items]:
            pad = torch.zeros(emb.size(0), 1, emb.size(2), device=emb.device)
            padded.append(torch.cat([emb, pad], dim=1))

        return obs, (padded[0], padded[1], padded[2], env["ActionTargets"])

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value
    

