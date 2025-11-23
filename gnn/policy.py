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
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True, use_residual=True):
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
        self.layernorm = nn.LayerNorm(num_heads * out_features)

    def forward(self, x, mask=None, positions=None, topk=None):
        batch_size, num_nodes, _ = x.shape
        h = torch.matmul(x, self.W).view(batch_size, num_nodes, self.num_heads, self.out_features)
        e_i = (h * self.a_src).sum(dim=-1)
        e_j = (h * self.a_dst).sum(dim=-1)
        if positions is not None and topk is not None:
            idx = topk_neighbors(positions, k=topk)
            h_neighbors = h.gather(1, idx.unsqueeze(-2).expand(-1, -1, self.num_heads, self.out_features))
            e_i_neighbors = e_i.gather(1, idx)
            e_j_neighbors = e_j.gather(1, idx)
            e = self.leakyrelu(e_i_neighbors.unsqueeze(2) + e_j_neighbors.unsqueeze(1))
        else:
            e = self.leakyrelu(e_i.unsqueeze(2) + e_j.unsqueeze(1))
        if mask is not None:
            mask_matrix = mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(2).unsqueeze(-1)
            e = e.masked_fill(mask_matrix == 0, -1e9)
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)
        h_prime = torch.einsum('bnjh,bjhd->bnhd', attention, h)
        if self.concat:
            h_prime = h_prime.reshape(batch_size, num_nodes, -1)
        else:
            h_prime = h_prime.mean(dim=2)
        if self.use_residual:
            h_prime = self.layernorm(h_prime + x)
        return h_prime

class SpatialGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, topk=8):
        super().__init__()
        self.gat = GATLayer(in_features, out_features, num_heads, dropout)
        self.distance_embed = nn.Linear(1, num_heads)
        self.topk = topk

    def forward(self, x, positions, mask=None):
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        distances = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        dist_weights = torch.sigmoid(-distances / 10.0)
        gnn_features = self.gat(x, mask, positions=positions, topk=self.topk)
        return gnn_features, dist_weights

class TeamEncoder(nn.Module):
    def __init__(self, input_size, num_heads=4, distance_threshold=7):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.entity_proj = nn.Linear(31, input_size)
        self.gat1 = GATLayer(input_size, input_size // num_heads, num_heads=num_heads)
        self.gat2 = GATLayer(input_size, input_size // num_heads, num_heads=num_heads, concat=False)
        self.fc = nn.Linear(input_size // num_heads, input_size)

    def forward(self, entity_obs, my_id):
        batch_size, num_entities, feat_dim = entity_obs.shape
        ids = entity_obs[:, :, EntityId]
        mask_self = (ids == my_id.unsqueeze(1))
        ally_mask = (ids != 0) & (~mask_self)
        positions = entity_obs[:, :, 2:4]
        entity_features = self.entity_proj(entity_obs.float())
        h1 = F.relu(self.gat1(entity_features, ally_mask.float()))
        h2 = F.relu(self.gat2(h1, ally_mask.float()))
        if mask_self.any():
            mask_self_int = mask_self.int()
            self_pos = entity_obs[torch.arange(batch_size), mask_self_int.argmax(dim=1), 2:4]
            dists = torch.norm(positions - self_pos.unsqueeze(1), dim=-1)
            close_mask = ally_mask & (dists <= self.distance_threshold)
        else:
            close_mask = ally_mask
        close_mask_expanded = close_mask.unsqueeze(-1).float()
        if close_mask_expanded.sum(dim=1).max() > 0:
            team_embedding = (h2 * close_mask_expanded).sum(dim=1) / (close_mask_expanded.sum(dim=1) + 1e-8)
        else:
            team_embedding = h2.mean(dim=1)
        return self.fc(team_embedding)

class TileEncoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.register_buffer('tile_offset', torch.tensor([i * 256 for i in range(3)], dtype=torch.long))
        self.embedding = nn.Embedding(3 * 256, 32)
        self.spatial_gat = SpatialGATLayer(96, 64, num_heads=4)
        self.gat2 = GATLayer(256, 128, num_heads=2, concat=False)
        self.tile_conv_1 = nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = nn.Conv2d(32, 8, 3)
        self.tile_fc = nn.Linear(8 * 11 * 11 + 256, input_size)

    def forward(self, tile):
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7
        tile_embedded = self.embedding(tile.long().clip(0, 255) + self.tile_offset)
        agents, tiles, features, embed = tile_embedded.shape
        tile_flat = tile_embedded.view(agents, tiles, features * embed)
        positions = tile[:, :, :2].float()
        gnn_features, _ = self.spatial_gat(tile_flat, positions)
        gnn_pooled = gnn_features.mean(dim=1)
        tile_conv_input = tile_flat.transpose(1, 2).view(agents, features * embed, 15, 15)
        tile_conv = F.relu(self.tile_conv_1(tile_conv_input))
        tile_conv = F.relu(self.tile_conv_2(tile_conv))
        tile_conv = tile_conv.contiguous().view(agents, -1)
        combined = torch.cat([tile_conv, gnn_pooled], dim=-1)
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
        one_hot_npc_type = F.one_hot(npc_type.long(), num_classes=self.num_classes_npc_type).float()
        one_hot_agents = torch.cat([agents[:, :, :1], one_hot_npc_type, agents[:, :, 2:]], dim=-1).float()
        agent_ids = one_hot_agents[:, :, EntityId]
        entity_mask = (agent_ids != 0).float()
        agent_features = self.feature_proj(one_hot_agents)
        h1 = F.relu(self.gat1(agent_features, entity_mask))
        h2 = F.relu(self.gat2(h1, entity_mask))
        agent_embeddings = self.agent_fc(h2)
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1)))
        my_agent_embeddings = h2[torch.arange(h2.shape[0]), row_indices]
        my_agent_embeddings = F.relu(self.my_agent_fc(my_agent_embeddings))
        return agent_embeddings, my_agent_embeddings

class ItemEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super().__init__()
        self.discrete_idxs = [1, 14]
        self.continuous_idxs = [3,4,5,6,7,8,9,10,11,12,13,15]
        self.register_buffer('discrete_offset', torch.tensor([2,0], dtype=torch.float))
        self.register_buffer('continuous_scale', torch.tensor([1/100.0]*12, dtype=torch.float))
        self.feature_proj = nn.Linear(18 + 2 + 12, hidden_size)
        self.gat1 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads)
        self.gat2 = GATLayer(hidden_size, hidden_size // num_heads, num_heads=num_heads, concat=False)
        self.fc = nn.Linear(hidden_size // num_heads, hidden_size)

    def forward(self, items):
        one_hot_discrete_equipped = F.one_hot(items[:, :, 14].long(), num_classes=2).float()
        one_hot_discrete_type_id = F.one_hot(items[:, :, 1].long(), num_classes=18).float()
        one_hot_discrete = torch.cat([one_hot_discrete_type_id, one_hot_discrete_equipped], dim=-1)
        continuous = items[:, :, self.continuous_idxs] * self.continuous_scale
        item_features = torch.cat([one_hot_discrete, continuous], dim=-1).float()
        item_mask = (items[:, :, 1] != 0).float()
        item_proj = self.feature_proj(item_features)
        h1 = F.relu(self.gat1(item_proj, item_mask))
        h2 = F.relu(self.gat2(h1, item_mask))
        return self.fc(h2)

class InventoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = nn.Linear(12 * hidden_size, input_size)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
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
        self.layers = nn.ModuleDict({
            "attack_style": nn.Linear(hidden_size, 3),
            "attack_target": nn.Linear(hidden_size, hidden_size),
            "market_buy": nn.Linear(hidden_size, hidden_size),
            "inventory_destroy": nn.Linear(hidden_size, hidden_size),
            "inventory_give_item": nn.Linear(hidden_size, hidden_size),
            "inventory_give_player": nn.Linear(hidden_size, hidden_size),
            "gold_quantity": nn.Linear(hidden_size, 99),
            "gold_target": nn.Linear(hidden_size, hidden_size),
            "move": nn.Linear(hidden_size, 5),
            "inventory_sell": nn.Linear(hidden_size, hidden_size),
            "inventory_price": nn.Linear(hidden_size, 99),
            "inventory_use": nn.Linear(hidden_size, hidden_size),
        })

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            hidden = hidden.masked_fill(mask == 0, -1e9)
        return hidden

    def forward(self, hidden, lookup):
        player_embeddings, inventory_embeddings, market_embeddings, action_targets = lookup
        embeddings = {
            "attack_target": player_embeddings,
            "market_buy": market_embeddings,
            "inventory_destroy": inventory_embeddings,
            "inventory_give_item": inventory_embeddings,
            "inventory_give_player": player_embeddings,
            "gold_target": player_embeddings,
            "inventory_sell": inventory_embeddings,
            "inventory_use": inventory_embeddings,
        }
        actions = []
        for key, layer in self.layers.items():
            mask = action_targets[key]
            embs = embeddings.get(key)
            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)
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
        self.team_encoder = TeamEncoder(input_size)
        self.proj_fc = nn.Linear(7 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        device = flat_observations.device if hasattr(flat_observations, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(self.parameters()).device != device:
            self.to(device)
        env_outputs = unpack_batched_obs(flat_observations, self.unflatten_context)
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(env_outputs["Entity"], env_outputs["AgentId"][:, 0])
        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        market_embeddings = self.item_encoder(env_outputs["Market"])
        market = self.market_encoder(market_embeddings)
        task = self.task_encoder(env_outputs["Task"])
        pooled_item_embeddings = item_embeddings.mean(dim=1)
        pooled_player_embeddings = player_embeddings.mean(dim=1)
        team = self.team_encoder(env_outputs["Entity"], env_outputs["AgentId"][:, 0])
        obs = torch.cat([tile, my_agent, pooled_player_embeddings, pooled_item_embeddings, market, task, team], dim=-1)
        obs = self.proj_fc(obs)
        embeddings = [player_embeddings, item_embeddings, market_embeddings]
        padded_embeddings = []
        for embedding in embeddings:
            padding_size = 1
            padding = torch.zeros(embedding.size(0), padding_size, embedding.size(2), device=embedding.device)
            padded_embeddings.append(torch.cat([embedding, padding], dim=1))
        player_embeddings, item_embeddings, market_embeddings = padded_embeddings
        return obs, (player_embeddings, item_embeddings, market_embeddings, env_outputs["ActionTargets"])

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value
    

