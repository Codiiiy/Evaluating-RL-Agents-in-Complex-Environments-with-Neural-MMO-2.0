import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
import pufferlib.emulation
from pufferlib.emulation import unpack_batched_obs

from nmmo.entity.entity import EntityState

EntityId = EntityState.State.attr_name_to_col["id"]


class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)


class Baseline(pufferlib.models.Policy):

    def __init__(self, env, input_size=256, hidden_size=256, task_size=2048):
        super().__init__(env)

        self.unflatten_context = env.unflatten_context

        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.inventory_encoder = InventoryEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)
        self.proj_fc = torch.nn.Linear(6 * input_size, input_size)
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def encode_observations(self, flat_observations):
        device = flat_observations.device if hasattr(flat_observations, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if next(self.parameters()).device != device:
            self.to(device)
        
        env_outputs = unpack_batched_obs(flat_observations, self.unflatten_context)
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )

        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        market_embeddings = self.item_encoder(env_outputs["Market"])
        market = self.market_encoder(market_embeddings)
        task = self.task_encoder(env_outputs["Task"])
        pooled_item_embeddings = item_embeddings.mean(dim=1)
        pooled_player_embeddings = player_embeddings.mean(dim=1)
        obs = torch.cat(
            [tile, my_agent, pooled_player_embeddings, pooled_item_embeddings, market, task], dim=-1
        )
        obs = self.proj_fc(obs)


        embeddings = [player_embeddings, item_embeddings, market_embeddings]
        padded_embeddings = []
        for embedding in embeddings:
            padding_size = 1
            padding = torch.zeros(
                embedding.size(0), padding_size, embedding.size(2), device=embedding.device
            )
            padded_embedding = torch.cat([embedding, padding], dim=1)
            padded_embeddings.append(padded_embedding)
        
        player_embeddings, item_embeddings, market_embeddings = padded_embeddings

        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class TileEncoder(torch.nn.Module):
    """Encodes tile observations with CNN"""
    def __init__(self, input_size):
        super().__init__()
        self.register_buffer('tile_offset', torch.tensor([i * 256 for i in range(3)]))
        self.embedding = torch.nn.Embedding(3 * 256, 32)

        self.tile_conv_1 = torch.nn.Conv2d(96, 32, 3)
        self.tile_conv_2 = torch.nn.Conv2d(32, 8, 3)
        self.tile_fc = torch.nn.Linear(8 * 11 * 11, input_size)

    def forward(self, tile):
        if tile.device != next(self.parameters()).device:
            self.to(tile.device)
        tile[:, :, :2] -= tile[:, 112:113, :2].clone()
        tile[:, :, :2] += 7
        if self.tile_offset.device != tile.device:
            self.tile_offset = self.tile_offset.to(tile.device)
        if next(self.embedding.parameters()).device != tile.device:
            self.embedding = self.embedding.to(tile.device)
        tile = self.embedding(tile.long().clip(0, 255) + self.tile_offset)

        agents, tiles, features, embed = tile.shape
        tile = (
            tile.view(agents, tiles, features * embed)
            .transpose(1, 2)
            .view(agents, features * embed, 15, 15)
        )

        tile = F.relu(self.tile_conv_1(tile))
        tile = F.relu(self.tile_conv_2(tile))
        tile = tile.contiguous().view(agents, -1)
        tile = F.relu(self.tile_fc(tile))

        return tile


class PlayerEncoder(torch.nn.Module):
    """Encodes player/entity observations"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.entity_dim = 31
        self.num_classes_npc_type = 5
        self.agent_fc = torch.nn.Linear(self.entity_dim + 5 - 1, hidden_size)
        self.my_agent_fc = torch.nn.Linear(self.entity_dim + 5 - 1, input_size)

    def forward(self, agents, my_id):
        npc_type = agents[:, :, 1]
        one_hot_npc_type = F.one_hot(
            npc_type.long(), num_classes=self.num_classes_npc_type
        ).float()
        one_hot_agents = torch.cat(
            [agents[:, :, :1], one_hot_npc_type, agents[:, :, 2:]], dim=-1
        ).float()

        agent_ids = one_hot_agents[:, :, EntityId]
        mask = (agent_ids == my_id.unsqueeze(1)) & (agent_ids != 0)
        mask = mask.int()
        row_indices = torch.where(
            mask.any(dim=1), mask.argmax(dim=1), torch.zeros_like(mask.sum(dim=1))
        )

        my_agent_embeddings = one_hot_agents[torch.arange(one_hot_agents.shape[0]), row_indices]
        agent_embeddings = self.agent_fc(one_hot_agents)
        my_agent_embeddings = self.my_agent_fc(my_agent_embeddings)
        my_agent_embeddings = F.relu(my_agent_embeddings)

        return agent_embeddings, my_agent_embeddings


class ItemEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(18 + 2 + 14 - 2, hidden_size)
        self.discrete_idxs = [1, 14]
        self.discrete_offset = torch.Tensor([2, 0])
        self.continuous_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15]
        self.continuous_scale = torch.Tensor([1 / 100] * 12)

    def forward(self, items):
        if self.discrete_offset.device != items.device:
            self.discrete_offset = self.discrete_offset.to(items.device)
            self.continuous_scale = self.continuous_scale.to(items.device)

        one_hot_discrete_equipped = F.one_hot(items[:, :, 14].long(), num_classes=2).float()
        one_hot_discrete_type_id = F.one_hot(items[:, :, 1].long(), num_classes=18).float()
        one_hot_discrete = torch.concat(
            [one_hot_discrete_type_id, one_hot_discrete_equipped], dim=-1
        )
        continuous = items[:, :, self.continuous_idxs] * self.continuous_scale
        item_embeddings = torch.cat([one_hot_discrete, continuous], dim=-1).float()
        item_embeddings = self.fc(item_embeddings)
        return item_embeddings


class InventoryEncoder(torch.nn.Module):
    """Encodes full inventory"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(12 * hidden_size, input_size)

    def forward(self, inventory):
        agents, items, hidden = inventory.shape
        inventory = inventory.view(agents, items * hidden)
        return self.fc(inventory)


class MarketEncoder(torch.nn.Module):
    """Encodes market observations"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, market):
        return self.fc(market).mean(-2)


class TaskEncoder(torch.nn.Module):
    """Encodes task specifications"""
    def __init__(self, input_size, hidden_size, task_size):
        super().__init__()
        self.fc = torch.nn.Linear(task_size, input_size)

    def forward(self, task):
        return self.fc(task.clone().float())


class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "attack_style": torch.nn.Linear(hidden_size, 3),
                "attack_target": torch.nn.Linear(hidden_size, hidden_size),
                "market_buy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_destroy": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_item": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_give_player": torch.nn.Linear(hidden_size, hidden_size),
                "gold_quantity": torch.nn.Linear(hidden_size, 99),
                "gold_target": torch.nn.Linear(hidden_size, hidden_size),
                "move": torch.nn.Linear(hidden_size, 5),
                "inventory_sell": torch.nn.Linear(hidden_size, hidden_size),
                "inventory_price": torch.nn.Linear(hidden_size, 99),
                "inventory_use": torch.nn.Linear(hidden_size, hidden_size),
            }
        )

    def apply_layer(self, layer, embeddings, mask, hidden):
        hidden = layer(hidden)
        if hidden.dim() == 2 and embeddings is not None:
            hidden = torch.matmul(embeddings, hidden.unsqueeze(-1)).squeeze(-1)

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

        action_targets = {
            "attack_style": action_targets["Attack"]["Style"],
            "attack_target": action_targets["Attack"]["Target"],
            "market_buy": action_targets["Buy"]["MarketItem"],
            "inventory_destroy": action_targets["Destroy"]["InventoryItem"],
            "inventory_give_item": action_targets["Give"]["InventoryItem"],
            "inventory_give_player": action_targets["Give"]["Target"],
            "gold_quantity": action_targets["GiveGold"]["Price"],
            "gold_target": action_targets["GiveGold"]["Target"],
            "move": action_targets["Move"]["Direction"],
            "inventory_sell": action_targets["Sell"]["InventoryItem"],
            "inventory_price": action_targets["Sell"]["Price"],
            "inventory_use": action_targets["Use"]["InventoryItem"],
        }

        actions = []
        for key, layer in self.layers.items():
            mask = action_targets[key]
            embs = embeddings.get(key)

            action = self.apply_layer(layer, embs, mask, hidden)
            actions.append(action)

        return actions