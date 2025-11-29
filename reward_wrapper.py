from reinforcement_learning.stat_wrapper import BaseStatWrapper
import numpy as np
from collections import defaultdict


class RewardWrapper(BaseStatWrapper):
    def __init__(
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,
        # === existing weights ===
        heal_bonus_weight=0.1,
        explore_bonus_weight=0.05,
        clip_unique_event=3,
        movement_penalty=0.0,
        combat_bonus_weight=0.0,
        coop_survival_weight=0.1,
        coop_combat_weight=0.2,
        coop_travel_weight=0.05,
        coop_heal_weight=0.1,
        coop_trade_weight=0.1,
        coop_distance_threshold=3,
        # === NEW: equipment / food reward weights (for Equip / Consume columns) ===
        equipment_gain_weight=0.01,
        equipment_loss_weight=0.01,
        food_gain_weight=0.05,
        food_loss_weight=0.05,
        # which type_ids count as "food" (fill this list for your env)
        food_type_ids=None,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix

        # existing weights
        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.movement_penalty = movement_penalty
        self.combat_bonus_weight = combat_bonus_weight
        self.coop_survival_weight = coop_survival_weight
        self.coop_combat_weight = coop_combat_weight
        self.coop_travel_weight = coop_travel_weight
        self.coop_heal_weight = coop_heal_weight
        self.coop_trade_weight = coop_trade_weight
        self.coop_distance_threshold = coop_distance_threshold

        # NEW: equipment / food weights
        self.equipment_gain_weight = equipment_gain_weight
        self.equipment_loss_weight = equipment_loss_weight
        self.food_gain_weight = food_gain_weight
        self.food_loss_weight = food_loss_weight

        # NEW: food type_ids
        self.food_type_ids = set(food_type_ids or [])

        # MUST MATCH ItemEncoder.continuous_idxs
        # ItemEncoder:
        #   continuous_idxs = [3,4,5,6,7,8,9,10,11,12,13,15]
        self.item_continuous_idxs = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15])

        self._reset_reward_vars()

    # ------------------------------------------------------------------
    # internal bookkeeping
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        self._reset_reward_vars()
        return super().reset(**kwargs)

    def _reset_reward_vars(self):
        # Per-agent history
        self._history = defaultdict(
            lambda: {
                "prev_price": 0,
                "prev_moves": [],
                "prev_health": 0,
                "prev_damage_dealt": 0,
                # inventory-based stats
                "prev_equipment_value": 0.0,
                "prev_food_count": 0.0,
                "equipment_delta": 0.0,
                "food_delta": 0.0,
            }
        )

        # Cooperative state (for coop_* rewards)
        self._coop_state = defaultdict(
            lambda: {
                "last_pos": None,
                "teammates_near": set(),
                "last_attack_target": None,
                "healed_ally": False,
                "traded": False,
            }
        )

    # ------------------------------------------------------------------
    # inventory helpers (aligned with ItemEncoder)
    # ------------------------------------------------------------------
    def _compute_inventory_values(self, inventory):
        """
        inventory: np.ndarray [num_items, feat_dim]
        Layout must match ItemEncoder:
          - type_id at index 1
          - equipped flag at index 14
          - continuous stats at self.item_continuous_idxs
        Returns:
          equipment_value: float  (sum of stats of equipped items)
          food_count:      float  (# of items whose type_id in self.food_type_ids)
        """
        if inventory is None:
            return 0.0, 0.0

        inventory = np.asarray(inventory)
        if inventory.size == 0 or inventory.ndim != 2 or inventory.shape[1] <= 15:
            return 0.0, 0.0

        type_ids = inventory[:, 1].astype(int)
        equipped_flags = inventory[:, 14].astype(int)

        # Equipment value = sum of continuous stats for equipped items
        equipped_mask = equipped_flags == 1
        if equipped_mask.any():
            try:
                eq_stats = inventory[equipped_mask][:, self.item_continuous_idxs]
                equipment_value = float(eq_stats.sum())
            except Exception:
                equipment_value = 0.0
        else:
            equipment_value = 0.0

        # Food count = items with type_id in self.food_type_ids
        if self.food_type_ids:
            food_mask = np.isin(type_ids, list(self.food_type_ids))
            food_count = float(food_mask.sum())
        else:
            food_count = 0.0

        return equipment_value, food_count

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------
    def observation(self, agent_id, agent_obs):
        # Existing "Sell price" trick
        if "ActionTargets" in agent_obs and "Sell" in agent_obs["ActionTargets"]:
            prev_price = self._history[agent_id]["prev_price"]
            if prev_price < len(agent_obs["ActionTargets"]["Sell"]["Price"]):
                agent_obs["ActionTargets"]["Sell"]["Price"][prev_price] = 0

        # NEW: track inventory changes here so deltas are ready for reward()
        inventory = agent_obs.get("Inventory", None)
        if inventory is not None:
            eq_val, food_count = self._compute_inventory_values(inventory)
            hist = self._history[agent_id]

            hist["equipment_delta"] = eq_val - hist["prev_equipment_value"]
            hist["food_delta"] = food_count - hist["prev_food_count"]

            hist["prev_equipment_value"] = eq_val
            hist["prev_food_count"] = food_count

        return agent_obs

    def action(self, agent_id, agent_atn):
        try:
            if "Sell" in agent_atn and "Price" in agent_atn["Sell"]:
                self._history[agent_id]["prev_price"] = agent_atn["Sell"]["Price"]

            if "Move" in agent_atn and "Direction" in agent_atn["Move"]:
                self._history[agent_id]["prev_moves"].append(agent_atn["Move"]["Direction"])
                if len(self._history[agent_id]["prev_moves"]) > 10:
                    self._history[agent_id]["prev_moves"].pop(0)

            if "Give" in agent_atn:
                self._coop_state[agent_id]["traded"] = True
        except Exception:
            # be robust in case the env structure changes
            pass

        return agent_atn

    # ------------------------------------------------------------------
    # main reward function
    # ------------------------------------------------------------------
    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        realm = self.env.realm
        player = realm.players.get(agent_id, None) if realm is not None else None

        # ---------------- core existing components ----------------
        # Healing bonus
        healing_bonus = 0.0
        if self.heal_bonus_weight > 0 and player is not None:
            if getattr(player.resources, "health_restore", 0) > 0:
                healing_bonus = self.heal_bonus_weight * player.resources.health_restore

        # Exploration bonus (uses BaseStatWrapper's unique event tracking)
        explore_bonus = 0.0
        uniq = self._unique_events.get(agent_id, None)
        if uniq and uniq["curr_count"] > uniq["prev_count"]:
            delta = min(
                self.clip_unique_event,
                uniq["curr_count"] - uniq["prev_count"],
            )
            explore_bonus = delta * self.explore_bonus_weight

        # Cooperative rewards
        coop_reward = 0.0

        # 1. Survival together
        if realm is not None and hasattr(realm, "players"):
            alive_agents = [
                a
                for a, p in realm.players.items()
                if p is not None
                and getattr(p, "alive", True)
                and getattr(getattr(p, "resources", None), "health", 1) > 0
            ]
            if len(alive_agents) > 1:
                coop_reward += self.coop_survival_weight * (len(alive_agents) - 1) / max(1, len(realm.players))

        # 2. Travel together
        if player is not None and hasattr(player, "pos") and player.pos is not None:
            px, py = player.pos if isinstance(player.pos, (tuple, list)) else (player.pos.x, player.pos.y)
            nearby = []
            for aid, ally in realm.players.items():
                if aid == agent_id or ally is None or not getattr(ally, "resources", None):
                    continue
                if getattr(ally.resources, "health", 1) <= 0 or not getattr(ally, "alive", True):
                    continue
                if not hasattr(ally, "pos") or ally.pos is None:
                    continue

                ax, ay = ally.pos if isinstance(ally.pos, (tuple, list)) else (ally.pos.x, ally.pos.y)
                dist = ((ax - px) ** 2 + (ay - py) ** 2) ** 0.5
                if dist <= self.coop_distance_threshold:
                    nearby.append(aid)

            if nearby:
                coop_reward += self.coop_travel_weight * len(nearby)
                self._coop_state[agent_id]["teammates_near"] = set(nearby)

        # 3. Fighting together
        if player is not None and hasattr(player, "history") and hasattr(player.history, "last_attacked"):
            target = player.history.last_attacked
            if target:
                for ally_id, ally in realm.players.items():
                    if ally_id == agent_id or ally is None:
                        continue
                    if hasattr(ally, "history") and getattr(ally.history, "last_attacked", None) == target:
                        coop_reward += self.coop_combat_weight
                        break

        # 4. Healing allies
        if player is not None and getattr(getattr(player, "resources", None), "health_restore", 0) > 0:
            coop_reward += self.coop_heal_weight

        # 5. Trading items or gold
        if self._coop_state[agent_id]["traded"]:
            coop_reward += self.coop_trade_weight
            self._coop_state[agent_id]["traded"] = False

        # ---------------- NEW: equipment & food (Equip / Consume) ----------------
        hist = self._history[agent_id]
        eq_delta = hist.get("equipment_delta", 0.0)
        food_delta = hist.get("food_delta", 0.0)

        equipment_reward = 0.0  # → “Equip” column
        food_reward = 0.0       # → “Consume” column

        # Equipment: reward improvements, penalize downgrades
        if eq_delta > 0:
            equipment_reward = self.equipment_gain_weight * eq_delta
        elif eq_delta < 0:
            equipment_reward = -self.equipment_loss_weight * (-eq_delta)

        # Food: reward stocking food, penalize losing it (eating / wasting)
        if food_delta > 0:
            food_reward = self.food_gain_weight * food_delta
        elif food_delta < 0:
            food_reward = -self.food_loss_weight * (-food_delta)

        # reset deltas once used
        hist["equipment_delta"] = 0.0
        hist["food_delta"] = 0.0

        # ---------------- combine ----------------
        custom_reward = healing_bonus + explore_bonus + coop_reward + equipment_reward + food_reward
        reward += custom_reward

        if custom_reward != 0:
            # Expose components so eval script can aggregate into table columns
            info.setdefault("custom_rewards", {}).update(
                {
                    "healing_bonus": float(healing_bonus),
                    "explore_bonus": float(explore_bonus),
                    "coop_reward": float(coop_reward),
                    "equipment_reward": float(equipment_reward),  # -> Equip
                    "food_reward": float(food_reward),            # -> Consume
                    "total_custom": float(custom_reward),
                }
            )

        return reward, terminated, truncated, info
