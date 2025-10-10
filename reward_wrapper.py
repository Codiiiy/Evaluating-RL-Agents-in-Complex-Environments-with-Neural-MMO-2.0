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
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix
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

    def reset(self, **kwargs):
        self._reset_reward_vars()
        return super().reset(**kwargs)

    def _reset_reward_vars(self):
        self._history = defaultdict(lambda: {
            "prev_price": 0,
            "prev_moves": [],
            "prev_health": 0,
            "prev_damage_dealt": 0,
        })
        self._coop_state = defaultdict(lambda: {
            "last_pos": None,
            "teammates_near": set(),
            "last_attack_target": None,
            "healed_ally": False,
            "traded": False,
        })

    def observation(self, agent_id, agent_obs):
        if "ActionTargets" in agent_obs and "Sell" in agent_obs["ActionTargets"]:
            prev_price = self._history[agent_id]["prev_price"]
            if prev_price < len(agent_obs["ActionTargets"]["Sell"]["Price"]):
                agent_obs["ActionTargets"]["Sell"]["Price"][prev_price] = 0
        return agent_obs

    def action(self, agent_id, agent_atn):
        if "Sell" in agent_atn and "Price" in agent_atn["Sell"]:
            self._history[agent_id]["prev_price"] = agent_atn["Sell"]["Price"]
        if "Move" in agent_atn and "Direction" in agent_atn["Move"]:
            self._history[agent_id]["prev_moves"].append(agent_atn["Move"]["Direction"])
            if len(self._history[agent_id]["prev_moves"]) > 10:
                self._history[agent_id]["prev_moves"].pop(0)
        if "Give" in agent_atn:
            self._coop_state[agent_id]["traded"] = True
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        realm = self.env.realm
        player = realm.players.get(agent_id, None)
        custom_reward = 0

        healing_bonus = 0
        if self.heal_bonus_weight > 0 and player:
            if player.resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight * player.resources.health_restore

        explore_bonus = 0
        uniq = self._unique_events.get(agent_id, None)
        if uniq and uniq["curr_count"] > uniq["prev_count"]:
            event_delta = min(self.clip_unique_event, uniq["curr_count"] - uniq["prev_count"])
            explore_bonus = event_delta * self.explore_bonus_weight

        coop_reward = 0

       # 1. Survival together
        alive_agents = [a for a, p in realm.players.items() if getattr(p, "alive", True) and p.resources.health > 0]
        if len(alive_agents) > 1:
            coop_reward += self.coop_survival_weight * (len(alive_agents) - 1) / max(1, len(realm.players))

        # 2. Travel together
        if player and hasattr(player, "pos"):
            nearby = []
            if hasattr(player, "pos") and player.pos is not None:
                px, py = player.pos if isinstance(player.pos, (tuple, list)) else (player.pos.x, player.pos.y)
                for aid, ally in realm.players.items():
                    if aid == agent_id or not getattr(ally, "resources", None):
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
        if player and hasattr(player, "history") and hasattr(player.history, "last_attacked"):
            target = player.history.last_attacked
            if target:
                for ally_id, ally in realm.players.items():
                    if ally_id != agent_id and hasattr(ally, "history") and getattr(ally.history, "last_attacked", None) == target:
                        coop_reward += self.coop_combat_weight
                        break

        # 4. Healing allies
        if player and player.resources.health_restore > 0:
            coop_reward += self.coop_heal_weight

        # 5. Trading items or gold
        if self._coop_state[agent_id]["traded"]:
            coop_reward += self.coop_trade_weight
            self._coop_state[agent_id]["traded"] = False

        # Combine all custom rewards
        custom_reward = healing_bonus + explore_bonus + coop_reward
        reward += custom_reward

        if custom_reward != 0:
            info.setdefault("custom_rewards", {}).update({
                "healing_bonus": healing_bonus,
                "explore_bonus": explore_bonus,
                "coop_reward": coop_reward,
                "total_custom": custom_reward,
            })

        return reward, terminated, truncated, info
