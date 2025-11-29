from reinforcement_learning.stat_wrapper import BaseStatWrapper
import numpy as np
from collections import defaultdict
import math

class RewardWrapper(BaseStatWrapper):

    def __init__(
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=True,

        survival_bonus=0.02,
        explore_bonus_weight=0.06,
        clip_unique_event=3,

        harvest_bonus=0.35,
        fish_bonus=0.45,
        mine_bonus=0.65,
        sell_bonus=0.6,

        combat_bonus_weight=0.5,
        kill_bonus=6.0,

        movement_penalty=-0.01,

        spawn_distance_weight=0.001,
        movement_diversity_weight=0.02,
        resource_proximity_weight=0.05,
        enemy_seek_weight=0.8,
        low_health_threshold=0.3,
        high_health_threshold=0.6,
        max_health_estimate=100.0,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)

        self.survival_bonus = survival_bonus
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

        self.harvest_bonus = harvest_bonus
        self.fish_bonus = fish_bonus
        self.mine_bonus = mine_bonus
        self.sell_bonus = sell_bonus
        
        self.combat_bonus_weight = combat_bonus_weight
        self.kill_bonus = kill_bonus

        self.movement_penalty = movement_penalty

        self.spawn_distance_weight = spawn_distance_weight
        self.movement_diversity_weight = movement_diversity_weight
        self.resource_proximity_weight = resource_proximity_weight
        self.enemy_seek_weight = enemy_seek_weight
        self.low_health_threshold = low_health_threshold
        self.high_health_threshold = high_health_threshold
        self.max_health_estimate = max_health_estimate

        self._history = defaultdict(lambda: {
            "prev_damage_dealt": 0.0,
            "prev_gold": 0.0,
            "prev_pos": None,
            "prev_moves": [],         
            "spawn_pos": None,       
            "prev_resources": {},    
        })

    def reset(self, **kwargs):
        self._history.clear()
        return super().reset(**kwargs)

    def action(self, agent_id, agent_atn):
        try:
            if "Move" in agent_atn and "Direction" in agent_atn["Move"]:
                dirv = agent_atn["Move"]["Direction"]
                hist = self._history[agent_id]["prev_moves"]
                hist.append(dirv)
                if len(hist) > 8:
                    hist.pop(0)
        except Exception:
            pass
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        """
        Main reward function. Compose base solo rewards + smarter-roaming bonuses.
        Defensive checks so missing attributes don't crash.
        """
        realm = getattr(self.env, "realm", None)
        player = None
        if realm is not None:
            player = realm.players.get(agent_id, None)

        if player is None:
            return reward, terminated, truncated, info

        custom_reward = 0.0
        hist = self._history[agent_id]

        if getattr(player, "alive", False):
            custom_reward += self.survival_bonus

        uniq = self._unique_events.get(agent_id, None)
        if uniq and uniq.get("curr_count", 0) > uniq.get("prev_count", 0):
            delta = uniq["curr_count"] - uniq["prev_count"]
            custom_reward += min(delta, self.clip_unique_event) * self.explore_bonus_weight

        last_moves = hist.get("prev_moves", [])
        if not last_moves or last_moves[-1] in (None, 0):
            custom_reward += self.movement_penalty

        if len(last_moves) >= 2:
            distinct = len(set(last_moves))
            diversity = distinct / len(last_moves)
            custom_reward += diversity * self.movement_diversity_weight

        pos = None
        try:
            pos_attr = getattr(player, "pos", None)
            if pos_attr is not None:
                pos = (pos_attr.x, pos_attr.y) if not isinstance(pos_attr, (tuple, list)) else tuple(pos_attr)
        except Exception:
            pos = None

        if hist["spawn_pos"] is None and pos is not None:
            hist["spawn_pos"] = pos

        if pos is not None and hist["spawn_pos"] is not None:
            dx = pos[0] - hist["spawn_pos"][0]
            dy = pos[1] - hist["spawn_pos"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            custom_reward += dist * self.spawn_distance_weight

        prev_dmg = float(hist.get("prev_damage_dealt", 0.0))
        dmg = prev_dmg
        try:
            dmg_val = getattr(getattr(player, "history", None), "damage_dealt", None)
            if dmg_val is not None:
                dmg = float(dmg_val)
        except Exception:
            dmg = prev_dmg

        if dmg > prev_dmg:
            custom_reward += (dmg - prev_dmg) * self.combat_bonus_weight
        hist["prev_damage_dealt"] = dmg

        if getattr(getattr(player, "history", None), "last_kill", None):
            custom_reward += self.kill_bonus

        res = getattr(player, "resources", None)
        if res is not None:
            def safe_get(rsrc, name):
                try:
                    return getattr(rsrc, name, 0)
                except Exception:
                    return 0

            forage = safe_get(res, "forage")
            fish = safe_get(res, "fish")
            stone = safe_get(res, "stone")
            ore = safe_get(res, "ore")
            wood = safe_get(res, "wood")
            gold = safe_get(res, "gold")

            prev_resources = hist.get("prev_resources", {})

            delta_forage = max(0, forage - prev_resources.get("forage", 0))
            delta_fish = max(0, fish - prev_resources.get("fish", 0))
            delta_stone = max(0, stone - prev_resources.get("stone", 0))
            delta_ore = max(0, ore - prev_resources.get("ore", 0))
            delta_wood = max(0, wood - prev_resources.get("wood", 0))
            delta_gold = max(0, gold - prev_resources.get("gold", 0))

            if delta_forage > 0:
                custom_reward += delta_forage * self.harvest_bonus
            if delta_fish > 0:
                custom_reward += delta_fish * self.fish_bonus
            mined_total = delta_stone + delta_ore + delta_wood
            if mined_total > 0:
                custom_reward += mined_total * self.mine_bonus
            if delta_gold > 0:
                custom_reward += self.sell_bonus * (1.0 if delta_gold > 0 else 0.0)

            hist["prev_resources"] = {
                "forage": forage,
                "fish": fish,
                "stone": stone,
                "ore": ore,
                "wood": wood,
                "gold": gold,
            }

        health = None
        max_health = None
        if res is not None:
            try:
                health = safe_get(res, "health")
                max_health = safe_get(res, "max_health")
            except Exception:
                health = None
                max_health = None

        if max_health is None or max_health == 0:
            max_health = self.max_health_estimate

        if health is None:
            health_prop = 1.0
        else:
            try:
                health_prop = float(health) / float(max_health)
            except Exception:
                health_prop = 1.0

        nearest_enemy_dist = None
        nearest_enemy_pos = None
        try:
            if realm is not None and hasattr(realm, "players") and pos is not None:
                px, py = pos
                dmin = None
                for aid, other in realm.players.items():
                    if aid == agent_id:
                        continue
                    if other is None or not getattr(other, "alive", True):
                        continue
                    other_pos_attr = getattr(other, "pos", None)
                    if other_pos_attr is None:
                        continue
                    ox, oy = (other_pos_attr.x, other_pos_attr.y) if not isinstance(other_pos_attr, (tuple, list)) else tuple(other_pos_attr)
                    d = math.hypot(ox - px, oy - py)
                    if dmin is None or d < dmin:
                        dmin = d
                        nearest_enemy_pos = (ox, oy)
                nearest_enemy_dist = dmin
        except Exception:
            nearest_enemy_dist = None
            nearest_enemy_pos = None

        moving_toward_enemy = False
        prev_pos = hist.get("prev_pos", None)
        if nearest_enemy_pos is not None and pos is not None and prev_pos is not None:
            try:
                pe_dist_now = math.hypot(nearest_enemy_pos[0] - pos[0], nearest_enemy_pos[1] - pos[1])
                pe_dist_prev = math.hypot(nearest_enemy_pos[0] - prev_pos[0], nearest_enemy_pos[1] - prev_pos[1])
                moving_toward_enemy = pe_dist_now < pe_dist_prev
            except Exception:
                moving_toward_enemy = False

        if nearest_enemy_dist is not None:
            if health_prop >= self.high_health_threshold:
                if moving_toward_enemy:
                    approach_bonus = (1.0 / (1.0 + nearest_enemy_dist)) * self.enemy_seek_weight
                    custom_reward += approach_bonus
            elif health_prop <= self.low_health_threshold:
                if moving_toward_enemy:
                    retreat_penalty = -(1.0 / (1.0 + nearest_enemy_dist)) * self.enemy_seek_weight
                    custom_reward += retreat_penalty
                else:
                    away_bonus = (1.0 / (1.0 + nearest_enemy_dist)) * (self.enemy_seek_weight * 0.5)
                    custom_reward += away_bonus

        recently_gained_any = False
        prev_resources = hist.get("prev_resources", {})
        for k, v in prev_resources.items():
            if v and v > 0:
                recently_gained_any = True
                break
        if recently_gained_any and last_moves:
            custom_reward += self.resource_proximity_weight

        hist["prev_pos"] = pos

        reward += custom_reward

        if custom_reward != 0:
            info.setdefault("custom_rewards", {})["smarter_roam"] = custom_reward

        return reward, terminated, truncated, info