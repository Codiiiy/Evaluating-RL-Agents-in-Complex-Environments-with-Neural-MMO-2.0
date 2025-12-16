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

        survival_bonus=0.05,
        explore_bonus_weight=0.15,
        clip_unique_event=5,

        harvest_bonus=0.8,
        fish_bonus=1.0,
        mine_bonus=1.2,
        sell_bonus=1.5,
        skill_bonus=0.8,
        consume_bonus=0.5,

        combat_bonus_weight=1.2,
        kill_bonus=10.0,
        damage_taken_penalty=-0.3,

        movement_penalty=-0.005,
        idle_penalty=-0.08,

        spawn_distance_weight=0.003,
        movement_diversity_weight=0.04,
        resource_proximity_weight=0.1,
        enemy_seek_weight=1.5,
        low_health_threshold=0.35,
        high_health_threshold=0.65,
        max_health_estimate=100.0,

        level_up_bonus=2.0,
        item_acquisition_bonus=0.6,
        equipment_bonus=1.2,
        death_penalty=-5.0,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)

        self.survival_bonus = survival_bonus
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event

        self.harvest_bonus = harvest_bonus
        self.fish_bonus = fish_bonus
        self.mine_bonus = mine_bonus
        self.sell_bonus = sell_bonus
        self.skill_bonus = skill_bonus
        self.consume_bonus = consume_bonus

        self.combat_bonus_weight = combat_bonus_weight
        self.kill_bonus = kill_bonus
        self.damage_taken_penalty = damage_taken_penalty

        self.movement_penalty = movement_penalty
        self.idle_penalty = idle_penalty

        self.spawn_distance_weight = spawn_distance_weight
        self.movement_diversity_weight = movement_diversity_weight
        self.resource_proximity_weight = resource_proximity_weight
        self.enemy_seek_weight = enemy_seek_weight
        self.low_health_threshold = low_health_threshold
        self.high_health_threshold = high_health_threshold
        self.max_health_estimate = max_health_estimate

        self.level_up_bonus = level_up_bonus
        self.item_acquisition_bonus = item_acquisition_bonus
        self.equipment_bonus = equipment_bonus
        self.death_penalty = death_penalty

        self._history = defaultdict(lambda: {
            "prev_damage_dealt": 0.0,
            "prev_damage_taken": 0.0,
            "prev_gold": 0.0,
            "prev_pos": None,
            "prev_moves": [],
            "spawn_pos": None,
            "prev_resources": {},
            "prev_skills": 0,
            "prev_consumed": 0,
            "prev_level": 0,
            "prev_inventory_size": 0,
            "prev_equipment_count": 0,
            "idle_count": 0,
            "steps_alive": 0,
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
                if len(hist) > 10:
                    hist.pop(0)
        except Exception:
            pass
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        realm = getattr(self.env, "realm", None)
        player = None
        if realm is not None:
            player = realm.players.get(agent_id, None)

        if player is None:
            return reward, terminated, truncated, info

        custom_reward = 0.0
        hist = self._history[agent_id]

        is_alive = getattr(player, "alive", False)
        if is_alive:
            hist["steps_alive"] += 1
            survival_mult = 1.0 + (hist["steps_alive"] / 1000.0)
            custom_reward += self.survival_bonus * min(survival_mult, 2.0)
        else:
            custom_reward += self.death_penalty

        uniq = self._unique_events.get(agent_id, None)
        if uniq and uniq.get("curr_count", 0) > uniq.get("prev_count", 0):
            delta = uniq["curr_count"] - uniq["prev_count"]
            custom_reward += min(delta, self.clip_unique_event) * self.explore_bonus_weight

        last_moves = hist.get("prev_moves", [])
        is_idle = not last_moves or last_moves[-1] in (None, 0)
        
        if is_idle:
            hist["idle_count"] += 1
            custom_reward += self.movement_penalty
            if hist["idle_count"] >= 3:
                custom_reward += self.idle_penalty
        else:
            hist["idle_count"] = 0

        if len(last_moves) >= 3:
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
            delta_dmg = dmg - prev_dmg
            custom_reward += delta_dmg * self.combat_bonus_weight
        hist["prev_damage_dealt"] = dmg

        prev_dmg_taken = float(hist.get("prev_damage_taken", 0.0))
        dmg_taken = prev_dmg_taken
        try:
            dmg_taken_val = getattr(getattr(player, "history", None), "damage_taken", None)
            if dmg_taken_val is not None:
                dmg_taken = float(dmg_taken_val)
        except Exception:
            dmg_taken = prev_dmg_taken

        if dmg_taken > prev_dmg_taken:
            delta_dmg_taken = dmg_taken - prev_dmg_taken
            custom_reward += delta_dmg_taken * self.damage_taken_penalty
        hist["prev_damage_taken"] = dmg_taken

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
            consumed = safe_get(res, "consumed")

            prev_resources = hist.get("prev_resources", {})

            delta_forage = max(0, forage - prev_resources.get("forage", 0))
            delta_fish = max(0, fish - prev_resources.get("fish", 0))
            delta_stone = max(0, stone - prev_resources.get("stone", 0))
            delta_ore = max(0, ore - prev_resources.get("ore", 0))
            delta_wood = max(0, wood - prev_resources.get("wood", 0))
            delta_gold = max(0, gold - prev_resources.get("gold", 0))
            delta_consumed = max(0, consumed - hist.get("prev_consumed", 0))

            if delta_forage > 0:
                custom_reward += delta_forage * self.harvest_bonus
            if delta_fish > 0:
                custom_reward += delta_fish * self.fish_bonus
            if delta_stone > 0:
                custom_reward += delta_stone * self.mine_bonus
            if delta_ore > 0:
                custom_reward += delta_ore * self.mine_bonus * 1.5
            if delta_wood > 0:
                custom_reward += delta_wood * self.mine_bonus
            if delta_gold > 0:
                custom_reward += delta_gold * self.sell_bonus
            if delta_consumed > 0:
                custom_reward += delta_consumed * self.consume_bonus

            hist["prev_resources"] = {
                "forage": forage,
                "fish": fish,
                "stone": stone,
                "ore": ore,
                "wood": wood,
                "gold": gold,
            }
            hist["prev_consumed"] = consumed

        prev_skills = hist.get("prev_skills", 0)
        skills_used = 0
        try:
            skills_used = getattr(getattr(player, "history", None), "skills_used", 0)
        except Exception:
            skills_used = prev_skills
        delta_skills = skills_used - prev_skills
        if delta_skills > 0:
            custom_reward += delta_skills * self.skill_bonus
        hist["prev_skills"] = skills_used

        level = 0
        try:
            level = safe_get(res, "level") if res else 0
        except Exception:
            level = 0
        
        prev_level = hist.get("prev_level", 0)
        if level > prev_level:
            custom_reward += (level - prev_level) * self.level_up_bonus
        hist["prev_level"] = level

        inventory = getattr(player, "inventory", None)
        if inventory is not None:
            try:
                inv_size = len([item for item in inventory if item is not None])
                prev_inv_size = hist.get("prev_inventory_size", 0)
                if inv_size > prev_inv_size:
                    custom_reward += (inv_size - prev_inv_size) * self.item_acquisition_bonus
                hist["prev_inventory_size"] = inv_size
            except Exception:
                pass

        equipped_count = 0
        try:
            if inventory is not None:
                equipped_count = len([item for item in inventory if getattr(item, "equipped", False)])
            prev_equipped = hist.get("prev_equipment_count", 0)
            if equipped_count > prev_equipped:
                custom_reward += (equipped_count - prev_equipped) * self.equipment_bonus
            hist["prev_equipment_count"] = equipped_count
        except Exception:
            pass

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

        if nearest_enemy_dist is not None and nearest_enemy_dist > 0:
            proximity_factor = 1.0 / (1.0 + nearest_enemy_dist)
            
            if health_prop >= self.high_health_threshold:
                if moving_toward_enemy:
                    approach_bonus = proximity_factor * self.enemy_seek_weight
                    custom_reward += approach_bonus
                else:
                    custom_reward -= proximity_factor * self.enemy_seek_weight * 0.3
            elif health_prop <= self.low_health_threshold:
                if moving_toward_enemy:
                    retreat_penalty = -proximity_factor * self.enemy_seek_weight * 1.5
                    custom_reward += retreat_penalty
                else:
                    away_bonus = proximity_factor * self.enemy_seek_weight * 0.8
                    custom_reward += away_bonus
            else:
                if moving_toward_enemy and nearest_enemy_dist < 5:
                    custom_reward += proximity_factor * self.enemy_seek_weight * 0.5

        recently_gained_resource = False
        prev_resources = hist.get("prev_resources", {})
        for k in ["forage", "fish", "stone", "ore", "wood"]:
            if k in prev_resources and prev_resources.get(k, 0) > 0:
                recently_gained_resource = True
                break
        
        if recently_gained_resource and not is_idle:
            custom_reward += self.resource_proximity_weight

        hist["prev_pos"] = pos

        reward += custom_reward

        if custom_reward != 0:
            info.setdefault("custom_rewards", {})["meta_v2"] = custom_reward

        return reward, terminated, truncated, info