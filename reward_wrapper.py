from reinforcement_learning.stat_wrapper import BaseStatWrapper

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
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix
        self.heal_bonus_weight = heal_bonus_weight
        self.explore_bonus_weight = explore_bonus_weight
        self.clip_unique_event = clip_unique_event
        self.movement_penalty = movement_penalty
        self.combat_bonus_weight = combat_bonus_weight

    def reset(self, **kwargs):
        self._reset_reward_vars()
        return super().reset(**kwargs)

    def _reset_reward_vars(self):
        self._history = {
            agent_id: {
                "prev_price": 0,
                "prev_moves": [],
                "prev_health": 0,
                "prev_damage_dealt": 0,
            }
            for agent_id in self.env.possible_agents
        }

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
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        realm = self.env.realm
        custom_reward = 0
        healing_bonus = 0
        if self.heal_bonus_weight > 0 and agent_id in realm.players:
            player = realm.players[agent_id]
            if player.resources.health_restore > 0:
                healing_bonus = self.heal_bonus_weight * player.resources.health_restore
        explore_bonus = 0
        if self.explore_bonus_weight > 0:
            uniq = self._unique_events[agent_id]
            if uniq["curr_count"] > uniq["prev_count"]:
                event_delta = min(self.clip_unique_event, uniq["curr_count"] - uniq["prev_count"])
                explore_bonus = event_delta * self.explore_bonus_weight
        movement_penalty = 0
        if self.movement_penalty > 0:
            recent_moves = self._history[agent_id]["prev_moves"]
            if len(recent_moves) >= 5:
                last_moves = recent_moves[-5:]
                if len(set(last_moves)) <= 2:
                    movement_penalty = -self.movement_penalty
        combat_bonus = 0
        if self.combat_bonus_weight > 0 and agent_id in realm.players:
            player = realm.players[agent_id]
            if hasattr(player, 'history') and hasattr(player.history, 'damage'):
                curr_damage = player.history.damage
                prev_damage = self._history[agent_id]["prev_damage_dealt"]
                if curr_damage > prev_damage:
                    combat_bonus = self.combat_bonus_weight * (curr_damage - prev_damage)
                self._history[agent_id]["prev_damage_dealt"] = curr_damage
        custom_reward = healing_bonus + explore_bonus + movement_penalty + combat_bonus
        reward += custom_reward
        if custom_reward != 0:
            if "custom_rewards" not in info:
                info["custom_rewards"] = {}
            info["custom_rewards"].update({
                "healing_bonus": healing_bonus,
                "explore_bonus": explore_bonus,
                "movement_penalty": movement_penalty,
                "combat_bonus": combat_bonus,
                "total_custom": custom_reward,
            })
        return reward, terminated, truncated, info

class MinimalRewardWrapper(BaseStatWrapper):
    def __init__(
        self,
        env,
        eval_mode=False,
        early_stop_agent_num=0,
        stat_prefix=None,
        use_custom_reward=False,
    ):
        super().__init__(env, eval_mode, early_stop_agent_num, stat_prefix, use_custom_reward)
        self.stat_prefix = stat_prefix

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def observation(self, agent_id, agent_obs):
        return agent_obs

    def action(self, agent_id, agent_atn):
        return agent_atn

    def reward_terminated_truncated_info(self, agent_id, reward, terminated, truncated, info):
        return reward, terminated, truncated, info
