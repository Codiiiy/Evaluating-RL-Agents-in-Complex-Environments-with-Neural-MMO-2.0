import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

json_path = "baselinecnn_policies/eval_pvp_1.json"

metric_categories = {
    "Length": ["length"],
    "Events": [
        "curriculum/Task_CountEvent_(event:EARN_GOLD_N:20)_reward_to:agent",
        "curriculum/Task_CountEvent_(event:GO_FARTHEST_N:64)_reward_to:agent",
        "curriculum/Task_CountEvent_(event:PLAYER_KILL_N:20)_reward_to:agent"
    ],
    "Equip": [
        "curriculum/Task_EquipItem_(item:Wand_level:1_num_agent:1)_reward_to:agent",
        "curriculum/Task_EquipItem_(item:Wand_level:3_num_agent:1)_reward_to:agent"
    ],
    "Fully Armed": [
        "curriculum/Task_FullyArmed_(combat_style:Mage_level:3_num_agent:1)_reward_to:agent",
        "curriculum/Task_FullyArmed_(combat_style:Melee_level:3_num_agent:1)_reward_to:agent",
        "curriculum/Task_FullyArmed_(combat_style:Range_level:3_num_agent:1)_reward_to:agent"
    ],
    "Skills": [
        "curriculum/Task_AttainSkill_(skill:Melee_level:10_num_agent:1)_reward_to:agent",
        "curriculum/Task_AttainSkill_(skill:Mage_level:10_num_agent:1)_reward_to:agent",
        "curriculum/Task_AttainSkill_(skill:Fishing_level:10_num_agent:1)_reward_to:agent"
    ],
    "Harvest": [
        "curriculum/Task_HarvestItem_(item:Whetstone_level:1_quantity:20)_reward_to:agent",
        "curriculum/Task_HarvestItem_(item:Runes_level:3_quantity:20)_reward_to:agent"
    ],
    "Defeat": [
        "curriculum/Task_DefeatEntity_(agent_type:npc_level:1_num_agent:20)_reward_to:agent",
        "curriculum/Task_DefeatEntity_(agent_type:npc_level:3_num_agent:20)_reward_to:agent"
    ],
    "Consume": [
        "curriculum/Task_ConsumeItem_(item:Potion_level:1_quantity:20)_reward_to:agent",
        "curriculum/Task_ConsumeItem_(item:Potion_level:3_quantity:20)_reward_to:agent"
    ],
    "Economy": [
        "curriculum/Task_MakeProfit_(amount:100)_reward_to:agent",
        "curriculum/Task_EarnGold_(amount:100)_reward_to:agent",
        "curriculum/Task_HoardGold_(amount:100)_reward_to:agent"
    ],
    "Ticks": ["curriculum/Task_TickGE_(num_tick:1024)_reward_to:agent"],
    "PvP": ["curriculum/Task_CountEvent_(event:PLAYER_KILL_N:20)_reward_to:agent"]
}

with open(json_path, "r") as f:
    data = json.load(f)

summary_records = []
for model, metrics in data.items():
    record = {"Model": model}
    for category, keys in metric_categories.items():
        values = []
        for key in keys:
            val = metrics.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                val = np.mean(val) if len(val) > 0 else np.nan
            values.append(val)
        record[category] = np.nanmean(values) if values else np.nan
    summary_records.append(record)

df = pd.DataFrame(summary_records)
df = df.fillna(0)
df = df.round(4)
df = df.set_index("Model")

df["Length"] = df["Length"] / df["Length"].max()
df["Total"] = df.drop(columns=["Length"]).sum(axis=1) / 5
df = df.sort_values(by="Total", ascending=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print("\nMODEL PERFORMANCE SUMMARY\n")
print(df)

df.plot(kind="bar", figsize=(14, 6), title="Model Comparison by Metric Category (Normalized Length)")
plt.ylabel("Average Reward / Value")
plt.tight_layout()
plt.show()
