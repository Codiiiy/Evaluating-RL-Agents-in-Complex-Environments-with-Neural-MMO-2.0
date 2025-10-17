import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", action="store_true", help="Include Model_Size_MB in Total metric")
parser.add_argument("--graphs", action="store_true", help="Generate and save graphs")
args = parser.parse_args()

folder = "policies"
pt_folder = folder

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

json_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]

all_metrics = {}
for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)
    for model, metrics in data.items():
        if model not in all_metrics:
            all_metrics[model] = {}
        for key, val in metrics.items():
            if key not in all_metrics[model]:
                all_metrics[model][key] = []
            if isinstance(val, list):
                val = np.mean(val) if len(val) > 0 else np.nan
            all_metrics[model][key].append(val)

avg_metrics = {}
for model, metrics in all_metrics.items():
    avg_metrics[model] = {k: np.nanmean(v) for k, v in metrics.items()}

summary_records = []
for model, metrics in avg_metrics.items():
    record = {"Model": model}
    for category, keys in metric_categories.items():
        values = []
        for key in keys:
            val = metrics.get(key)
            if val is None:
                continue
            values.append(val)
        record[category] = np.nanmean(values) if values else np.nan
    pt_path = os.path.join(pt_folder, f"{model}.pt")
    record["Model_Size_MB"] = os.path.getsize(pt_path) / (1024*1024) if os.path.exists(pt_path) else np.nan
    summary_records.append(record)

df = pd.DataFrame(summary_records)
df = df.fillna(0)
df = df.round(6)
df = df.set_index("Model")

df["Length"] = df["Length"] / df["Length"].max()
df["Model_Size_MB"] = df["Model_Size_MB"] / df["Model_Size_MB"].max()

exclude_from_total = []
included_cols = [c for c in df.columns if c not in exclude_from_total]
if args.size:
    included_cols.append("Model_Size_MB")

df["Total"] = df[included_cols].sum(axis=1) / len(included_cols)

df = df.sort_values(by="Total", ascending=False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

print("\nMODEL PERFORMANCE SUMMARY\n")
print(df)

csv_output = "model_performance_summary.csv"
df.to_csv(csv_output)
print(f"\nSummary exported to: {csv_output}")

if args.graphs:
    output_folder = "graphs"
    os.makedirs(output_folder, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = df.reset_index()
    table_data_formatted = table_data.copy()
    for col in table_data_formatted.columns:
        if col != 'Model':
            table_data_formatted[col] = table_data_formatted[col].apply(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else x)
    
    table = ax.table(cellText=table_data_formatted.values,
                     colLabels=table_data_formatted.columns,
                     cellLoc='right',
                     loc='center',
                     colWidths=[0.12] + [0.07] * (len(table_data_formatted.columns) - 1))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_data_formatted.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data_formatted) + 1):
        for j in range(len(table_data_formatted.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Model Performance Summary Table', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    table_output = os.path.join(output_folder, "performance_summary_table.png")
    plt.savefig(table_output, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary table image exported to: {table_output}")

if args.graphs:
    output_folder = "graphs"
    os.makedirs(output_folder, exist_ok=True)

    baseline_model = "BaselinecnnMP"
    core_categories = ["Events", "Equip", "Skills", "Harvest", "Defeat", "Consume", "Economy"]

    if baseline_model in df.index:
        baseline_data = df.loc[baseline_model, core_categories]
        other_models = df.drop(baseline_model)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(core_categories))
        width = 0.15
        
        ax.bar(x - width, baseline_data, width, label=baseline_model, color='#2ca02c', alpha=0.9)
        
        for i, model in enumerate(other_models.index):
            offset = width * (i + 1)
            ax.bar(x + offset, other_models.loc[model, core_categories], width, label=model, alpha=0.7)
        
        ax.set_xlabel('Category')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'Performance Comparison: {baseline_model} vs Others')
        ax.set_xticks(x)
        ax.set_xticklabels(core_categories, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "baseline_vs_others.png"), dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        baseline_length = df.loc[baseline_model, "Length"]
        other_lengths = df.drop(baseline_model)["Length"]
        
        bars = ax.bar(range(len(df)), df["Length"], color=['#2ca02c' if idx == baseline_model else '#1f77b4' for idx in df.index])
        bars[list(df.index).index(baseline_model)].set_edgecolor('black')
        bars[list(df.index).index(baseline_model)].set_linewidth(2)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Normalized Episode Length (Timesteps)')
        ax.set_title(f'Episode Length: {baseline_model} Highlighted')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "length_baseline_focus.png"), dpi=300, bbox_inches='tight')
        plt.close()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        baseline_ticks = df.loc[baseline_model, "Ticks"]
        other_ticks = df.drop(baseline_model)["Ticks"]
        bars1 = ax1.bar(range(len(df)), df["Ticks"], color=['#2ca02c' if idx == baseline_model else '#ff7f0e' for idx in df.index])
        bars1[list(df.index).index(baseline_model)].set_edgecolor('black')
        bars1[list(df.index).index(baseline_model)].set_linewidth(2)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Normalized Ticks Value')
        ax1.set_title(f'Ticks Performance: {baseline_model} Highlighted')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        baseline_pvp = df.loc[baseline_model, "PvP"]
        other_pvp = df.drop(baseline_model)["PvP"]
        bars2 = ax2.bar(range(len(df)), df["PvP"], color=['#2ca02c' if idx == baseline_model else '#d62728' for idx in df.index])
        bars2[list(df.index).index(baseline_model)].set_edgecolor('black')
        bars2[list(df.index).index(baseline_model)].set_linewidth(2)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Normalized PvP Value')
        ax2.set_title(f'PvP Performance: {baseline_model} Highlighted')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df.index, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "ticks_pvp_baseline_focus.png"), dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(df)), df["Total"], color=['#2ca02c' if idx == baseline_model else '#9467bd' for idx in df.index])
        bars[list(df.index).index(baseline_model)].set_edgecolor('black')
        bars[list(df.index).index(baseline_model)].set_linewidth(3)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Total Score')
        ax.set_title(f'Total Performance Score: {baseline_model} Highlighted')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "total_baseline_focus.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"\nWarning: {baseline_model} not found in data. Generating standard graphs.")
        plt.figure(figsize=(14,6))
        df[core_categories].plot(kind="bar")
        plt.title("Model Performance by Core Categories")
        plt.ylabel("Normalized Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, "performance_core_categories.png"))
        plt.close()