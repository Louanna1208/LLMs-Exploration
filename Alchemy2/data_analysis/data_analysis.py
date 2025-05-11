# Step 1: Import necessary libraries and data processing module
import json
import numpy as np
import os
os.chdir(r'/Github/LLMs_game/Alchemy2')
import pandas as pd
from data_analysis.data_process import LLM_data_preprocessing, Human_data_preprocessing
from data_analysis.data_process import element_name_to_code, element_code_to_name
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import ttest_ind


# Step 2: Call data_processing function to load and process data
Base_LLM_data_path = 'output/data/base_gpt-4o-2024-08-06_500_results.json'
Base_LLM_with_names, Base_LLM_with_codes = LLM_data_preprocessing(Base_LLM_data_path,'gpt-4o')

#save_path = 'output/data/base_LLM_with_names.csv'
#Base_LLM_with_names.to_csv(save_path, index=False)
human_data_path = 'dataset/alchemy2HumanData.csv'
Human_with_names, Human_with_codes = Human_data_preprocessing(human_data_path)

print(Base_LLM_with_codes.columns)
print(Base_LLM_with_codes.head())
print(Human_with_codes.columns)
len(Human_with_codes['id'].unique())

# load o1 results
o1_data_path = 'output/data/base_o1-2024-12-17_500_results.json'
o1_LLM_with_names, o1_LLM_with_codes = LLM_data_preprocessing(o1_data_path,'gpt-4o')
#save_path = 'output/data/o1_LLM_with_names_and_codes.csv'
#o1_LLM_with_names.to_csv(save_path, index=False)

# prompt engineering LLM data
prompt_engineering_LLM_data_path = 'output/data/prompt_engineering_gpt-4o-2024-08-06_500_results.json'
prompt_engineering_LLM_with_names, prompt_engineering_LLM_with_codes = LLM_data_preprocessing(prompt_engineering_LLM_data_path,'gpt-4o')

# Load deepseek-reasoner data
reasoning_LLM_data_path = 'output/data/reasoning_deepseek-reasoner_500_results.json'
deepseek_reasoner_LLM_with_names, deepseek_reasoner_LLM_with_codes = LLM_data_preprocessing(reasoning_LLM_data_path,'gpt-4o')
#save_path = 'output/data/deepseek_reasoner_LLM_with_names_and_codes.csv'
#deepseek_reasoner_LLM_with_names.to_csv(save_path, index=False)

#Llama3 data
Llama3_70B_LLM_data_path = 'output/data/Llama3_70B_500_results.json'
Llama3_70B_LLM_with_names, Llama3_70B_LLM_with_codes = LLM_data_preprocessing(Llama3_70B_LLM_data_path,'Llama3')
#save_path = 'output/data/Llama3_70B_LLM_with_names_and_codes.csv'
#Llama3_70B_LLM_with_names.to_csv(save_path, index=False)

Llama3_8B_LLM_data_path = 'output/data/Llama3_8B_500_results.json'
Llama3_8B_LLM_with_names, Llama3_8B_LLM_with_codes = LLM_data_preprocessing(Llama3_8B_LLM_data_path,'Llama3')
#save_path = 'output/data/Llama3_8B_LLM_with_names_and_codes.csv'
#Llama3_8B_LLM_with_names.to_csv(save_path, index=False)

# Calculate the percentage of players with final_trial <=500, it should be 0.912
final_trial = Human_with_codes.groupby('id')['trial'].max()
final_trial_500 = len(final_trial[final_trial <= 500])/len(final_trial)
print(final_trial_500)

# ten human best player at 500 trials
print(Human_with_codes[Human_with_codes['trial'] == 500].sort_values(by='inventory', ascending=False).head(10))

def fill_missing_inventory(df, trial_num):
    filled_dfs = []
    for uid, user_df in df.groupby('id'):
        user_df = user_df.set_index('trial').sort_index()
        # create a full index
        full_index = pd.Index(range(trial_num), name='trial')
        # rebuild the DataFrame, fill the missing
        user_df = user_df.reindex(full_index)
        user_df['id'] = uid
        # fill the missing inventory with the previous value, if the first item is still missing, fill it with 4
        user_df['inventory'] = user_df['inventory'].ffill().fillna(4)
        filled_dfs.append(user_df.reset_index())
    return pd.concat(filled_dfs, ignore_index=True)

#------------------------------------------------------------------------------------------------
# figure 1: plot the final inventory distribution
#------------------------------------------------------------------------------------------------
def final_trial_inventory(data, trials=500, data_name='Human'):
    if data_name == 'Human':
        data = data[data['trial'] <= 500]
    grouped_data = data.groupby('id')
    n_players = grouped_data.ngroups
    # color mapping for different LLM models
    colors = {
        'Human': '#A51C36',
        'GPT-4o': '#7ABBDB',
        'LLaMA3.1-70B': '#DBB428',
        'LLaMA3.1-8B': '#84BA42',
        'o1': '#682478',
        'GPT-4o(prompt-engineering)': '#4485C7',
        'LLaMA3.1_70B_intervention': '#FFEE6F',
        'DeepSeek-R1': '#bcfce7'
    }
    # get inventory sizes
    inventory_sizes = grouped_data['inventory'].max()
    if data_name == 'Human':
        ax = sns.histplot(data=inventory_sizes, kde=False, log_scale=True, bins=10, color=colors[data_name])
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
    else:
        ax = sns.histplot(data=inventory_sizes, kde=False, bins=10, color=colors[data_name])
        ticks = np.arange(0, inventory_sizes.max(), 10)  # Manually set x-axis tick positions
        ax.set_xticks(ticks)
        plt.minorticks_off()

    # plot mean
    ax.axvline(x=inventory_sizes.mean(), ls='dashed', color='#444444', linewidth=1)
    print('Maximum inventory size: {}'.format(inventory_sizes.max()))
    print('Minimum inventory size: {}'.format(inventory_sizes.min()))
    print('Mean inventory size: {}'.format(inventory_sizes.mean()))
    print('Confidence Intervals:{}'.format(sm.stats.DescrStatsW(inventory_sizes).tconfint_mean(alpha = 0.05)))


    # Set plot
    plt.title(f'{data_name} players\' final inventory distribution', fontsize=16)
    if data_name == 'Human':
        plt.ylim(bottom=0, top=9000)
    else:
        plt.ylim(bottom=0, top=15)
    plt.xlim(left=inventory_sizes.min(), right=inventory_sizes.max()*1.2)
    # set x-tick labels
    plt.xticks(fontsize=14)
    # set y-tick labels
    plt.yticks(fontsize=14)
    plt.xlabel('Final inventory', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    # add label for mean
    _, max_ylim = plt.ylim()
    # round mean to the nearest integer
    mean_rounded = int(round(inventory_sizes.mean()))
    plt.text(inventory_sizes.mean()*1.3, max_ylim*0.85, 'Mean:\n{:.1f}'.format(mean_rounded), fontsize=14)
    plt.tight_layout()
    # save figure
    #plt.savefig(f'output/picture/{data_name}_players_final_inventory_distribution.png', dpi=300)
    # save figure as pdf for latex
    #plt.savefig(f'output/picture/{data_name}_players_final_inventory_distribution.pdf', format='pdf',dpi=300)
    plt.show()
    return inventory_sizes

human_inventory = final_trial_inventory(Human_with_codes, data_name='Human')
gpt4o_inventory = final_trial_inventory(Base_LLM_with_codes, data_name='GPT-4o')
Llama3_70B_inventory = final_trial_inventory(Llama3_70B_LLM_with_codes, data_name='LLaMA3.1-70B')
Llama3_8B_inventory = final_trial_inventory(Llama3_8B_LLM_with_codes, data_name='LLaMA3.1-8B')
Base_LLM_best_temperature = Base_LLM_with_codes[Base_LLM_with_codes['temperature'] == 1.0]
best_temperature_inventory = final_trial_inventory(Base_LLM_best_temperature, data_name='GPT-4o')
gpt4o_prompt_engineering_inventory = final_trial_inventory(prompt_engineering_LLM_with_codes, data_name = 'GPT-4o(prompt-engineering)')
o1_inventory = final_trial_inventory(o1_LLM_with_codes, data_name = 'o1')
deepseek_reasoner_inventory = final_trial_inventory(deepseek_reasoner_LLM_with_codes, data_name = 'DeepSeek-R1')

# t-test between human and gpt-4o
# Perform t-tests
def perform_t_test(group1, group2, group1_name, group2_name):
    t_stat, p_value = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
    print(f"T-test between {group1_name} and {group2_name}:")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Result: Significant difference (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    print("\n")

# Conduct t-tests
perform_t_test(gpt4o_inventory, human_inventory, 'GPT-4o', 'Human')
perform_t_test(Llama3_70B_inventory, human_inventory, 'LLaMA3.1-70B', 'Human')
perform_t_test(Llama3_8B_inventory, human_inventory, 'LLaMA3.1-8B', 'Human')
perform_t_test(Llama3_8B_inventory, Llama3_70B_inventory, 'LLaMA3.1-8B', 'LLaMA3.1-70B')
perform_t_test(gpt4o_inventory, Llama3_70B_inventory, 'GPT-4o', 'LLaMA3.1-70B')
perform_t_test(gpt4o_inventory, gpt4o_prompt_engineering_inventory, 'GPT-4o', 'GPT-4o(prompt-engineering)')
perform_t_test(o1_inventory, human_inventory, 'o1', 'Human')
perform_t_test(deepseek_reasoner_inventory, human_inventory, 'DeepSeek-R1', 'Human')

# Save inventory sizes to CSV (optional)
#gpt4o_inventory.to_csv('output/data/gpt4o_inventory_sizes.csv', index=False)
#Llama3_70B_inventory.to_csv('output/data/Llama3_70B_inventory_sizes.csv', index=False)
#Llama3_8B_inventory.to_csv('output/data/Llama3_8B_inventory_sizes.csv', index=False)
#human_inventory.to_csv('output/data/Human_inventory_sizes.csv', index=False)

#------------------------------------------------------------------------------------------------
# figure 2: plot the best temperature for each model and compare with human
#------------------------------------------------------------------------------------------------
def plot_best_temperature_for_each_model(Base_LLM_with_codes,Llama3_70B_LLM_with_codes,Llama3_8B_LLM_with_codes,prompt_engineering_LLM_with_codes,deepseek_reasoner_LLM_with_codes,o1_LLM_with_codes, Human_with_codes, trial_num=500):
    # best temperature is the temperature=1.0
    Base_LLM_best_temperature = fill_missing_inventory(Base_LLM_with_codes[Base_LLM_with_codes['temperature'] == 1.0], trial_num)
    Llama3_70B_LLM_best_temperature = fill_missing_inventory(Llama3_70B_LLM_with_codes[Llama3_70B_LLM_with_codes['temperature'] == 1.0], trial_num)
    Llama3_8B_LLM_best_temperature = fill_missing_inventory(Llama3_8B_LLM_with_codes[Llama3_8B_LLM_with_codes['temperature'] == 1.0], trial_num)
    #plot the average inventory of each model
    plt.figure(figsize=(8, 6))
    plt.plot(Llama3_8B_LLM_best_temperature.groupby('trial')['inventory'].mean(), label='LLaMA3.1-8B(temp = 1.0)', color='#84BA42', linewidth=2)
    plt.plot(Llama3_70B_LLM_best_temperature.groupby('trial')['inventory'].mean(), label='LLaMA3.1-70B(temp = 1.0)', color='#DBB428', linewidth=2)
    plt.plot(Base_LLM_best_temperature.groupby('trial')['inventory'].mean(), label='GPT-4o(temp = 1.0)', color='#7ABBDB', linewidth=2)
    plt.plot(prompt_engineering_LLM_with_codes.groupby('trial')['inventory'].mean(), label='GPT-4o(prompt-engineering)', color='#4485C7', linewidth=2)
    plt.plot(o1_LLM_with_codes.groupby('trial')['inventory'].mean(), label='o1', color='#682478', linewidth=2)
    plt.plot(Human_with_codes.groupby('trial')['inventory'].mean()[:trial_num], label='Human', color='#A51C36', linewidth=2)
    #plt.plot(deepseek_reasoner_LLM_with_codes.groupby('trial')['inventory'].mean()[:trial_num], label='DeepSeek-R1', color='#bcfce7', linewidth=2)
    plt.title('Best Temperatures of Each Model and Human Performance', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Average Inventory', fontsize=16)
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    #plt.savefig(f'output/picture/best_temperature_comparison(DeepSeek-R1)_full.png', dpi=300)
    plt.savefig(f'output/picture/best_temperature_comparison(prompt-engineering)_full.png',dpi=300)
    plt.show()

plot_best_temperature_for_each_model(Base_LLM_with_codes,Llama3_70B_LLM_with_codes,Llama3_8B_LLM_with_codes,prompt_engineering_LLM_with_codes,deepseek_reasoner_LLM_with_codes,o1_LLM_with_codes, Human_with_codes, trial_num=500)

#------------------------------------------------------------------------------------------------
# figure 3: plot the average inventory of each model's temperature
#------------------------------------------------------------------------------------------------
def plot_average_inventory_of_each_model_temperature(model_name):
    # set the trial number
    trial_num = 500
    colors = {0: '#D75B4E', 0.3: '#2D8875', 0.7: '#EEB6D4', 1.0: '#B5CE4E'}
    plt.figure(figsize=(8, 6))
    # get the data for each temperature
    if model_name == 'GPT-4o':
        tem_group = Base_LLM_with_codes.groupby('temperature')
    elif model_name == 'LLaMA3.1-70B':
        tem_group = Llama3_70B_LLM_with_codes.groupby('temperature')
    elif model_name == 'LLaMA3.1-8B':
        tem_group = Llama3_8B_LLM_with_codes.groupby('temperature')
    for temperature in tem_group.groups:
        temp_data = tem_group.get_group(temperature) # get the data for each temperature
        # get the each temperature's final inventory max, min, mean, and confidence intervals
        final_inventory = temp_data.groupby('id')['inventory'].max()
        print(f'temperature = {temperature} Maximum inventory size: {final_inventory.max()}')   
        print(f'temperature = {temperature} Minimum inventory size: {final_inventory.min()}')
        print(f'temperature = {temperature} Mean inventory size: {final_inventory.mean()}')
        print(f'temperature = {temperature} Confidence Intervals:{sm.stats.DescrStatsW(final_inventory).tconfint_mean(alpha = 0.05)}')
        temp_data = fill_missing_inventory(temp_data, trial_num)
        temp_data = temp_data.groupby('trial')['inventory'].mean() # get the mean inventory for each trial
        color = colors[temperature]
        plt.plot(temp_data, label=f'{model_name}(temp = {temperature})', color=color)
    # deal the missing data
    deepseek_reasoner = fill_missing_inventory(deepseek_reasoner_LLM_with_codes, trial_num)
    o1 = fill_missing_inventory(o1_LLM_with_codes, trial_num)
    # plot the deepseek-reasoner data
    plt.plot(deepseek_reasoner.groupby('trial')['inventory'].mean()[:trial_num], label='DeepSeek-R1', color='#bcfce7', linewidth=2)
    # plot the human data
    plt.plot(Human_with_codes.groupby('trial')['inventory'].mean()[:trial_num], label='Human', color='#A51C36', linewidth=2)
    # plot the o1 data
    plt.plot(o1.groupby('trial')['inventory'].mean()[:trial_num], label='o1', color='#682478', linewidth=2)

    #plt.title(f'Human and o1 with {model_name} different temperatures comparison', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Average Inventory', fontsize=16)
    # set x-tick labels
    plt.xticks(fontsize=14)
    # set y-tick labels
    plt.yticks(fontsize=14)
    plt.xlim(0, 500)
    plt.ylim(0, 200)
    plt.legend(fontsize = 12)
    plt.tight_layout()
    plt.savefig(f'output/picture/{model_name}_different_temperature_comparison_full.png', dpi=300)
    # save figure as pdf for latex
    #plt.savefig(f'output/picture/{model_name}_different_temperature_comparison.pdf', format='pdf',dpi=300)
    plt.show()

plot_average_inventory_of_each_model_temperature('GPT-4o')
plot_average_inventory_of_each_model_temperature('LLaMA3.1-70B')
plot_average_inventory_of_each_model_temperature('LLaMA3.1-8B')

#------------------------------------------------------------------------------------------------
# figure 4: plot the final inventory distribution density
#------------------------------------------------------------------------------------------------
def final_trial_inventory_density(data, trials=500, data_name='Human', ax=None):
    if data_name == 'Human':
        data = data[data['trial'] <= trials]

    grouped_data = data.groupby('id')
    inventory_sizes = grouped_data['inventory'].max()

    # Color mapping for different models
    colors = {
        'LLaMA3.1-8B': '#84BA42',
        'LLaMA3.1-70B': '#DBB428',
        'GPT-4o': '#7ABBDB',
        'Human': '#A51C36',
        'DeepSeek-R1': '#bcfce7',
        'o1': '#682478' 
    }
    
    if data_name not in colors:
        raise ValueError(f"Color for {data_name} is not defined in the colors dictionary.")
    
    if ax is None:
        ax = plt.gca()
    
    # If only one player, plot a vertical line and add a legend entry
    if len(inventory_sizes) == 1:
        value = inventory_sizes.iloc[0]
        ax.axvline(
            x=value,
            linestyle='dashed',
            color=colors[data_name],
            linewidth=2,
            label=f"{data_name} (Mean: {value:.1f})"
        )
    else:
        # Density plot for the current dataset
        sns.kdeplot(
            data=inventory_sizes, 
            fill=True, 
            alpha=0.5, 
            color=colors[data_name], 
            linewidth=1.5,
            label=f"{data_name} (Mean: {inventory_sizes.mean():.1f})",
            ax=ax
        )
        # Plot the mean as a vertical dashed line
        ax.axvline(
            x=inventory_sizes.mean(), 
            linestyle='dashed', 
            color=colors[data_name], 
            linewidth=1,
        )

    return inventory_sizes


# Function to plot multiple datasets together
def plot_multiple_datasets(datasets, trials=500, trim_percentile=0.99):
    plt.figure(figsize=(8, 6))

    max_xlim = 0  # To calculate the maximum x-axis limit dynamically

    # Plot each dataset
    for data, name in datasets:
        inventory_sizes = final_trial_inventory_density(data, trials=trials, data_name=name)
        # Determine the x-axis limit based on the trim_percentile (e.g., 99th percentile)
        max_xlim = max(max_xlim, inventory_sizes.quantile(trim_percentile))

    # Set x-axis limits dynamically while accounting for outliers
    plt.xlim(0, max_xlim * 1.1)  # Add a small buffer for aesthetics

    # Customize the plot
    plt.title("Final Inventory Distribution Across Player Types (Density)", fontsize=16)
    plt.xlabel("Final inventory", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(title="Player Type", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plt.savefig('output/picture/final_inventory_distribution_density_comparison.png', dpi=300)
    #plt.savefig('output/picture/final_inventory_distribution_density_comparison.pdf', format='pdf', dpi=300)
    plt.show()

# Define your datasets and names
datasets = [
    (Llama3_8B_LLM_with_codes, 'LLaMA3.1-8B'),
    (Llama3_70B_LLM_with_codes, 'LLaMA3.1-70B'),
    (Base_LLM_with_codes, 'GPT-4o'),
    (Human_with_codes, 'Human'),
    (deepseek_reasoner_LLM_with_codes, 'DeepSeek-R1'),
    (o1_LLM_with_codes, 'o1')
]

# Plot all datasets together
plot_multiple_datasets(datasets)

#------------------------------------------------------------------------------------------------
# figure 5: plot the condition percentages of each model
#------------------------------------------------------------------------------------------------
# Define track_pairs function with explicit colorbar mappings
class ModelConditionTracker:
    def __init__(self, data, model_name):
        self.model_name = model_name
        if self.model_name == 'Human':
            # Filter Human data to remove IDs with <= 5 trials
            max_trials_per_id = data.groupby('id')['trial'].max()
            valid_ids = max_trials_per_id[max_trials_per_id > 499].index
            data = data[data['id'].isin(valid_ids)]
            print(f"Filtered Human data from {len(max_trials_per_id)} to {len(valid_ids)} IDs")
            # cut the trial number more than 500
            data = data[data['trial'] <= 500]
            self.data = data

        self.conditions = {
            "Failure with Existing Combination",
            "Failure with New Combination",
            "Success with Existing Combination",
            "Success with New Combination",
            "Invalid Trial"
        }

    def _categorize_trials_for_id(self, data):
        """Categorize trials for a single ID"""
        
        # miss trial number
        if self.model_name != 'Human':
            all_trial_numbers = set(range(500))   
        else:
            all_trial_numbers = set(range(data['trial'].max()+1))
        
        for id, player_data in data.groupby('id'):
            unique_pairs = set()
            successful_pairs = set()
            failed_trial_pairs = set()
            conditions = []
            exist_trial_numbers = set(player_data['trial'])
            missing_trial_numbers = all_trial_numbers - exist_trial_numbers
        
            # Record failed trials
            if missing_trial_numbers:
                for missing_trial in missing_trial_numbers:
                    failed_trial_pairs.add(f"Player {id}, Failed Trial {missing_trial}")
                    conditions.append("Invalid Trial")

            for _, row in player_data.iterrows():
                pair = tuple(sorted([row['first'], row['second']]))

                if row['success'] == 1:
                    if pair in successful_pairs:
                        conditions.append("Success with Existing Combination")
                    else:
                        successful_pairs.add(pair)
                        conditions.append("Success with New Combination")
                else:
                    if pair in unique_pairs:
                        conditions.append("Failure with Existing Combination")
                    else:
                        unique_pairs.add(pair)
                        conditions.append("Failure with New Combination")

        return conditions


    def calculate_model_percentages(self, data):
        """Calculate percentages for Human or Large LLM"""
        if data is None:
            return None
        id_percentages = []
        
        # Calculate percentages for each ID
        for id_val in data['id'].unique():
            id_data = data[data['id'] == id_val]
            conditions = self._categorize_trials_for_id(id_data)
            total_trials = len(conditions)
            id_dict = {}
            
            # Calculate percentages for each condition
            for condition in self.conditions:
                count = conditions.count(condition)
                percentage = count / total_trials * 100
                id_dict[condition] = percentage

            id_percentages.append(id_dict)
        
        # Calculate mean and std across IDs
        percentages_df = pd.DataFrame(id_percentages)
        return {
            'mean': percentages_df.mean(),
            'std': percentages_df.std(),
            'n_ids': len(data['id'].unique())
        }

    def get_and_save_temperature_condition_percentages(self, conditions, data):
        result = []
        if self.model_name in {"Human", "GPT-4o(prompt-engineering)", "o1", "DeepSeek-R1"}:
            percentages = self.calculate_model_percentages(data)
            if percentages:
                mean_percentages = [percentages['mean'][condition] for condition in conditions]
                std_percentages = [percentages['std'][condition] for condition in conditions]
                print(f'\n{self.model_name} averages:')
                for condition, mean_pct, std_pct in zip(conditions, mean_percentages, std_percentages):
                    print(f'{condition}: {mean_pct:.2f}% (±{std_pct:.2f})')
                    result.append(
                        {
                            "model": self.model_name,
                            "temperature": None,
                            "condition": condition,
                            "mean": mean_pct,
                            "std": std_pct
                        }
                    )
            # Convert result to DataFrame for plotting
            result_df = pd.DataFrame(result)
            # plot the condition percentages for each model's intervention
            plt.figure(figsize=(10, 6))
            for condition in result_df['condition'].unique():
                condition_data = result_df[result_df['condition'] == condition]
                sns.lineplot(data=condition_data, x='intervention', y='mean', marker='o', label=condition, linewidth=2)
            plt.title(f'{self.model_name} condition percentages', fontsize=16)
            plt.xlabel('Intervention', fontsize=16)
            plt.ylabel('Percentage (%)', fontsize=16)
            plt.legend(fontsize=12)
            plt.tight_layout()
            #plt.savefig(f'output/picture/{self.model_name}_condition_percentages.png', dpi=300)
            #plt.savefig(f'output/picture/{self.model_name}_empowerment_condition_percentages.png',dpi=300)
            plt.savefig(f'output/picture/{self.model_name}_original_condition_percentages.png',dpi=300)
            plt.show()

        else:
            # process gpt-4o LLM data with temperature = 1.0
            for temperature in [0, 0.3, 0.7, 1.0]:
                model_temp = self.calculate_model_percentages(data[data['temperature'] == temperature])
                if model_temp:
                    mean_percentages = [model_temp['mean'][condition] for condition in conditions]
                    std_percentages = [model_temp['std'][condition] for condition in conditions]
                    print(f'\n{self.model_name} (temperature = {temperature}) averages:')
                    for condition, mean_pct, std_pct in zip(conditions, mean_percentages, std_percentages):
                        print(f'{condition}: {mean_pct:.2f}% (±{std_pct:.2f})')
                        result.append(
                            {
                                "model": self.model_name,
                                "temperature": temperature,
                                "condition": condition,
                                "mean": mean_pct,
                                "std": std_pct
                            }
                        )
            # Convert result to DataFrame for plotting
            result_df = pd.DataFrame(result)
            #plot the condition percentages for each model's temperature
            # set the color of the line
            if self.model_name == 'GPT-4o':
                color = '#7ABBDB'
            elif self.model_name == 'LLaMA3.1-70B':
                color = '#DBB428'
            elif self.model_name == 'LLaMA3.1-8B':
                color = '#84BA42'
            # Plot condition percentages for each model's temperature
            plt.figure(figsize=(8, 6))

            # Loop over conditions and plot separate lines
            for condition in result_df['condition'].unique():
                condition_data = result_df[result_df['condition'] == condition]
                sns.lineplot(
                    data=condition_data,
                    x='temperature', y='mean', marker='o',
                    label=condition, linewidth=2
                )
        
            plt.title(f'{self.model_name} Condition Percentages', fontsize=16)
            plt.xticks([0, 0.3, 0.7, 1.0], fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel('Temperature', fontsize=16)
            plt.ylim(0, 100)
            plt.ylabel('Percentage (%)', fontsize=16)
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'output/picture/{self.model_name}_condition_percentages.png', dpi=300)
            # save figure as pdf for latex
            plt.savefig(f'output/picture/{self.model_name}_condition_percentages.pdf', format='pdf',dpi=300)
            plt.show()
        # save the result to a csv file
        #pd.DataFrame(result).to_csv(f'output/data/{self.model_name}_condition_percentages.csv', index=False)
        

    
# call the class
tracker = ModelConditionTracker(Base_LLM_with_codes, 'GPT-4o')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, Base_LLM_with_codes)
tracker = ModelConditionTracker(Llama3_70B_LLM_with_codes, 'LLaMA3.1-70B')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, Llama3_70B_LLM_with_codes)
tracker = ModelConditionTracker(Llama3_8B_LLM_with_codes, 'LLaMA3.1-8B')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, Llama3_8B_LLM_with_codes)
# human data will be processed in long time, so we will process it later
tracker = ModelConditionTracker(Human_with_codes, 'Human')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, Human_with_codes)
#  the gpt-4o
tracker = ModelConditionTracker(prompt_engineering_LLM_with_codes, 'GPT-4o(prompt-engineering)')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, prompt_engineering_LLM_with_codes)
# o1
tracker = ModelConditionTracker(o1_LLM_with_codes, 'o1')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, o1_LLM_with_codes)
# deepseek-reasoner
tracker = ModelConditionTracker(deepseek_reasoner_LLM_with_codes, 'DeepSeek-R1')
tracker.get_and_save_temperature_condition_percentages(tracker.conditions, deepseek_reasoner_LLM_with_codes)

#------------------------------------------------------------------------------------------------
# figure 6: plot the bar chart with best temperature(1.0)
#------------------------------------------------------------------------------------------------
def plot_bar_chart_with_best_temperature():
    conditions = [
        "Failure & New ",
        "Success & New ",
        "Success & Existing ",
        "Failure & Existing ",
        "Invalid Trial",
    ]
    # Percentage of each condition
    gpt_4o_means = [35.08, 7.96, 16.68, 37.80, 2.48]
    gpt_4o_stds = [2.99, 0.62, 1.79, 4.99, 0.81]

    llama3_70b_means = [17.96, 7.44, 29.92, 29.92, 14.76]
    llama3_70b_stds = [2.03, 0.61, 1.15, 3.36, 4.36]

    llama3_8b_means = [3.64, 2.04, 17.60, 24.16, 52.56]
    llama3_8b_stds = [2.15, 1.69, 11.40, 14.24, 26.41]

    #all human data
    #human_means = [42.82, 38.36, 7.17, 11.65, 0.00]
    #human_stds = [13.53, 19.63, 6.43, 9.60, 0.00]

    #human player completed more than500 trials
    human_means = [39.63, 43.21, 6.58, 10.59, 0.00]
    human_stds = [16.95, 24.29, 6.69, 9.78, 0.00]

    # gpt-4o(prompt-engineering)
    gpt4o_prompt_engineering_means = [42.80, 9.64, 10.84, 28.64, 8.08]
    gpt4o_prompt_engineering_stds = [5.45, 2.45, 1.85, 8.03, 11.48]

    # o1
    o1_means = [61.24,38.36, 0.04, 0.36, 0]
    o1_stds = [6.13, 6.13, 0.09, 0.09, 0]

    # deepseek-reasoner
    deepseek_reasoner_means = [77.04, 19.80, 0.64, 0.72, 1.80]
    deepseek_reasoner_stds = [5.72, 6.68, 0.5, 0.5, 0.89]

    x = np.arange(len(conditions))  # Label locations
    width = 0.13  # Width of the bars

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars
    ax.bar(x - 3 * width, llama3_8b_means, width, yerr=llama3_8b_stds, label='LLaMA3.1-8B(temp = 1.0)', color='#84BA42', capsize=3)
    ax.bar(x - 2 * width, llama3_70b_means, width, yerr=llama3_70b_stds, label='LLaMA3.1-70B(temp = 1.0)', color='#DBB428', capsize=3)
    ax.bar(x - 1 * width, gpt_4o_means, width, yerr=gpt_4o_stds, label='GPT-4o(temp = 1.0)', color='#7ABBDB', capsize=3)
    # GPT-4o(prompt-engineering)
    #ax.bar(x + 0 * width, gpt4o_prompt_engineering_means, width, yerr=gpt4o_prompt_engineering_stds, label='GPT-4o(prompt-engineering)', color='#4485C7', capsize=3)
    ax.bar(x + 0 * width, human_means, width, yerr=human_stds, label='Human', color='#A51C36', capsize=3)
    ax.bar(x + 1 * width, o1_means, width, yerr=o1_stds, label='o1', color='#682478', capsize=3)
    # DeepSeek-R1
    ax.bar(x + 2 * width, deepseek_reasoner_means, width, yerr=deepseek_reasoner_stds, label='DeepSeek-R1', color='#bcfce7', capsize=3)

    # Labels and Titles
    ax.set_xlabel("Conditions", fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=16)
    ax.set_title("Average Percentage of Each Condition", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, ha="center", fontsize=12)  # Add slight rotation for clarity
    ax.legend(fontsize=12)

    # Adjust layout to prevent clipping
    plt.yticks(fontsize=14)
    plt.tight_layout(pad=2)
    #plt.savefig(f'output/picture/LLM(6)_and_Human_behavior_comparison_fixed.png', dpi=300)
    #plt.savefig(f'output/picture/LLM(6)_and_Human_behavior_comparison_fixed(prompt-engineering).png', dpi=300)
    plt.savefig(f'output/picture/LLM(6)_and_Human_behavior_comparison_fixed(deepseek-reasoner).png', dpi=300)
    plt.show()

plot_bar_chart_with_best_temperature()

#------------------------------------------------------------------------------------------------
# figure 7: plot the LLM inventory vs human levels
#------------------------------------------------------------------------------------------------
def plot_llm_inventory_vs_human_levels(
    Base_LLM_with_codes, Llama3_70B_LLM_with_codes, Llama3_8B_LLM_with_codes,
    prompt_engineering_LLM_with_codes, deepseek_reasoner_LLM_with_codes, o1_LLM_with_codes, Human_with_codes, trial_num=500
):
    # Filter LLM data to temperature = 1.0
    Base_LLM_best_temperature = Base_LLM_with_codes[Base_LLM_with_codes['temperature'] == 1.0]
    Llama3_70B_LLM_best_temperature = Llama3_70B_LLM_with_codes[Llama3_70B_LLM_with_codes['temperature'] == 1.0]
    Llama3_8B_LLM_best_temperature = Llama3_8B_LLM_with_codes[Llama3_8B_LLM_with_codes['temperature'] == 1.0]
    
    # deal the missing data
    deepseek_reasoner = fill_missing_inventory(deepseek_reasoner_LLM_with_codes, trial_num)
    o1 = fill_missing_inventory(o1_LLM_with_codes, trial_num)
    Base_LLM_best_temperature = fill_missing_inventory(Base_LLM_best_temperature, trial_num)
    Llama3_70B_LLM_best_temperature = fill_missing_inventory(Llama3_70B_LLM_best_temperature, trial_num)
    Llama3_8B_LLM_best_temperature = fill_missing_inventory(Llama3_8B_LLM_best_temperature, trial_num)
    # Initialize plot
    plt.figure(figsize=(8, 6))
    
    # Define colors for each model
    colors = {
        'LLaMA3.1-8B(temp = 1.0)': '#84BA42',
        'LLaMA3.1-70B(temp = 1.0)': '#DBB428',
        'GPT-4o(temp = 1.0)': '#7ABBDB',
        'Human': '#A51C36',
        'DeepSeek-R1': '#bcfce7',
        'o1': '#682478',
    }
    
    # Fine-grained percentiles (e.g., 18%, 21%, ...)
    percentiles = np.arange(1, 101, 1)  # Percentiles from 1% to 100% with 1% steps
    # use human data before trial 500
    human_percentiles = Human_with_codes[Human_with_codes['trial'] < 500].groupby('trial')['inventory'].apply(
        lambda x: [np.percentile(x, q) for q in percentiles]
    )
    human_percentiles = pd.DataFrame(
        human_percentiles.tolist(), index=human_percentiles.index, columns=[f"{p}%" for p in percentiles]
    )
    
    # Helper function to calculate LLM performance relative to human percentiles
    def calculate_llm_percentile(LLM_data, human_percentiles):
        llm_mean_inventory = LLM_data.groupby('trial')['inventory'].mean()
        llm_percentile_levels = []
        for trial, mean_value in llm_mean_inventory.items():
            if trial in human_percentiles.index:
                human_levels = human_percentiles.loc[trial]
                # Find the percentile level closest to the LLM's mean inventory
                for p, level in zip(percentiles, human_levels):
                    if mean_value <= level:
                        llm_percentile_levels.append(p)
                        break
                else:
                    llm_percentile_levels.append(100)
            else:
                llm_percentile_levels.append(np.nan)
        return pd.Series(llm_percentile_levels, index=llm_mean_inventory.index)

    # Calculate and plot percentile levels for each model
    models = {
        'LLaMA3.1-8B(temp = 1.0)': Llama3_8B_LLM_best_temperature,
        'LLaMA3.1-70B(temp = 1.0)': Llama3_70B_LLM_best_temperature,
        'GPT-4o(temp = 1.0)': Base_LLM_best_temperature,
        'DeepSeek-R1': deepseek_reasoner,
        'o1': o1
    }
    
    for model_name, model_data in models.items():
        llm_percentile = calculate_llm_percentile(model_data, human_percentiles)
        plt.plot(
            llm_percentile.index,
            llm_percentile,
            label=model_name,
            color=colors[model_name],
            linewidth=2
        )

    # Add human performance for reference
    plt.axhline(y=50, color=colors['Human'], linestyle='--', label='Human Median (50%)')
    plt.axhline(y=90, color=colors['Human'], linestyle='--', label='Human Median (90%)')

    # Customize plot
    #plt.title('LLM inventory performance relative to Human percentiles', fontsize=16)
    plt.xlabel('Trial', fontsize=16)
    plt.ylabel('Percentile Level', fontsize=16)
    plt.ylim(0, 100)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()

    # Save and show plot
    plt.savefig(f'output/picture/llm_vs_human_percentiles_fine.png', dpi=300)
    plt.show()

# Call the function
plot_llm_inventory_vs_human_levels(
    Base_LLM_with_codes, Llama3_70B_LLM_with_codes, Llama3_8B_LLM_with_codes,prompt_engineering_LLM_with_codes, 
    deepseek_reasoner_LLM_with_codes, o1_LLM_with_codes, Human_with_codes, trial_num=500
)