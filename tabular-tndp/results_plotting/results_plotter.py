#%%
import json
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
# Set the color palette to start with two specific colors using matplotlib

start_colors = ["#9365b4", "#ff7f5e", "#1f77b4"]  # Blue and orange
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
custom_colors = start_colors + [c for c in default_colors if c not in start_colors]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)

plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'font.family': 'Georgia',
})

REQ_SEEDS = 10 # to control if a model was not run for sufficient seeds
#%%
api = wandb.Api()

def highlight_cells(cells, ax, **kwargs):
    """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
    """
    for cell in cells:
        (y, x) = cell
        rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, linewidth=2, **kwargs)
        ax.add_patch(rect)
    return rect

def read_json(file_path):
    with open(file_path, 'r') as file:
        json_content = json.load(file)
    return json_content

def gini(x, normalized=True):
    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cum_x = np.cumsum(sorted_x, axis=1, dtype=float)
    gi = (n + 1 - 2 * np.sum(cum_x, axis=1) / cum_x[:, -1]) / n
    if normalized:
        gi = gi * (n / (n - 1))
    return gi

def average_per_step(hvs_by_seed):
    # Determine the maximum length of the sublists
    max_length = max(len(sublist) for sublist in hvs_by_seed)
    
    # Pad shorter sublists with zeros
    padded_hvs_by_seed = [sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in hvs_by_seed]
    
    # Calculate the average per step and confidence interval
    averages = []
    mins = []
    maxs = []
    for i in range(max_length):
        step_values = [sublist[i] for sublist in padded_hvs_by_seed]
        average = sum(step_values) / len(hvs_by_seed)
        averages.append(average)
        # Calculate the confidence interval
        ci = 1.96 * np.std(step_values) / np.sqrt(len(step_values))
        mins.append(average - ci)
        maxs.append(average + ci)
    
    return averages, mins, maxs

def load_all_results_from_wadb(all_objectives, env_name=None):        
    all_results = pd.DataFrame()
    avg_reward_over_time = pd.DataFrame()

    for _, reward_type in enumerate(all_objectives):
        models_to_plot = pd.DataFrame(reward_type['models'])

        results_by_reward_type = {}
        for i, model_name in enumerate(models_to_plot['name'].unique()):
            models = models_to_plot[models_to_plot['name'] == model_name].to_dict('records')
            if len(models[0]['run_ids']) < REQ_SEEDS:
                print(f"!WARNING! {reward_type['reward_type']} reward type, {model_name} does not have enough seeds (has {len(models[0]['run_ids'])}, while {REQ_SEEDS} are required)")

            avg_rewards_by_seed = []
            for j, model in enumerate(models):
                print(f"Processing {env_name} {model_name} ({reward_type['reward_type']}) - {model['run_ids']}")
                # Read the content of the output file
                results_by_reward_type[model_name] = {'average_test_reward': []}
                for i in range(len(model['run_ids'])):
                    if model['run_ids'][i] == '':
                        print(f"WARNING - Empty run id in {model_name}")
                        continue
                    
                    if model_name == 'DeepRL' or model_name == 'TabularMNEP':
                        project_name = "TNDP-RL"
                    elif model_name == 'GA':
                        project_name = "TNDP-GA"
                    elif model_name == 'GS':
                        project_name = "TNDP-GS"
                    
                    run = api.run(f"{project_name}/{model['run_ids'][i]}")
                    
                    if model_name not in ['GA', 'GS']:
                        results_by_reward_type[model_name]['average_test_reward'].append(run.summary['Average-Test-Reward'])
                    else:
                        results_by_reward_type[model_name]['average_test_reward'].append(run.summary['average_reward'])

                    if model_name != 'DeepRL':
                        keys_to_load = ['episode', 'average_reward'] if model_name != 'GA' else ['generation', 'average_reward']
                        history = []
                        for row in run.scan_history(keys=keys_to_load):
                            history.append(row)

                        # Convert to DataFrame
                        history = pd.DataFrame(history)
                        rewards = history[history['average_reward'] > 0]['average_reward'].tolist()
                        
                        if len(rewards) > 0:
                            avg_rewards_by_seed.append(rewards)
                        else:
                            print(f"WARNING - No average_reward values in {model_name}, {reward_type} - {model['run_ids'][i]}")
                    else:
                        rewards = []
                        

            model_name_adj = model_name.replace(f'-{env_name}', '')
            if len(avg_rewards_by_seed) > 0:
                averages, ci_upper, ci_lower = average_per_step(avg_rewards_by_seed)
                
                avg_reward_over_time = pd.concat([avg_reward_over_time, 
                                                  pd.DataFrame({f"{model_name_adj}_{reward_type['reward_type']}": averages,
                                                                f"{model_name_adj}_{reward_type['reward_type']}_upper": ci_upper,
                                                                f"{model_name_adj}_{reward_type['reward_type']}_lower": ci_lower})])
            
        # Quite a hacky way to get the results in a dataframe, but didn't have time to do it properly (thanks copilot)
        # Convert all_results to a dataframe, with columns 'model', 'metric', 'value', and each row is a different value and not a list
        # results_by_objective = pd.DataFrame([(name, metric, value) for name in results_by_objective.keys() for metric in results_by_objective[name].keys() for value in results_by_objective[name][metric]], columns=['model', 'metric', 'value'])
        # Convert all_results to a dataframe, with columns 'model', 'lambda; 'metric', 'value', and each row is a different value and not a list
        results_by_reward_type = pd.DataFrame([(reward_type['reward_type'], name, metric, value) for name in results_by_reward_type.keys() 
                                            for metric in results_by_reward_type[name].keys() 
                                            for value in results_by_reward_type[name][metric]], columns=['reward_type', 'model', 'metric', 'value'])
        results_by_reward_type['model'] = results_by_reward_type['model'].str.replace(f'-{env_name}', '')
        all_results = pd.concat([all_results, results_by_reward_type])
        
    return all_results, avg_reward_over_time

xian_results, xian_avg_rw_over_time = load_all_results_from_wadb(read_json('./result_ids_xian.txt'), 'Xian')
ams_results, ams_avg_rw_over_time = load_all_results_from_wadb(read_json('./result_ids_ams.txt'), 'Amsterdam')
# %%
# Calculate mean and confidence interval (CI)
def mean_ci(series):
    mean = series.mean()
    std_err = np.std(series, ddof=1) / np.sqrt(len(series))
    ci_max = mean + std_err * 1.96
    ci_min = mean - std_err * 1.96
    return pd.Series([mean, ci_min, ci_max], index=['mean', 'ci_min', 'ci_max'])

xian_results_summary = xian_results.groupby(['reward_type', 'model', 'metric'])['value'].apply(mean_ci).unstack().reset_index()

ams_results_summary = ams_results.groupby(['reward_type', 'model', 'metric'])['value'].apply(mean_ci).unstack().reset_index()

#%%
# Plot a bar chart of xian_results_summary with error bars
def plot_results_summary(results_summary, city_name, models_to_plot):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8))
    axs = axs.flatten()

    reward_types = ['max_efficiency', 'ggi2', 'ggi4', 'rawls']
    reward_type_names = ['Max Efficiency', 'GGI(2)', 'GGI(4)', 'Rawls']

    for idx, reward_type in enumerate(reward_types):
        # Filter data for the current reward type
        data = results_summary[results_summary['reward_type'] == reward_type]
        
        # Plot bars with error bars in the specified order
        for i, model in enumerate(models_to_plot):
            model_data = data[data['model'] == model]
            if not model_data.empty:
                axs[idx].bar(i, model_data['mean'], 
                             yerr=[model_data['mean'] - model_data['ci_min'], model_data['ci_max'] - model_data['mean']], 
                             capsize=12, label=model)
        
        # Set labels and title for each subplot
        axs[idx].set_xticks(range(len(models_to_plot)))
        axs[idx].set_xticklabels(models_to_plot)
        axs[idx].set_ylabel(f'{reward_type_names[idx]}')
        # axs[idx].set_title(f'{reward_type_names[idx]}')
    
    # plt.suptitle(f'Results {city_name}')
    plt.tight_layout()
    plt.show()

# Usage:
models_to_plot = ['GA', 'DeepRL', 'TabularMNEP']
plot_results_summary(xian_results_summary, 'Xian', models_to_plot)
plot_results_summary(ams_results_summary, 'Amsterdam', models_to_plot)

#%% Plot average reward over time
wandb_to_local_mapper = [
    {
        "environment": "xian",
        "reward_type": "max_efficiency",
        "mapping": {
            'y1s2ebq8': 'xian_20241003_11_05_54.565620',
            '75h2klzp': 'xian_20241003_11_06_57.001302',
            'ziz0qbk8': 'xian_20241017_00_03_25.938628',
            'ma9yhy1o': 'xian_20241002_21_31_53.751992',
            '2a5aes6q': 'xian_20241003_11_10_45.833755',
        }
    },
    {
        "environment": "xian",
        "reward_type": "rawls",
        "mapping": {
            'eng3ru2n': 'xian_20241004_13_04_11.931713',
            'fgor0rft': 'xian_20241004_13_04_21.155581',
            'hxqtppx7': 'xian_20241004_16_34_34.483372',
            'hv9p3c2u': 'xian_20241004_16_34_34.482076',
            'j0ejk07m': 'xian_20241004_16_34_47.456490',
        }
    },
    {
        "environment": "amsterdam",
        "reward_type": "max_efficiency",
        "mapping": {
            '1se01ud5': 'amsterdam_20241010_12_22_01.564520',
            '5sif9a7w': 'amsterdam_20241010_12_22_01.568074',
            'z4a9e3fw': 'amsterdam_20241010_12_22_01.570698',
            '6ot0zo6e': 'amsterdam_20241010_12_22_01.567504',
            'odcongwb': 'amsterdam_20241010_14_40_41.815377',
        }
    },
    {
        "environment": "amsterdam",
        "reward_type": "rawls",
        "mapping": {
            'xz7id1ah': 'amsterdam_20241010_14_42_47.909724',
            '9crampld': 'amsterdam_20241011_08_00_43.281005',
            'qwzu05ht': 'amsterdam_20241011_08_32_46.147697',
            '425us7iw': 'amsterdam_20241011_09_05_22.118755',
            'fxnv8t7d': 'amsterdam_20241011_09_31_59.329177',
        }
    },
]

result_dir = '../../fair-network-expansion/result/new'

fig, ax = plt.subplots(figsize=(6.4, 4))

reward_type = 'max_efficiency'
reward_type_name = 'Max Efficiency'

# Plotting for Xian
mapper = next((m for m in wandb_to_local_mapper if m['reward_type'] == reward_type and m['environment'] == 'xian'), None)
if mapper:
    mapping = mapper['mapping']
    all_reward_data = []
    for wandb_id, local_dir in mapping.items():
        file_path = f"{result_dir}/{local_dir}/reward_actloss_criloss.txt"
        try:
            reward_data = np.loadtxt(file_path)
            all_reward_data.append(reward_data[:, 0])
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    
    if all_reward_data:
        avg_reward_data = np.mean(all_reward_data, axis=0)
        ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
        ax.semilogx(np.arange(1, len(avg_reward_data) + 1), avg_reward_data, label="DeepRL", linewidth=2)
        ax.fill_between(np.arange(1, len(avg_reward_data) + 1), avg_reward_data - ci, avg_reward_data + ci, alpha=0.3)

# Then plot the other data
data = xian_avg_rw_over_time.filter(regex=f'.*{reward_type}').iloc[::128].reset_index(drop=True)
ax.semilogx(data.index + 1, data[f'TabularMNEP_{reward_type}'], label='TabularMNEP', linewidth=2)
ax.fill_between(data.index + 1, data[f'TabularMNEP_{reward_type}_lower'], data[f'TabularMNEP_{reward_type}_upper'], alpha=0.3)

# Set labels and title for both subplots
ax.set_xlabel('Steps (x128 episodes)')
ax.set_ylabel('Average Reward')
ax.set_title(f'Average Reward Over Time')
ax.legend(loc='upper left', fontsize=16)

# Plotting for Amsterdam
# ax = axs[1]
# mapper = next((m for m in wandb_to_local_mapper if m['reward_type'] == reward_type and m['environment'] == 'amsterdam'), None)
# if mapper:
#     mapping = mapper['mapping']
#     all_reward_data = []
#     for wandb_id, local_dir in mapping.items():
#         file_path = f"{result_dir}/{local_dir}/reward_actloss_criloss.txt"
#         try:
#             reward_data = np.loadtxt(file_path)
#             all_reward_data.append(reward_data[:, 0])
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
    
#     if all_reward_data:
#         avg_reward_data = np.mean(all_reward_data, axis=0)
#         ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
#         ax.semilogx(np.arange(1, len(avg_reward_data) + 1), avg_reward_data, label="DeepRL", linewidth=2)
#         ax.fill_between(np.arange(1, len(avg_reward_data) + 1), avg_reward_data - ci, avg_reward_data + ci, alpha=0.3)

# data = ams_avg_rw_over_time.filter(regex=f'.*{reward_type}').iloc[::128].reset_index(drop=True)
# ax.semilogx(data.index + 1, data[f'TabularMNEP_{reward_type}'], label='TabularMNEP', linewidth=2)
# ax.fill_between(data.index + 1, data[f'TabularMNEP_{reward_type}_lower'], data[f'TabularMNEP_{reward_type}_upper'], alpha=0.3)

# # Set labels and title for both subplots
# for ax in axs:
#     ax.set_xlabel('Steps (x128 episodes)')
#     ax.set_ylabel('Average Reward')
#     ax.set_title(f'Average Reward Over Time - {reward_type_name}')
#     ax.legend(loc='lower right', fontsize=16)

# plt.tight_layout()
# plt.show()

# %%

import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
from pathlib import Path
from gymnasium.envs.registration import register
import matplotlib.patches as mpatches

register(
    id="motndp_xian-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_amsterdam-v0",
    entry_point="motndp.motndp:MOTNDP",
)

xian_city = City(Path('../envs/mo-tndp/cities/xian'), groups_file='price_groups_5.txt', ignore_existing_lines=False)   
xian_env = mo_gym.make('motndp_xian-v0', city=xian_city, constraints=MetroConstraints(xian_city), nr_stations=0, od_type='abs', chained_reward=True)

ams_city = City(Path('../envs/mo-tndp/cities/amsterdam'), groups_file='price_groups_5.txt', ignore_existing_lines=False)   
ams_env = mo_gym.make('motndp_amsterdam-v0', city=ams_city, constraints=MetroConstraints(ams_city), nr_stations=0, od_type='abs', chained_reward=True)

#%%
cp = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]
markers = ["o", "s", "^", "D", "v"]
hatches = ['', '/', '-',  'o', '+', 'x', 'o', 'O', '.', '*']
# plt.rcParams.update({'font.size': 28})

plt.rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 20,
    'font.family': 'Georgia',
})

LINEWIDTH = 7
MARKERSIZE = 15
def plot_environment_lines(runs_to_plot_lines, environment_name, env, grp_legend_loc='lower right', sat_od_type='pct'):
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4))  
    map_ax = axs[0]
    barplot_ax = axs[1]
    
    # Plotting the map
    im = map_ax.imshow(env.unwrapped.city.grid_groups, alpha=0.1)
    
    labels = ['1st quintile', '2nd quintile', '3rd quintile', '4th quintile', '5th quintile']
    values = (np.unique(env.unwrapped.city.grid_groups[~np.isnan(env.unwrapped.city.grid_groups)]))
    colors = [ im.cmap(im.norm(value)) for value in values]
    # patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(labels)) ]
    # map_ax.legend(handles=patches, loc=grp_legend_loc, prop={'size': 16})

    
    # Plot existing lines
    for i, l in enumerate(env.unwrapped.city.existing_lines):
        station_locs = env.unwrapped.city.index_to_grid(l)
        map_ax.plot(station_locs[:, 1], station_locs[:, 0], '--', color='#363232', label='Existing lines' if i == 0 else None, linewidth=5)
        
    # map_ax.legend(labels=['Existing lines'], loc=grp_legend_loc, prop={'size': 22})

    style_index = 0
    group_names = ('1st quint.', '2nd', '3rd', '4th', '5th')
    for run_info in runs_to_plot_lines:
        if run_info["environment"] == environment_name:
            for reward_type, run_id in run_info["runs"].items():
                project_name = "TNDP-RL" if reward_type != 'GA' else "TNDP-GA"
                run = api.run(f"{project_name}/{run_id}")
                run.file(f"eval/{run_id}-average-generated-line.npy").download(replace=True)
                line = np.load(f"eval/{run_id}-average-generated-line.npy")
                # map_ax.plot(line[:, 1], line[:, 0], f'{markers[style_index]}-', color=cp[style_index % len(cp)], label=f'{reward_type}', linewidth=LINEWIDTH, markersize=MARKERSIZE)
                map_ax.plot(line[:, 1], line[:, 0], f'-', color=cp[style_index % len(cp)], label=f'{reward_type}', linewidth=LINEWIDTH, markersize=MARKERSIZE)
                
                width = 0.2  
                ind = np.arange(len(group_names))
                # Plot the group satisfied OD flows in a bar chart with space between bars
                run.file(f"eval/{run_id}-average-satisfied-ods-by-group.npy").download(replace=True)
                sat_group_ods = np.load(f"eval/{run_id}-average-satisfied-ods-by-group.npy")
                
                if sat_od_type == 'pct':
                    sat_group_ods = sat_group_ods / env.unwrapped.city.group_od_sum * 100
                
                barplot_ax.bar(ind + style_index * width, sat_group_ods, width, color=cp[style_index], hatch=hatches[style_index])
                style_index += 1
                
    # barplot_ax.legend(loc='upper left', fontsize=16) 
    barplot_ax.set_ylabel('Satisfied OD %')
    
    barplot_ax.set_xticks(ind + width * 2)
    barplot_ax.set_xticklabels(group_names)

    # fig.suptitle(f'Generated lines and Benefits Distribution - {environment_name}', fontsize=38, y=1.05)
    fig.tight_layout()
    
    fig.legend(loc='lower center', ncol=len(runs_to_plot_lines[0]['runs']) + 1, bbox_to_anchor=(0.5, -0.1))
    
    return fig


runs_to_plot_lines = [
    {
        "environment": "Xian",
        "grid_y_size": 29,
        "runs": {
            "Max Efficiency": "mub6vtfm", # this replaces iay1mvek, but has group distro (groups = 5)
            "GGI(2)": "v4txrdje",
            "GGI(4)": "l014wil7",
            "Rawls": "aprwjpef",
        }
    },
    {
        "environment": "Amsterdam",
        "grid_y_size": 47,
        "runs": {
            "Max Efficiency": "986jke14", # this replaces ots8htds, but has group distro (groups = 5)
            "GGI(2)": "quimcanf",
            "GGI(4)": "kzmkqh42",
            "Rawls": "rtvw0hds",
        }
    }
]
# Plotting for Xian environment
fig_xian = plot_environment_lines(runs_to_plot_lines, "Xian", xian_env)

# Plotting for Amsterdam environment
fig_ams = plot_environment_lines(runs_to_plot_lines, "Amsterdam", ams_env, "lower left")  # Assuming amsterdam_env is defined elsewhere
fig_xian.savefig('genlines_xian.pdf', bbox_inches='tight')
fig_ams.savefig('genlines_ams.pdf', bbox_inches='tight')

# %% Combine Q-Start and Q-table tables into a single plot
runs_to_plot = ['gxtct65s', 'vfvxgiva', '2iz9aexo', 'bbwmfluu', 'lin6e4ts']
Q_start_combined = np.zeros((xian_env.unwrapped.city.grid_x_size, xian_env.unwrapped.city.grid_y_size))
for run_id in runs_to_plot:
    project_name = "TNDP-RL" if reward_type != 'GA' else "TNDP-GA"
    run = api.run(f"{project_name}/{run_id}")
    run.file(f"q_tables/{run_id}_qstart.npy").download(replace=True)
    line = np.load(f"q_tables/{run_id}_qstart.npy")
        
    Q_start_combined += line
    
    
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(Q_start_combined, cmap='Blues')
fig.colorbar(im)
fig.suptitle('Average Q-Start over Reward Types')
plt.show()