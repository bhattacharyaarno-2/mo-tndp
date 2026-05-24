from collections import deque
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path
import copy

dir_up = (-1, 0)
dir_down = (1, 0)
dir_left = (0, -1)
dir_right = (0, 1)
dir_upleft = (-1, -1)
dir_upright = (-1, 1)
dir_downleft = (1, -1)
dir_downright = (1, 1)

ACTION_TO_DIRECTION = np.array(
    [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
)

class GreedyTNDP:
    def __init__(
        self,
        env,
        nr_stations,
        nr_groups,
        seed,
        wandb_project_name=None,
        wandb_experiment_name=None,
        wandb_run_id=None,
        log: bool = True,
        
    ):
        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.nr_stations = nr_stations
        self.nr_groups = nr_groups
        self.seed = seed
        self.wandb_project_name = wandb_project_name
        self.wandb_experiment_name = wandb_experiment_name
        self.wandb_run_id = wandb_run_id
        self.log = log
        
        if log:
            if not wandb_run_id:
                self.setup_wandb()
            else:
                wandb.init(
                    project=self.wandb_project_name,
                    id=self.wandb_run_id,
                    resume=True,
                    config=self.get_config(),
                )
        
    def get_config(self) -> dict:
        """Get configuration of Greedy Search."""
        return {
            "env_id": self.env_id,
            "od_type": self.env.unwrapped.od_type,
            "nr_stations": self.nr_stations,
            "nr_groups": self.nr_groups,
            "seed": self.seed,
            "chained_reward": self.env.unwrapped.chained_reward,
            "ignore_existing_lines": self.env.unwrapped.city.ignore_existing_lines,
        }
        
    def highlight_cells(self, cells, ax, **kwargs):
        """Highlights a cell in a grid plot. https://stackoverflow.com/questions/56654952/how-to-mark-cells-in-matplotlib-pyplot-imshow-drawing-cell-borders
        """
        for cell in cells:
            (y, x) = cell
            rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, linewidth=2, **kwargs)
            ax.add_patch(rect)
        return rect
    
    def gen_line_plot_grid(self, lines):
        """Generates a grid_x_max * grid_y_max grid where each grid is valued by the frequency it appears in the generated lines.
        Essentially creates a grid of the given line to plot later on.

        Args:
            line (list): list of generated lines of the model
            grid_x_max (int): nr of lines in the grid
            grid_y_mask (int): nr of columns in the grid
        """
        data = np.zeros((self.env.unwrapped.city.grid_x_size, self.env.unwrapped.city.grid_y_size))

        for line in lines:
            for station in line:
                data[station[0], station[1]] += 1
        
        data = data/len(lines)

        return data

    
    def setup_wandb(self, entity=None, group=None):
        wandb.init(
            project=self.wandb_project_name,
            entity=entity,
            config=self.get_config(),
            name=f"{self.env_id}__{self.wandb_experiment_name}__{self.seed}__{int(time.time())}",
            save_code=True,
            group=group,
        )
        
        wandb.define_metric("*", step_metric="episode")
        
    def calculate_reward(self, reward: np.array, reward_type: str):
        """Calculate the reward based on the reward type.

        Args:
            reward (np.array): reward vector
            reward_type (str): type of reward

        Returns:
            float: reward
        """
        if reward_type == 'max_efficiency':
            return reward.sum()
        elif reward_type == 'ggi2':
            return self.ggi_reward(reward, 2)
        elif reward_type == 'ggi4':
            return self.ggi_reward(reward, 4)
        elif reward_type == 'rawls':
            return reward[0]
        else:
            raise ValueError(f"Reward type {reward_type} not implemented")
        
    def ggi_reward(self, reward: np.array, weight: int):
        """Generalized Gini Index reward (see paper for more information).
        Exponentially smaller weights are assigned to the groups with the highest satisfied origin-destination flows.

        Args:
            reward (np.array): reward vector
            weight (int): weight of the reward

        Returns:
            float: total ggi
        """
        # Generate weights
        weights = 1 / (weight ** np.arange(reward.shape[0]))
        # Normalize the weights
        weights /= weights.sum()
        # Sort rewards and calculate GGI reward
        sorted_reward = np.sort(reward)
        
        return np.sum(sorted_reward * weights)

    def run(self, reward_type, starting_loc=None):
        if starting_loc is not None:
            raise ValueError("Starting location is not implemented yet")
        
        wandb.config['reward_type'] = reward_type
        
        best_episode_reward = 0
        
        # Find the starting point by greedily looping through the OD matrix and finding the highest value
        od_matrix = self.env.unwrapped.city.agg_od_mx()
        
        if reward_type == 'max_efficiency':
            max_od_pair = np.unravel_index(np.argmax(od_matrix, axis=None), od_matrix.shape)
        elif reward_type == 'ggi2':            
            weights = 1 / (2 ** np.arange(self.nr_groups))
            weights /= weights.sum()
            weighed_od_matrix = self.env.unwrapped.city.group_od_mx * weights[:, None, None]
            summed_weighted_od_matrix = np.sum(weighed_od_matrix, axis=0)
            max_od_pair = np.unravel_index(np.argmax(summed_weighted_od_matrix, axis=None), summed_weighted_od_matrix.shape)
        elif reward_type == 'ggi4':
            weights = 1 / (4 ** np.arange(self.nr_groups))
            weights /= weights.sum()
            weighed_od_matrix = self.env.unwrapped.city.group_od_mx * weights[:, None, None]
            summed_weighted_od_matrix = np.sum(weighed_od_matrix, axis=0)
            max_od_pair = np.unravel_index(np.argmax(summed_weighted_od_matrix, axis=None), summed_weighted_od_matrix.shape)
        elif reward_type == 'rawls':
            od_matrix_group_0 = self.env.unwrapped.city.group_od_mx[0]
            max_od_pair = np.unravel_index(np.argmax(od_matrix_group_0, axis=None), od_matrix_group_0.shape)
        else:
            raise ValueError(f"Reward type {reward_type} not implemented")
        
        starting_loc = tuple(self.env.unwrapped.city.index_to_grid(max_od_pair[0]))

        obs, info = self.env.reset(options={'loc': starting_loc})
        state = info['location_grid_index']
        episode_reward = 0
        visited_locations = [starting_loc]
        actions = []
        for _ in range(self.nr_stations - 1):
            action_mask = info['action_mask']
            best_action = None
            best_reward = -np.inf

            for action in range(len(action_mask)):
                if action_mask[action] == 1:
                    env_copy = copy.deepcopy(self.env)
                    next_possible_state, possible_reward, possible_done, _, possible_info = env_copy.step(action)
                    total_reward = self.calculate_reward(possible_reward, reward_type)
                    
                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_action = action

            if best_action is None:
                break

            obs, reward, done, _, info = self.env.step(best_action)
            state = info['location_grid_index']
            episode_reward += self.calculate_reward(reward, reward_type)
            visited_locations.append(self.env.unwrapped.city.index_to_grid(state)[0])
            actions.append(best_action)
            if done:
                break
            
        best_episode_cells = info['covered_cells_coordinates']
        best_episode_reward = episode_reward

        if self.log:
            wandb.log(
                {
                    "episode": 0,
                    "average_reward": episode_reward,
                    "best_episode_reward": best_episode_reward,
                    "best_episode_cells": best_episode_cells,
                })
            
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(self.env.unwrapped.city.agg_od_mx())
            
            if not self.env.unwrapped.city.ignore_existing_lines:
                for i, l in enumerate(self.env.unwrapped.city.existing_lines):
                    station_locs = self.env.unwrapped.city.index_to_grid(l)
                    ax.plot(station_locs[:, 1], station_locs[:, 0], '-o', color='#A1A9FF', label='Existing lines' if i == 0 else None)
            
            best_episode_cells = np.array(best_episode_cells)
            ax.plot(best_episode_cells[:, 1], best_episode_cells[:, 0], '-or', label='Generated line')
            # If the test episodes are more than 1, we can plot the average line

            self.highlight_cells([(best_episode_cells[0][0].item(), best_episode_cells[0][1].item())], ax=ax, color='limegreen')
            fig.suptitle(f'Best Episode line \n reward: {best_episode_reward}')
            fig.legend(loc='lower center', ncol=2)
            wandb.log({"Best-Episode-Line": wandb.Image(fig)})
            plt.close(fig)
        
            wandb.finish()
            
        return
    
    