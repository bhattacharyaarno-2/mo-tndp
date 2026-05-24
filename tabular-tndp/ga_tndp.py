from collections import deque
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from pathlib import Path

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

class GATNDP:
    def __init__(
        self,
        env,
        init_pop_size,
        mutation_rate,
        crossover_rate,
        generations,
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
        self.init_pop_size = init_pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
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
        """Get configuration of QLearning."""
        return {
            "env_id": self.env_id,
            "init_pop_size": self.init_pop_size,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "generations": self.generations,
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
        
        wandb.define_metric("*", step_metric="generation")
        
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
    
    def get_direction(self, current_location, visited_locations):
            
        direction = current_location - visited_locations[-2]
        direction[direction > 0] = 1
        direction[direction < 0] = -1
        direction = tuple(direction[0])
        
        for loc in visited_locations[:-3]:
            dir = current_location - loc
            dir[dir > 0] = 1
            dir[dir < 0] = -1
            dir = tuple(dir[0])

            if np.array_equal(dir, dir_upleft):
                direction = dir_upleft
            elif np.array_equal(dir, dir_upright):
                direction = dir_upright
            elif np.array_equal(dir, dir_downleft):
                direction = dir_downleft
            elif np.array_equal(dir, dir_downright):
                direction = dir_downright

        return direction
    
    def update_action_mask_for_direction(self, action_mask, direction):
        # up-left movement
        if (np.array_equal(direction, dir_upleft)):
            action_mask[[1, 2, 3, 4, 5]] = 0
        # up-right movement
        elif (np.array_equal(direction, dir_upright)):
            action_mask[[3, 4, 5, 6, 7]] = 0
        # down-left movement
        elif (np.array_equal(direction, dir_downleft)):
            action_mask[[0, 1, 2, 3, 7]] = 0
        # down-right movement
        elif (np.array_equal(direction, dir_downright)):
            action_mask[[0, 1, 5, 6, 7]] = 0
        # upwards movement
        elif np.array_equal(direction, dir_up):
            action_mask[[3, 4, 5]] = 0
        # downwards movement
        elif np.array_equal(direction, dir_down):
            action_mask[[0, 1, 7]] = 0
        # left movement
        elif np.array_equal(direction, dir_left):
            action_mask[[1, 2, 3]] = 0
        # right movement
        elif np.array_equal(direction, dir_right):
            action_mask[[5, 6, 7]] = 0
            
        return action_mask
    
    
    def crossover_parents(self, parent1, parent2):
        # Crossover only if the parents have common stations in the same index
        crossover_points = [i for i in range(min(len(parent1[0]), len(parent2[0]))) if parent1[0][i] == parent2[0][i]]

        if len(crossover_points) > 0:
            crossover_point = random.choice(crossover_points)
            child1 = parent1[0][:crossover_point] + parent2[0][crossover_point:]
            child2 = parent2[0][:crossover_point] + parent1[0][crossover_point:]
            child1_actions = parent1[1][:crossover_point] + parent2[1][crossover_point:]
            child2_actions = parent2[1][:crossover_point] + parent1[1][crossover_point:]
        else:
            child1 = parent1[0].copy()
            child2 = parent2[0].copy()
            child1_actions = parent1[1].copy()
            child2_actions = parent2[1].copy()
            
        return child1, child2, child1_actions, child2_actions
    
    
    def mutate_path(self, child, child_actions, mutation_rng):
        if len(child) < 3:
            return child, []
        
        # Get the mutation point
        mutation_point = mutation_rng.randint(1, len(child) - 2)  # Don't mutate the first or last position

        # Get the previous and next states (locations)
        prev_location = child[mutation_point - 1]
        next_location = child[mutation_point + 1]
        
        current_location_grid = self.env.unwrapped.city.index_to_grid(child[mutation_point])
        prev_location_grid = self.env.unwrapped.city.index_to_grid(prev_location)
        next_location_grid = self.env.unwrapped.city.index_to_grid(next_location)

        action_mask = np.ones(8)
        # Do not allow the action that leads to the mutation point
        action_mask[child_actions[mutation_point - 1]] = 0
        
        # Calculate the direction of the previous and next states, to determine the direction of the mutation
        direction = self.get_direction(next_location_grid, [self.env.unwrapped.city.index_to_grid(loc) for loc in child[:mutation_point + 1]])
        action_mask = self.update_action_mask_for_direction(action_mask, direction)
        
        # Check what actions are possible from the previous location
        possible_next_locations = prev_location_grid + ACTION_TO_DIRECTION[action_mask.astype(bool)]
        possible_next_locations = [loc for loc in possible_next_locations if not np.array_equal(loc, current_location_grid)]
        
        # Check that the possible next locations are within the grid and not already in the path
        possible_next_locations = [loc for loc in possible_next_locations if loc[0] >= 0 and loc[0] < self.env.unwrapped.city.grid_x_size and loc[1] >= 0 and loc[1] < self.env.unwrapped.city.grid_y_size]
        
        
        if len(possible_next_locations) == 0:
            return child, child_actions
        
        # Randomly choose a location from the possible next locations
        new_location_grid = mutation_rng.choice(possible_next_locations)
        new_location = self.env.unwrapped.city.grid_to_index(new_location_grid[None, :])[0]

        # To ensure continuity, fill the cells between the mutated point and the new location
        # We need to "interpolate" the steps between prev_location and new_location
        current_location = int(new_location)
        current_location_grid = new_location_grid

        # Create a list of intermediate cells
        intermediate_cells = [current_location]
        
        # Move step-by-step from the previous location to the new one
        while current_location != next_location:
            direction = np.array(next_location_grid) - np.array(current_location_grid)
            direction[direction > 0] = 1
            direction[direction < 0] = -1
            current_location = int(self.env.unwrapped.city.grid_to_index(np.array(current_location_grid) + direction))
            current_location_grid = self.env.unwrapped.city.index_to_grid(current_location)

            if current_location != next_location:
                intermediate_cells.append(current_location)

        # Update the path with the new location
        child = child[:mutation_point] + intermediate_cells + child[mutation_point + 1:]

        new_actions = []
        for i in range(len(child) - 1):
            direction  = self.env.unwrapped.city.index_to_grid(child[i+1]) - self.env.unwrapped.city.index_to_grid(child[i])
            direction[direction > 0] = 1
            direction[direction < 0] = -1
            new_actions.append(self.get_action_for_direction(direction[0]))
        
        
        return child, new_actions
    

    def get_action_for_direction(self, direction):
        # This method should return the corresponding action for the given direction
        if np.array_equal(direction, dir_up):
            return 0  # Example: corresponding action index
        elif np.array_equal(direction, dir_upright):
            return 1
        elif np.array_equal(direction, dir_right):
            return 2
        elif np.array_equal(direction, dir_downright):
            return 3
        elif np.array_equal(direction, dir_down):
            return 4
        elif np.array_equal(direction, dir_downleft):
            return 5
        elif np.array_equal(direction, dir_left):
            return 6
        elif np.array_equal(direction, dir_upleft):
            return 7
        return None

    def run(self, reward_type, starting_loc=None):
        if starting_loc is not None:
            raise ValueError("Starting location is not implemented yet")
        
        wandb.config['reward_type'] = reward_type
        
        generation = 0
        best_episode_reward = 0
        best_episode_cells = []
        # Frequency of starting locations in the grid
        starting_loc_freq = np.zeros((self.env.unwrapped.city.grid_x_size, self.env.unwrapped.city.grid_y_size))
        starting_loc_avg_reward = np.zeros((self.env.unwrapped.city.grid_x_size, self.env.unwrapped.city.grid_y_size))
        state_visit_freq = np.zeros((self.env.unwrapped.city.grid_x_size, self.env.unwrapped.city.grid_y_size))
        
        # Initialize the population
        pop = []
        start_loc_rng = random.Random(self.seed)
        crossover_rng = random.Random(self.seed)
        mutation_rng = random.Random(self.seed)
        self.env.action_space.seed(self.seed)
        for i in range(self.init_pop_size):
            episode_states = []
            episode_actions = []
            
            starting_loc = (start_loc_rng.randint(0, self.env.unwrapped.city.grid_x_size-1), start_loc_rng.randint(0, self.env.unwrapped.city.grid_y_size-1))
            
            if i == 0:
                obs, info = self.env.reset(seed=self.seed, options={'loc': starting_loc})
                state = info['location_grid_index']
            else:
                obs, info = self.env.reset(options={'loc': starting_loc})
                state = info['location_grid_index']
            
            actual_starting_loc = info['location_grid_coordinates'].tolist()
            starting_loc_freq[actual_starting_loc[0], actual_starting_loc[1]] += 1
            state_visit_freq[actual_starting_loc[0], actual_starting_loc[1]] += 1
            episode_reward = 0
            episode_step = 0
            
            while True:
                action = self.env.action_space.sample(mask=info['action_mask'])
                new_obs, reward, done, _, info = self.env.step(action)
                new_state = info['location_grid_index']
                episode_reward += self.calculate_reward(reward, reward_type)
                
                episode_states.append(state)
                episode_actions.append(action)
                
                episode_step += 1
                state = new_state
                state_visit_freq[info['location_grid_coordinates'][0].item(), info['location_grid_coordinates'][0].item()] += 1
                if done:
                    episode_states.append(state)
                    break
            
            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_cells = info['covered_cells_coordinates']
            
            pop.append((episode_states, episode_actions, episode_reward))
            
        if self.log:
            wandb.log(
                {
                    "generation": generation,
                    "average_reward": np.mean([p[2] for p in pop]),
                    "best_episode_reward": best_episode_reward,
                    "best_episode_cells": best_episode_cells,
                })

        for generation in range(1, self.generations+1):
            print(f'Generation: {generation}')
            # Sort the population based on the reward
            fitness_values = np.array([p[2] for p in pop])
            total_fitness = fitness_values.sum()
            cumsum_fitness = np.cumsum(fitness_values)

            new_pop = []
            for i in range(self.init_pop_size):
                parent_picks = np.random.uniform(0, total_fitness, size=2)
                parent1 = pop[min(np.searchsorted(cumsum_fitness, parent_picks[0]), len(pop) - 1)]
                parent2 = pop[min(np.searchsorted(cumsum_fitness, parent_picks[1]), len(pop) - 1)]
                
                if crossover_rng.random() < self.crossover_rate:
                    child1, child2, child1_actions, child2_actions = self.crossover_parents(parent1, parent2)
                else:
                    child1 = parent1[0].copy()
                    child2 = parent2[0].copy()
                    child1_actions = parent1[1].copy()
                    child2_actions = parent2[1].copy()

                # Mutate
                if mutation_rng.random() < self.mutation_rate:
                    child1, child1_actions = self.mutate_path(child1, child1_actions, mutation_rng)
                    child2, child2_actions = self.mutate_path(child2, child2_actions, mutation_rng)
                    
                # Ensure max length
                child1 = child1[:self.nr_stations]
                child2 = child2[:self.nr_stations]
                                
                # For every child, remove duplicates
                child1 = list(dict.fromkeys(child1))
                child2 = list(dict.fromkeys(child2))
                    
                # Evaluate the children
                for child in [(child1, child1_actions), (child2, child2_actions)]:
                    valid = True
                    state, info = self.env.reset(options={'loc': tuple(map(tuple, self.env.unwrapped.city.index_to_grid(child[0][0])))[0]})
                    episode_reward = 0
                    for action in child[1]:
                        try:
                            new_obs, reward, done, _, info = self.env.step(action)
                            new_state = info['location_grid_index']
                        except:
                            valid = False
                            # print(f"{child[0]} is invalid, actions {child[1]}, from parents {parent1[0]} and {parent2[0]}")
                            break
                            
                        episode_reward += self.calculate_reward(reward, reward_type)
                        state = new_state
                        if done:
                            break
                    if valid:
                        new_pop.append((child[0], child[1], episode_reward))
                        if episode_reward > best_episode_reward:
                            best_episode_reward = episode_reward
                            best_episode_cells = info['covered_cells_coordinates']
                    
                    
            pop = new_pop
            if self.log:
                wandb.log(
                    {
                        "generation": generation,
                        "average_reward": np.mean([p[2] for p in pop]),
                        "best_episode_reward": best_episode_reward,
                        "best_episode_cells": best_episode_cells,
                    })
            
        if self.log:
            # Plot the starting location frequency
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(starting_loc_freq, label='Starting Locations', cmap='viridis')
            for i in range(starting_loc_freq.shape[0]):
                for j in range(starting_loc_freq.shape[1]):
                    ax.text(j, i, int(starting_loc_freq[i, j]),
                                ha="center", va="center", color="w", fontsize=6)
            fig.colorbar(im)
            fig.suptitle('Starting Locations Frequency')
            wandb.log({"Starting-Locations-Frequency": wandb.Image(fig)})
            plt.close(fig)
            
            # Plot average reward of starting locations
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(starting_loc_avg_reward, label='Average reward of starting locs', cmap='Blues')
            fig.colorbar(im)
            fig.suptitle('Average Reward when starting at a location')
            wandb.log({"Avg-Reward-Starting-Locations-Table": wandb.Image(fig)})
            plt.close(fig)
                        
            # Plot the state visitation frequency
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(state_visit_freq, label='States', cmap='viridis')
            for i in range(state_visit_freq.shape[0]):
                for j in range(state_visit_freq.shape[1]):
                    ax.text(j, i, int(state_visit_freq[i, j]),
                                ha="center", va="center", color="w", fontsize=6)
            fig.colorbar(im)
            fig.suptitle('State Visitation Frequency')
            wandb.log({"State-Visitation-Frequency": wandb.Image(fig)})
            plt.close(fig)
            
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
    
    