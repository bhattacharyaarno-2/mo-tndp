import argparse
import os
from pathlib import Path
import random
import mo_gymnasium as mo_gym
import numpy as np
import wandb
import wandb.sdk
import yaml
from motndp.city import City
from motndp.constraints import MetroConstraints

from gymnasium.envs.registration import register

from qlearning_tndp import QLearningTNDP

register(
    id="motndp_dilemma-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_amsterdam-v0",
    entry_point="motndp.motndp:MOTNDP",
)

register(
    id="motndp_xian-v0",
    entry_point="motndp.motndp:MOTNDP",
)


import os
import yaml
import wandb

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def train(seed, args, config):
    def make_env(gym_env):
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
        
        env = mo_gym.make(gym_env, 
                        city=city, 
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        od_type=args.od_type,
                        chained_reward=args.chained_reward,)

        return env

    with wandb.init(project=args.project_name, config=config) as run:
        # Set the seed
        seed_everything(seed)

        env = make_env(args.env_id)

        # Launch the agent training
        print(f"Seed {seed}. Training agent...")
        
        config.env_id = args.env_id
        config.od_type = args.od_type
        config.nr_stations = args.nr_stations
        config.nr_groups = args.nr_groups
        config.chained_reward = args.chained_reward
        config.ignore_existing_lines = args.ignore_existing_lines
            
        agent = QLearningTNDP(
            env,
            alpha=config.alpha,
            gamma=config.gamma,
            exploration_type=config.exploration_type,
            initial_epsilon=config.initial_epsilon,
            final_epsilon=config.final_epsilon,
            epsilon_warmup_steps=config.epsilon_warmup_steps,
            epsilon_decay_steps=config.epsilon_decay_steps,
            q_start_initial_value=config.q_start_initial_value,
            q_initial_value=config.q_initial_value,
            train_episodes=config.train_episodes,
            test_episodes=config.test_episodes,
            nr_stations=config.nr_stations,
            nr_groups = config.nr_groups,
            seed=seed,
            wandb_project_name=args.project_name,
            ucb_c_qstart=config.ucb_c_qstart,
            ucb_c_q=config.ucb_c_q,
            update_method=config.update_method
        )
            
        agent.train(args.reward_type, args.starting_loc)
            
def main(args, seeds):
    config_file = os.path.join(args.config_path)

    # Set up the default hyperparameters
    with open(config_file) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)

    # Set up the sweep -- if a sweep id is provided, use it, otherwise create a new sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, entity=args.wandb_entity, project=args.project_name)

    # Define a wrapper function for wandb.agent
    def sweep_wrapper():
        # Initialize a new wandb run
        with wandb.init() as run:
            # Get the current configuration
            config = run.config
            # Call the train function with the current configuration
            train(seeds[0], args, config)

    # Start the sweep agent
    wandb.agent(sweep_id, function=sweep_wrapper, count=args.sweep_count, entity=args.wandb_entity, project=args.project_name)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, help="environment", required=True)
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity to use for the sweep", required=False)
    parser.add_argument("--project-name", type=str, help="Project name to use for the sweep", default="MORL-Baselines")
    parser.add_argument("--sweep-id", type=str, help="Sweep id to use if it already exists (helpful to parallelize the search)", required=False)
    parser.add_argument("--sweep-count", type=int, help="Number of trials to do in the sweep worker", default=10)
    parser.add_argument("--num-seeds", type=int, help="Number of seeds to use for the sweep", default=3)
    parser.add_argument("--seed", type=int, help="Random seed to start from, seeds will be in [seed, seed+num-seeds)", default=42)
    parser.add_argument("--config-path", type=str, help="path of config file.")
    parser.add_argument('--nr_groups', default=2, type=int)
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    parser.add_argument('--ignore_existing_lines', action='store_true', default=False)
    parser.add_argument('--od_type', default='pct', type=str, choices=['pct', 'abs'])
    parser.add_argument('--chained_reward', action='store_true', default=False)
    parser.add_argument('--reward_type', default='max_efficiency', type=str, choices=['max_efficiency'])
    
    Path("./q_tables").mkdir(parents=True, exist_ok=True)

    args = parser.parse_args()
    
    args.project_name = "TNDP-RL"
    args.ignore_existing_lines = args.ignore_existing_lines
    args.od_type = args.od_type
    args.chained_reward = args.chained_reward
    
    if args.env == 'dilemma':
        args.env_id = 'motndp_dilemma-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.groups_file = "groups.txt"
        args.experiment_name = "Q-Learning-Dilemma"
    elif args.env == 'xian':
        args.env_id = 'motndp_xian-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.nr_stations = 47
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.experiment_name = "Q-Learning-Dilemma"
    elif args.env == 'amsterdam':
        args.env_id = 'motndp_amsterdam-v0'
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = 20
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.experiment_name = "Q-Learning-Amsterdam"
        
    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None
        
    # Create an array of seeds to use for the sweep
    seeds = [args.seed + i for i in range(args.num_seeds)]

    main(args, seeds)