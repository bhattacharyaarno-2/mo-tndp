import envs
from pathlib import Path
import random
import time
from matplotlib import pyplot as plt
import mo_gymnasium as mo_gym
from ga_tndp import GATNDP
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
from qlearning_tndp import QLearningTNDP
# import torch
import argparse
import wandb

def main(args):
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
    
    env = make_env(args.gym_env)
    algo = GATNDP(env, 
                args.init_pop_size, 
                args.mutation_rate, 
                args.crossover_rate, 
                args.generations, 
                args.nr_stations, 
                args.nr_groups,
                args.seed,
                args.project_name, 
                args.experiment_name, 
                log=not args.no_log)
    
    algo.run(args.reward_type, args.starting_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for MO-TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam'
    parser.add_argument('--env', default='dilemma', type=str)
    # For xian/amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    parser.add_argument('--nr_stations', type=int)
    parser.add_argument('--init_pop_size', default=500, type=int)
    parser.add_argument('--mutation_rate', default=0.9, type=float)
    parser.add_argument('--crossover_rate', default=0.9, type=float)
    parser.add_argument('--generations', default=3500, type=int)

    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--ignore_existing_lines', action='store_true', default=False)
    parser.add_argument('--od_type', default='pct', type=str, choices=['pct', 'abs'])
    parser.add_argument('--chained_reward', action='store_true', default=False)
    parser.add_argument('--reward_type', default='max_efficiency', type=str, choices=['max_efficiency', 'ggi2', 'ggi4', 'rawls'])
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    args.project_name = "TNDP-GA"

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'dilemma':
        args.city_path = Path(f"./envs/mo-tndp/cities/dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_dilemma-v0'
        args.groups_file = "groups.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "GA-Dilemma"
    elif args.env == 'margins':
        args.city_path = Path(f"./envs/mo-tndp/cities/margins_5x5")
        args.nr_stations = 9
        args.gym_env = 'motndp_margins-v0'
        args.groups_file = f"groups.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "GA-Margins"
    elif args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.nr_stations = args.nr_stations
        args.gym_env = 'motndp_amsterdam-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "GA-Amsterdam"
    elif args.env == 'xian':
        # Xian pre-defined
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.nr_stations = args.nr_stations
        args.gym_env = 'motndp_xian-v0'
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = args.ignore_existing_lines
        args.experiment_name = "GA-Xian"

    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)
