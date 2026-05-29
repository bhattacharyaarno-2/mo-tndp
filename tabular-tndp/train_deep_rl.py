import argparse
from pathlib import Path
import random
import sys

import envs
import mo_gymnasium as mo_gym
import numpy as np
import torch

from deep_rl_tndp import DeepQLearningTNDP
from motndp.city import City
from motndp.constraints import MetroConstraints


DEFAULTS = {
    "nr_stations": 10,
    "learning_rate": 1e-3,
    "start_learning_rate": 5e-4,
    "final_epsilon": 0.05,
    "epsilon_decay_steps": 2000,
    "train_episodes": 3000,
    "test_episodes": 5,
    "batch_size": 128,
    "replay_buffer_size": 50000,
    "target_update_freq": 250,
    "learning_starts": 500,
    "hidden_dims": [256, 256],
    "start_hidden_dims": [64, 64],
    "random_start_episodes": 300,
    "freeze_cycle_episodes": 500,
    "reward_type": "max_efficiency",
    "chained_reward": False,
    "run_name": "deep-rl-motndp",
    "no_data_penalty": 0.0,
    "existing_connection_bonus": 0.0,
    "existing_overlap_penalty": 0.0,
}


def resolve_city_path(city_name):
    candidates = [
        Path(f"./cities/{city_name}"),
        Path(f"./envs/mo-tndp/cities/{city_name}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find city data for {city_name}. Checked: {candidates}")


def build_env(args):
    city = City(
        args.city_path,
        groups_file=args.groups_file,
        ignore_existing_lines=args.ignore_existing_lines,
    )
    return mo_gym.make(
        args.gym_env,
        city=city,
        constraints=MetroConstraints(city),
        nr_stations=args.nr_stations,
        od_type=args.od_type,
        chained_reward=args.chained_reward,
    )


def configure_env_args(args):
    args.project_name = "TNDP-RL"

    if args.env == "dilemma":
        args.city_path = resolve_city_path("dilemma_5x5")
        args.nr_stations = 9
        args.gym_env = "motndp_dilemma-v0"
        args.groups_file = "groups.txt"
        args.experiment_name = "Deep-RL-Dilemma"
    elif args.env == "margins":
        args.city_path = resolve_city_path("margins_5x5")
        args.nr_stations = 9
        args.gym_env = "motndp_margins-v0"
        args.groups_file = "groups.txt"
        args.experiment_name = "Deep-RL-Margins"
    elif args.env == "amsterdam":
        args.city_path = resolve_city_path("amsterdam")
        args.gym_env = "motndp_amsterdam-v0"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.experiment_name = "Deep-RL-Amsterdam"
        if args.nr_stations == DEFAULTS["nr_stations"]:
            args.nr_stations = 30
        if args.learning_rate == DEFAULTS["learning_rate"]:
            args.learning_rate = 3e-4
        if args.start_learning_rate == DEFAULTS["start_learning_rate"]:
            args.start_learning_rate = 2e-4
        if args.final_epsilon == DEFAULTS["final_epsilon"]:
            args.final_epsilon = 0.02
        if args.epsilon_decay_steps == DEFAULTS["epsilon_decay_steps"]:
            args.epsilon_decay_steps = 700
        if args.train_episodes == DEFAULTS["train_episodes"]:
            args.train_episodes = 700
        if args.test_episodes == DEFAULTS["test_episodes"]:
            args.test_episodes = 3
        if args.batch_size == DEFAULTS["batch_size"]:
            args.batch_size = 64
        if args.replay_buffer_size == DEFAULTS["replay_buffer_size"]:
            args.replay_buffer_size = 20000
        if args.target_update_freq == DEFAULTS["target_update_freq"]:
            args.target_update_freq = 200
        if args.learning_starts == DEFAULTS["learning_starts"]:
            args.learning_starts = 250
        if args.hidden_dims == DEFAULTS["hidden_dims"]:
            args.hidden_dims = [256, 256]
        if args.start_hidden_dims == DEFAULTS["start_hidden_dims"]:
            args.start_hidden_dims = [64, 64]
        if args.reward_type == DEFAULTS["reward_type"]:
            args.reward_type = "ggi2"
        if args.chained_reward == DEFAULTS["chained_reward"]:
            args.chained_reward = True
        if args.run_name == DEFAULTS["run_name"]:
            args.run_name = "amsterdam-dqn-freeze-v1"
    elif args.env == "amsterdam_10x10":
        args.city_path = resolve_city_path("amsterdam_10x10")
        args.gym_env = "motndp_amsterdam_10x10-v0"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.experiment_name = "Deep-RL-Amsterdam-10x10"
    elif args.env == "xian":
        args.city_path = resolve_city_path("xian")
        args.gym_env = "motndp_xian-v0"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.experiment_name = "Deep-RL-Xian"
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None


def main(args):
    env = build_env(args)
    actual_nr_groups = env.unwrapped.nr_groups

    checkpoint_path = Path("./deep_models") / f"{args.run_name}.pt" if args.run_name else None

    agent = DeepQLearningTNDP(
        env=env,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        train_episodes=args.train_episodes,
        test_episodes=args.test_episodes,
        nr_stations=args.nr_stations,
        nr_groups=actual_nr_groups,
        seed=args.seed,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        epsilon_warmup_steps=args.epsilon_warmup_steps,
        epsilon_decay_steps=args.epsilon_decay_steps,
        replay_buffer_size=args.replay_buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        hidden_dims=args.hidden_dims,
        start_hidden_dims=args.start_hidden_dims,
        start_learning_rate=args.start_learning_rate,
        random_start_episodes=args.random_start_episodes,
        freeze_cycle_episodes=args.freeze_cycle_episodes,
        grad_clip_norm=args.grad_clip_norm,
        tau=args.tau,
        policy=args.policy,
        wandb_project_name=args.project_name,
        wandb_experiment_name=args.experiment_name,
        checkpoint_path=checkpoint_path,
        log=not args.no_log,
        device=args.device,
        no_data_penalty=args.no_data_penalty,
        existing_connection_bonus=args.existing_connection_bonus,
        existing_overlap_penalty=args.existing_overlap_penalty,
    )

    print(
        f"Starting deep RL run '{args.run_name}' "
        f"with reward_type={args.reward_type}, chained_reward={args.chained_reward}, "
        f"no_data_penalty={args.no_data_penalty}, nr_groups={actual_nr_groups}, "
        f"existing_connection_bonus={args.existing_connection_bonus}, "
        f"existing_overlap_penalty={args.existing_overlap_penalty}, "
        f"random_start_episodes={args.random_start_episodes}, "
        f"freeze_cycle_episodes={args.freeze_cycle_episodes}",
        flush=True,
    )
    agent.train(args.reward_type, starting_loc=args.starting_loc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep RL for MO-TNDP")
    parser.add_argument("--env", default="dilemma", type=str)
    parser.add_argument("--nr_groups", default=5, type=int)
    parser.add_argument("--starting_loc_x", default=None, type=int)
    parser.add_argument("--starting_loc_y", default=None, type=int)
    parser.add_argument("--nr_stations", default=10, type=int)
    parser.add_argument("--policy", default=None, type=str)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--start_learning_rate", default=5e-4, type=float)
    parser.add_argument("--initial_epsilon", default=1.0, type=float)
    parser.add_argument("--final_epsilon", default=0.05, type=float)
    parser.add_argument("--epsilon_warmup_steps", default=0, type=int)
    parser.add_argument("--epsilon_decay_steps", default=2000, type=int)
    parser.add_argument("--train_episodes", default=3000, type=int)
    parser.add_argument("--test_episodes", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--replay_buffer_size", default=50000, type=int)
    parser.add_argument("--target_update_freq", default=250, type=int)
    parser.add_argument("--learning_starts", default=500, type=int)
    parser.add_argument("--train_freq", default=1, type=int)
    parser.add_argument("--grad_clip_norm", default=10.0, type=float)
    parser.add_argument("--tau", default=0.05, type=float)
    parser.add_argument("--hidden_dims", nargs="+", default=[256, 256], type=int)
    parser.add_argument("--start_hidden_dims", nargs="+", default=[64, 64], type=int)
    parser.add_argument("--random_start_episodes", default=300, type=int)
    parser.add_argument("--freeze_cycle_episodes", default=500, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--run_name", default="deep-rl-motndp", type=str)
    parser.add_argument("--no_log", action="store_true", default=False)
    parser.add_argument("--ignore_existing_lines", action="store_true", default=False)
    parser.add_argument("--od_type", default="pct", type=str, choices=["pct", "abs"])
    parser.add_argument("--chained_reward", action="store_true", default=False)
    parser.add_argument("--reward_type", default="max_efficiency", type=str, choices=["max_efficiency", "ggi2", "ggi4", "rawls"])
    parser.add_argument("--no_data_penalty", default=0.0, type=float)
    parser.add_argument("--existing_connection_bonus", default=0.0, type=float)
    parser.add_argument("--existing_overlap_penalty", default=0.0, type=float)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()
    configure_env_args(args)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    print(args, flush=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    Path("./eval").mkdir(parents=True, exist_ok=True)
    Path("./deep_models").mkdir(parents=True, exist_ok=True)

    main(args)
