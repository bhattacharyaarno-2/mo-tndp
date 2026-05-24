#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=qlearning_tndp
#SBATCH --partition=all
##SBATCH --partition=all6000
##SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

if [ -z "$1" ]; then
    echo "Error: Missing required argument <seed>"
    echo "Usage: $0 <seed>"
    exit 1
fi

SEED=$1

# Max Efficiency
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python train_qlearning.py --env amsterdam --nr_stations=21 --nr_groups=1 --train_episodes=25000 --epsilon_decay_steps=14000 --epsilon_warmup_steps 0 --test_episodes=1 --initial_epsilon=1 --final_epsilon=0.01 --alpha=0.1 --gamma=1 --od_type=abs --chained_reward --reward_type=max_efficiency --exploration_type=egreedy --update_method mc --seed=$SEED"

# Rawls
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python train_qlearning.py --env amsterdam --nr_stations=26 --nr_groups=5 --train_episodes=25000 --epsilon_decay_steps=14000 --epsilon_warmup_steps 0 --test_episodes=1 --initial_epsilon=1 --final_epsilon=0.01 --alpha=0.1 --gamma=1 --od_type=abs --chained_reward --reward_type=rawls --exploration_type=egreedy --update_method mc --seed=$SEED"

# GGI(2)
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python train_qlearning.py --env amsterdam --nr_stations=21 --nr_groups=5 --train_episodes=25000 --epsilon_decay_steps=14000 --epsilon_warmup_steps 0 --test_episodes=1 --initial_epsilon=1 --final_epsilon=0.01 --alpha=0.1 --gamma=1 --od_type=abs --chained_reward --reward_type=ggi2 --exploration_type=egreedy --update_method mc --seed=$SEED"

# GGI(4)
CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python train_qlearning.py --env amsterdam --nr_stations=21 --nr_groups=5 --train_episodes=25000 --epsilon_decay_steps=14000 --epsilon_warmup_steps 0 --test_episodes=1 --initial_epsilon=1 --final_epsilon=0.01 --alpha=0.1 --gamma=1 --od_type=abs --chained_reward --reward_type=ggi4 --exploration_type=egreedy --update_method mc --seed=$SEED"

$CMD