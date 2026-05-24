#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=ga_tndp
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
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python run_ga.py --env xian --nr_stations=47 --nr_groups=1 --od_type=abs --chained_reward --reward_type=max_efficiency --mutation_rate=0.05 --crossover_rate=0.6 --generations=50 --seed=$SEED"

# Rawls
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python run_ga.py --env xian --nr_stations=47 --nr_groups=5 --od_type=abs --chained_reward --reward_type=rawls --mutation_rate=0.05 --crossover_rate=0.6 --generations=50 --seed=$SEED"

# GGI4
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python run_ga.py --env xian --nr_stations=47 --nr_groups=5 --od_type=abs --chained_reward --reward_type=ggi4 --mutation_rate=0.05 --crossover_rate=0.6 --generations=50 --seed=$SEED"

# GGI2
# CMD="/home/dmichai/anaconda3/envs/tabular-tndp/bin/python run_ga.py --env xian --nr_stations=47 --nr_groups=5 --od_type=abs --chained_reward --reward_type=ggi2 --mutation_rate=0.05 --crossover_rate=0.6 --generations=50 --seed=$SEED"


$CMD