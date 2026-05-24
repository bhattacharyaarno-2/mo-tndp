# tabular-tndp
Tabular Q-Learning for TNDP

## Deep RL

The repository now also includes a PyTorch-based deep RL path for MO-TNDP:

```powershell
python train_deep_rl.py --env amsterdam --nr_groups 5 --nr_stations 20 --reward_type max_efficiency
```

This keeps the same environment setup and reward scalarization options as the tabular pipeline, but replaces the Q-tables with:

- a neural Q-network for action selection
- a neural start-value network for choosing the episode's starting cell
- replay memory and a target network for DQN-style updates

By default the deep RL trainer now uses a two-network stabilization schedule:

- `--random_start_episodes 300` samples valid start cells randomly while training only the action Q-network.
- `--freeze_cycle_episodes 500` alternates between training the action Q-network and the start-value network.
- Set `--freeze_cycle_episodes 0` to return to joint updates every episode.
