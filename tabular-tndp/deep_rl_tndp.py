from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time

from codecarbon import EmissionsTracker
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb


def linearly_decaying_value(initial_value, decay_period, step, warmup_steps, final_value):
    steps_left = decay_period + warmup_steps - step
    bonus = (initial_value - final_value) * steps_left / decay_period
    value = final_value + bonus
    value = np.clip(value, min(initial_value, final_value), max(initial_value, final_value))
    return value


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class StartValueNetwork(nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()

        layers = []
        prev_dim = 2
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


@dataclass
class TransitionBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    next_action_masks: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, observation, action, reward, next_observation, done, next_action_mask):
        self.buffer.append(
            (
                np.asarray(observation, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_observation, dtype=np.float32),
                float(done),
                np.asarray(next_action_mask, dtype=np.float32),
            )
        )

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        observations, actions, rewards, next_observations, dones, next_action_masks = zip(*batch)
        return TransitionBatch(
            observations=torch.as_tensor(np.stack(observations), dtype=torch.float32, device=device),
            actions=torch.as_tensor(actions, dtype=torch.int64, device=device),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            next_observations=torch.as_tensor(np.stack(next_observations), dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
            next_action_masks=torch.as_tensor(np.stack(next_action_masks), dtype=torch.float32, device=device),
        )


class DeepQLearningTNDP:
    def __init__(
        self,
        env,
        gamma,
        learning_rate,
        train_episodes,
        test_episodes,
        nr_stations,
        nr_groups,
        seed,
        initial_epsilon,
        final_epsilon,
        epsilon_warmup_steps,
        epsilon_decay_steps,
        replay_buffer_size,
        batch_size,
        target_update_freq,
        learning_starts,
        train_freq,
        hidden_dims,
        start_hidden_dims,
        start_learning_rate,
        random_start_episodes,
        freeze_cycle_episodes,
        grad_clip_norm,
        tau,
        policy=None,
        wandb_project_name=None,
        wandb_experiment_name=None,
        wandb_run_id=None,
        checkpoint_path=None,
        log=True,
        device=None,
        no_data_penalty=0.0,
        existing_connection_bonus=0.0,
        existing_overlap_penalty=0.0,
    ):
        self.env = env
        self.env_id = env.unwrapped.spec.id
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.train_episodes = train_episodes
        self.test_episodes = test_episodes
        self.nr_stations = nr_stations
        self.nr_groups = nr_groups
        self.seed = seed
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_warmup_steps = epsilon_warmup_steps
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.hidden_dims = hidden_dims
        self.start_hidden_dims = start_hidden_dims
        self.start_learning_rate = start_learning_rate
        self.random_start_episodes = int(random_start_episodes)
        self.freeze_cycle_episodes = int(freeze_cycle_episodes)
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau
        self.policy = policy
        self.wandb_project_name = wandb_project_name
        self.wandb_experiment_name = wandb_experiment_name
        self.wandb_run_id = wandb_run_id
        self.checkpoint_path = checkpoint_path
        self.log = log
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.no_data_penalty = float(no_data_penalty)
        self.existing_connection_bonus = float(existing_connection_bonus)
        self.existing_overlap_penalty = float(existing_overlap_penalty)
        self.reward_type = None

        self.observation_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = self.env.action_space.n

        self.q_network = QNetwork(self.observation_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_network = QNetwork(self.observation_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.start_network = StartValueNetwork(self.start_hidden_dims).to(self.device)
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.start_optimizer = torch.optim.Adam(self.start_network.parameters(), lr=self.start_learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.start_location_features = self._build_start_location_features()
        self.valid_start_mask = self._build_valid_start_mask()

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load(checkpoint_path)

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

    def _build_start_location_features(self):
        rows = self.env.unwrapped.city.grid_x_size
        cols = self.env.unwrapped.city.grid_y_size
        coords = []
        for row in range(rows):
            for col in range(cols):
                row_norm = row / max(rows - 1, 1)
                col_norm = col / max(cols - 1, 1)
                coords.append([row_norm, col_norm])
        return torch.as_tensor(coords, dtype=torch.float32, device=self.device)

    def _build_valid_start_mask(self):
        city = self.env.unwrapped.city
        if not hasattr(city, "grid_groups"):
            return None
        valid_mask = np.isfinite(city.grid_groups)
        if not np.any(valid_mask):
            return None
        return valid_mask

    def get_config(self):
        return {
            "env_id": self.env_id,
            "od_type": self.env.unwrapped.od_type,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "start_learning_rate": self.start_learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "final_epsilon": self.final_epsilon,
            "epsilon_warmup_steps": self.epsilon_warmup_steps,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "train_episodes": self.train_episodes,
            "test_episodes": self.test_episodes,
            "nr_stations": self.nr_stations,
            "nr_groups": self.nr_groups,
            "seed": self.seed,
            "policy": self.policy,
            "batch_size": self.batch_size,
            "replay_buffer_size": self.replay_buffer.buffer.maxlen,
            "target_update_freq": self.target_update_freq,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "hidden_dims": self.hidden_dims,
            "start_hidden_dims": self.start_hidden_dims,
            "random_start_episodes": self.random_start_episodes,
            "freeze_cycle_episodes": self.freeze_cycle_episodes,
            "grad_clip_norm": self.grad_clip_norm,
            "tau": self.tau,
            "chained_reward": self.env.unwrapped.chained_reward,
            "ignore_existing_lines": self.env.unwrapped.city.ignore_existing_lines,
            "algorithm": "deep_q_learning",
            "device": str(self.device),
            "no_data_penalty": self.no_data_penalty,
            "existing_connection_bonus": self.existing_connection_bonus,
            "existing_overlap_penalty": self.existing_overlap_penalty,
            "reward_type": self.reward_type,
            "checkpoint_name": Path(self.checkpoint_path).stem if self.checkpoint_path else None,
            "wandb_run_id": wandb.run.id if self.log and wandb.run else self.wandb_run_id,
        }

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

    def calculate_reward(self, reward, reward_type):
        reward = np.asarray(reward, dtype=np.float32)
        if reward_type == "max_efficiency":
            return reward.sum()
        if reward_type == "ggi2":
            return self.ggi_reward(reward, 2)
        if reward_type == "ggi4":
            return self.ggi_reward(reward, 4)
        if reward_type == "rawls":
            return reward[0]
        raise ValueError(f"Reward type {reward_type} not implemented")

    def ggi_reward(self, reward, weight):
        weights = 1 / (weight ** np.arange(reward.shape[0]))
        weights /= weights.sum()
        sorted_reward = np.sort(reward)
        return float(np.sum(sorted_reward * weights))

    def _select_start_location(self, epsilon):
        rows = self.env.unwrapped.city.grid_x_size
        cols = self.env.unwrapped.city.grid_y_size
        valid_start_indices = None
        if self.valid_start_mask is not None:
            valid_start_indices = np.argwhere(self.valid_start_mask)

        if random.random() < epsilon:
            if valid_start_indices is not None and len(valid_start_indices) > 0:
                row, col = valid_start_indices[random.randrange(len(valid_start_indices))]
                return (int(row), int(col)), None
            return (random.randint(0, rows - 1), random.randint(0, cols - 1)), None

        with torch.no_grad():
            values = self.start_network(self.start_location_features).reshape(rows, cols)
            if self.valid_start_mask is not None:
                invalid_locations = torch.as_tensor(~self.valid_start_mask, dtype=torch.bool, device=self.device)
                values = values.masked_fill(invalid_locations, float("-inf"))
            flat_index = int(torch.argmax(values).item())
        return np.unravel_index(flat_index, (rows, cols)), values.detach().cpu().numpy()

    def _select_random_start_location(self):
        rows = self.env.unwrapped.city.grid_x_size
        cols = self.env.unwrapped.city.grid_y_size
        if self.valid_start_mask is not None:
            valid_start_indices = np.argwhere(self.valid_start_mask)
            if len(valid_start_indices) > 0:
                row, col = valid_start_indices[random.randrange(len(valid_start_indices))]
                return (int(row), int(col))
        return (random.randint(0, rows - 1), random.randint(0, cols - 1))

    def _training_phase(self, episode):
        if self.freeze_cycle_episodes <= 0:
            return "joint", True, True
        if episode < self.random_start_episodes:
            return "random_start_warmup", True, False

        phase_index = (episode - self.random_start_episodes) // self.freeze_cycle_episodes
        if phase_index % 2 == 0:
            return "train_action", True, False
        return "train_start", False, True

    def _is_no_data_cell(self, location):
        if self.valid_start_mask is None:
            return False
        row, col = int(location[0]), int(location[1])
        return not bool(self.valid_start_mask[row, col])

    def _existing_line_hits(self, segment_indices):
        city = self.env.unwrapped.city
        hits = []
        if not getattr(city, "existing_lines_full", None):
            return hits

        segment_indices = np.asarray(segment_indices, dtype=np.int64)
        for line_index, line_full in enumerate(city.existing_lines_full):
            if np.isin(segment_indices, np.asarray(line_full).reshape(-1)).any():
                hits.append(line_index)
        return hits

    def _existing_overlap_count(self, segment_indices):
        city = self.env.unwrapped.city
        if not getattr(city, "existing_lines_full", None):
            return 0

        segment_indices = np.asarray(segment_indices, dtype=np.int64)
        existing_cells = np.concatenate([np.asarray(line).reshape(-1) for line in city.existing_lines_full])
        return int(np.isin(segment_indices, existing_cells).sum())

    def _apply_reward_shaping(self, vector_reward, location, segment_indices=None, connected_existing_lines=None):
        scalar_adjustment = 0.0
        shaping_components = {
            "no_data": 0.0,
            "existing_connection": 0.0,
            "existing_overlap": 0.0,
        }
        if self.no_data_penalty > 0 and self._is_no_data_cell(location):
            scalar_adjustment -= self.no_data_penalty
            shaping_components["no_data"] = -self.no_data_penalty

        if segment_indices is not None:
            if self.existing_connection_bonus > 0 and connected_existing_lines is not None:
                new_hits = [line_index for line_index in self._existing_line_hits(segment_indices) if line_index not in connected_existing_lines]
                if new_hits:
                    bonus = self.existing_connection_bonus * len(new_hits)
                    scalar_adjustment += bonus
                    shaping_components["existing_connection"] = bonus
                    connected_existing_lines.update(new_hits)

            if self.existing_overlap_penalty > 0:
                overlap_count = self._existing_overlap_count(segment_indices)
                if overlap_count > 0:
                    penalty = self.existing_overlap_penalty * overlap_count
                    scalar_adjustment -= penalty
                    shaping_components["existing_overlap"] = -penalty

        return np.asarray(vector_reward, dtype=np.float32), float(scalar_adjustment), shaping_components

    def _select_action(self, observation, action_mask, epsilon):
        valid_actions = np.flatnonzero(action_mask)
        if len(valid_actions) == 0:
            return 0, None

        if random.random() < epsilon:
            return int(random.choice(valid_actions.tolist())), None

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(observation_tensor).squeeze(0)
            masked_q_values = q_values.masked_fill(
                torch.as_tensor(action_mask == 0, dtype=torch.bool, device=self.device),
                float("-inf"),
            )
        return int(torch.argmax(masked_q_values).item()), q_values.detach().cpu().numpy()

    def _optimize_q_network(self):
        batch = self.replay_buffer.sample(self.batch_size, self.device)

        current_q_values = self.q_network(batch.observations).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(batch.next_observations)
            invalid_next = batch.next_action_masks == 0
            next_q_values = next_q_values.masked_fill(invalid_next, float("-inf"))
            best_next_q = next_q_values.max(dim=1).values
            best_next_q = torch.where(torch.isfinite(best_next_q), best_next_q, torch.zeros_like(best_next_q))
            targets = batch.rewards + (1.0 - batch.dones) * self.gamma * best_next_q

        loss = F.smooth_l1_loss(current_q_values, targets)
        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip_norm)
        self.q_optimizer.step()
        return float(loss.item())

    def _optimize_start_network(self, start_loc, episode_reward):
        rows = self.env.unwrapped.city.grid_x_size
        cols = self.env.unwrapped.city.grid_y_size
        row_norm = start_loc[0] / max(rows - 1, 1)
        col_norm = start_loc[1] / max(cols - 1, 1)
        feature = torch.as_tensor([[row_norm, col_norm]], dtype=torch.float32, device=self.device)
        target = torch.as_tensor([episode_reward], dtype=torch.float32, device=self.device)
        prediction = self.start_network(feature)
        loss = F.smooth_l1_loss(prediction, target)
        self.start_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.start_network.parameters(), self.grad_clip_norm)
        self.start_optimizer.step()
        return float(loss.item())

    def _soft_update_target_network(self):
        with torch.no_grad():
            for target_param, source_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * source_param.data)

    def save(self, path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "start_network": self.start_network.state_dict(),
                "config": self.get_config(),
            },
            save_path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.start_network.load_state_dict(checkpoint["start_network"])

    def train(self, reward_type, starting_loc=None):
        self.reward_type = reward_type
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(line_buffering=True)
        tracker = EmissionsTracker(output_dir="carbon_logs", project_name=wandb.run.id if self.log else "deep-rl")
        tracker.start()

        if self.log:
            wandb.config.update({"reward_type": reward_type}, allow_val_change=True)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        epsilon = self.initial_epsilon
        training_step = 0
        best_episode_reward = float("-inf")
        best_episode_cells = []
        last_50_rewards = deque(maxlen=50)
        last_q_loss = None
        last_start_loss = None
        train_start_time = time.time()
        phase_codes = {
            "joint": 0,
            "random_start_warmup": 1,
            "train_action": 2,
            "train_start": 3,
        }

        for episode in range(self.train_episodes):
            phase, train_q_network, train_start_network = self._training_phase(episode)
            if starting_loc is not None:
                loc = starting_loc
            elif phase == "random_start_warmup":
                loc = self._select_random_start_location()
            else:
                loc, _ = self._select_start_location(epsilon)

            if episode == 0:
                observation, info = self.env.reset(seed=self.seed, options={"loc": loc})
            else:
                observation, info = self.env.reset(options={"loc": loc})

            actual_start_loc = tuple(np.asarray(info["location_grid_coordinates"]).tolist())
            episode_reward = 0.0
            episode_step = 0
            connected_existing_lines = set()
            shaping_components = {
                "no_data": 0.0,
                "existing_connection": 0.0,
                "existing_overlap": 0.0,
            }

            while True:
                if self.policy:
                    action = self.policy[episode_step]
                else:
                    action, _ = self._select_action(observation, info["action_mask"], epsilon)

                from_idx = int(info["location_grid_index"])
                next_observation, vector_reward, done, _, next_info = self.env.step(action)
                to_idx = int(next_info["location_grid_index"])
                _, scalar_adjustment, shaping_components = self._apply_reward_shaping(
                    vector_reward,
                    next_info["location_grid_coordinates"],
                    segment_indices=[from_idx, to_idx],
                    connected_existing_lines=connected_existing_lines,
                )
                scalar_reward = self.calculate_reward(vector_reward, reward_type) + scalar_adjustment
                self.replay_buffer.add(
                    observation,
                    action,
                    scalar_reward,
                    next_observation,
                    done,
                    next_info["action_mask"],
                )

                if (
                    train_q_network
                    and
                    len(self.replay_buffer) >= self.batch_size
                    and training_step >= self.learning_starts
                    and training_step % self.train_freq == 0
                ):
                    last_q_loss = self._optimize_q_network()

                if train_q_network and training_step > 0 and training_step % self.target_update_freq == 0:
                    self._soft_update_target_network()

                observation = next_observation
                info = next_info
                episode_reward += scalar_reward
                training_step += 1
                episode_step += 1

                if done:
                    break

            if train_start_network:
                last_start_loss = self._optimize_start_network(actual_start_loc, episode_reward)

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                best_episode_cells = info["covered_cells_coordinates"]
                if self.checkpoint_path:
                    self.save(self.checkpoint_path)

            last_50_rewards.append(episode_reward)

            if self.log:
                wandb.log(
                    {
                        "episode": episode,
                        "reward": episode_reward,
                        "average_reward": np.mean(last_50_rewards),
                        "epsilon": epsilon,
                        "training_step": training_step,
                        "best_episode_reward": best_episode_reward,
                        "q_loss": last_q_loss,
                        "start_loss": last_start_loss,
                        "training_phase_code": phase_codes[phase],
                        "train_q_network": int(train_q_network),
                        "train_start_network": int(train_start_network),
                        "penalty_no_data": shaping_components["no_data"],
                        "bonus_existing_connection": shaping_components["existing_connection"],
                        "penalty_existing_overlap": shaping_components["existing_overlap"],
                    }
                )
                wandb.run.summary["training_phase"] = phase
                wandb.run.summary["random_start_episodes"] = self.random_start_episodes
                wandb.run.summary["freeze_cycle_episodes"] = self.freeze_cycle_episodes

            elapsed_seconds = time.time() - train_start_time
            completed_episodes = episode + 1
            average_episode_time = elapsed_seconds / max(completed_episodes, 1)
            remaining_episodes = self.train_episodes - completed_episodes
            eta_seconds = max(0.0, remaining_episodes * average_episode_time)
            running_average = float(np.mean(last_50_rewards)) if last_50_rewards else 0.0

            print(
                f"[train] episode {completed_episodes}/{self.train_episodes} "
                f"| reward={episode_reward:.4f} "
                f"| avg50={running_average:.4f} "
                f"| best={best_episode_reward:.4f} "
                f"| epsilon={epsilon:.3f} "
                f"| phase={phase} "
                f"| steps={training_step} "
                f"| no_data_penalty={shaping_components['no_data']:.4f} "
                f"| existing_bonus={shaping_components['existing_connection']:.4f} "
                f"| existing_overlap={shaping_components['existing_overlap']:.4f} "
                f"| elapsed={elapsed_seconds/60:.1f}m "
                f"| eta={eta_seconds/60:.1f}m",
                flush=True,
            )

            epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                episode,
                self.epsilon_warmup_steps,
                self.final_epsilon,
            )

        tracker.stop()

        if self.checkpoint_path and not Path(self.checkpoint_path).exists():
            self.save(self.checkpoint_path)

        if self.test_episodes > 0:
            self.test(self.test_episodes, reward_type, starting_loc=starting_loc, policy=self.policy)

        if self.log:
            wandb.finish()

        return best_episode_reward, best_episode_cells

    def test(self, test_episodes, reward_type, starting_loc=None, policy=None):
        total_rewards = 0.0
        total_satisfied_ods_by_group = np.zeros(self.nr_groups, dtype=np.float64)
        generated_lines = []

        if starting_loc is not None:
            test_starting_loc = starting_loc
        else:
            test_starting_loc, _ = self._select_start_location(0.0)

        for episode in range(test_episodes):
            observation, info = self.env.reset(options={"loc": test_starting_loc})
            locations = [info["location_grid_coordinates"].tolist()]
            actions = []
            episode_reward = 0.0
            episode_satisfied_ods_by_group = np.zeros(self.nr_groups, dtype=np.float64)
            episode_step = 0
            connected_existing_lines = set()

            while True:
                if policy is not None:
                    action = policy[episode_step]
                else:
                    action, _ = self._select_action(observation, info["action_mask"], 0.0)

                actions.append(action)
                from_idx = int(info["location_grid_index"])
                next_observation, reward, done, _, info = self.env.step(action)
                to_idx = int(info["location_grid_index"])
                locations.append(info["location_grid_coordinates"].tolist())
                episode_satisfied_ods_by_group += reward
                _, scalar_adjustment, _ = self._apply_reward_shaping(
                    reward,
                    info["location_grid_coordinates"],
                    segment_indices=[from_idx, to_idx],
                    connected_existing_lines=connected_existing_lines,
                )
                episode_reward += self.calculate_reward(reward, reward_type) + scalar_adjustment
                observation = next_observation
                episode_step += 1

                if done:
                    break

            total_rewards += episode_reward
            total_satisfied_ods_by_group += episode_satisfied_ods_by_group
            generated_lines.append(locations)

        print(f"Average reward over {test_episodes} episodes: {total_rewards / test_episodes:.4f}")
        print(f"Actions of last episode: {actions}")
        print(f"vids of last episode: {self.env.unwrapped.city.grid_to_index(np.array(locations)).tolist()}")

        if self.log:
            wandb.log({"Average-Test-Reward": total_rewards / test_episodes})
