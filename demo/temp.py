import argparse
import os
import uuid
import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import A2C
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Callable, Tuple, Union, Dict
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym
from gymnasium import spaces
from algorithms.wrappers import StableBaselineWrapperNum
from environment.mind.configs import get_enviroment_from_args, get_base_parser
from environment import load_LLM
from algorithms.logging_config import get_logger

logger = get_logger("suber_logger")

# Set environment variable to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define arguments
def parse_args():
    parser = get_base_parser()
    parser.add_argument("--model-device", type=str, default="cuda:0")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args()
    return args

def linear_schedule(initial_value: float):
    def func(progress: float) -> float:
        return initial_value * progress
    return func

class CombinedCallback(BaseCallback):
    def __init__(self, save_freq=2000, log_freq=500, save_path="./tmp/models/", name_prefix="rl_model", verbose=0):
        super(CombinedCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.metrics = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            rewards = self.locals['rewards']
            episode_length = self.locals.get('episode_lengths', None)
            value_loss = self.locals.get('value_loss', None)
            policy_loss = self.locals.get('policy_loss', None)
            if value_loss is not None:
                self.metrics.append({
                    "step": self.num_timesteps,
                    "reward": rewards,
                    "episode_length": episode_length,
                    "value_loss": value_loss,
                    "policy_loss": policy_loss
                })
            # Print learning rate for debugging
            if self.model.policy.optimizer:
                current_lr = self.model.policy.optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")

        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/{self.name_prefix}_{self.num_timesteps}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True

class Net(nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, num_users: int, num_items: int, learning_rate: float = 0.001):
        super().__init__()
        embedding_dim = args.embedding_dim
        self.latent_dim_pi = embedding_dim * 2
        self.latent_dim_vf = embedding_dim * 2

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)

        self.policy_net = nn.Sequential(
            nn.Linear(self.user_embedding.embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, num_items)
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.user_embedding.embedding_dim + num_items, self.latent_dim_vf * 2),
            nn.ReLU(),
            nn.Linear(self.latent_dim_vf * 2, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, self.latent_dim_vf),
            nn.ReLU()
        )

    def forward(self, features: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        user_id = features["user_id"].squeeze(1)
        news_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, news_seen], dim=1)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits, self.value_net(user_embedding_value)

    def forward_actor(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        user_embedding = self.user_embedding(user_id)
        user_bias = self.user_bias(user_id)

        mask = features["items_interact"].to(dtype=torch.bool)
        logits = self.policy_net(user_embedding) + user_bias
        logits[mask] = -torch.inf
        return logits

    def forward_critic(self, features: TensorDict) -> torch.Tensor:
        user_id = features["user_id"].squeeze(1)
        news_seen = features["items_interact"]

        user_embedding = self.user_embedding(user_id)
        user_embedding_value = torch.cat([user_embedding, news_seen], dim=1)
        return self.value_net(user_embedding_value)

class DistributionUseLogitsDirectly(CategoricalDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity(latent_dim)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Callable[[float], float], *args, **kwargs):
        kwargs["ortho_init"] = True
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        self.action_dist = DistributionUseLogitsDirectly(action_space.n)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        default_lr = 0.01
        self.mlp_extractor = Net(self.observation_space, train_env.num_users, train_env.num_items, learning_rate=default_lr)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.mlp_extractor.optimizer = optimizer
        print("Optimizer set in policy.")

    def step_scheduler(self):
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
            print("Scheduler stepped.")

class ExtractPass(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations["user_id"] = observations["user_id"].int()
        return observations

if __name__ == "__main__":
    args = parse_args()
    llm = load_LLM(args.llm_model)

    dir_name = f"{args.llm_model}_{args.llm_rater}_{args.items_retrieval}_{args.user_dataset}_{args.news_dataset}_{args.perturbator}_{args.reward_shaping}_{args.seed}_{args.model_device}_{args.gamma}_{args.embedding_dim}_{args.learning_rate}"
    sanitized_dir_name = dir_name.replace('/', '_').replace(':', '_').replace('.', '_')
    save_path = f"./tmp/models/{sanitized_dir_name}"
    wandb_path = f"./tmp/wandb"
    os.makedirs(save_path, exist_ok=True)

    train_env = get_enviroment_from_args(llm, args)
    test_env = get_enviroment_from_args(llm, args, seed=args.seed + 600)

    policy_kwargs = dict(features_extractor_class=ExtractPass)
    train_env = StableBaselineWrapperNum(train_env)
    test_env = Monitor(StableBaselineWrapperNum(test_env))

    check_env(train_env)
    check_env(test_env)

    model = A2C(CustomActorCriticPolicy, train_env, verbose=1, policy_kwargs=policy_kwargs, device=args.model_device, learning_rate=args.learning_rate, tensorboard_log=save_path, gamma=args.gamma, ent_coef=0.01)

    optimizer = optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

    model.policy.set_optimizer(optimizer)
    model.policy.scheduler = scheduler

    combined_callback = CombinedCallback(save_freq=10000, log_freq=100, save_path=save_path, name_prefix="rl_model", verbose=1)
    callback = CallbackList([combined_callback])

    logger.info("Model starts learning")
    for i in range(0, 10000, 100):
        model.learn(total_timesteps=100, reset_num_timesteps=False, callback=callback, tb_log_name="t_logs")
        model.policy.step_scheduler()  # Step the scheduler manually
        print(f"Scheduler stepped at step {i}")


    logger.info("Model Ends Learning")

    logger.info("Evaluating the Policy")
    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=100)
    logger.info(f"Mean reward: {mean_reward} +/- {std_reward}")

    reward_file_path = os.path.join(save_path, f"reward_{mean_reward:.2f}.txt")
    with open(reward_file_path, 'w') as file:
        file.write(f"Mean reward: {mean_reward} +/- {std_reward}\n")

    print(f"Reward information saved to {reward_file_path}")
    print(f"Mean Reward: {mean_reward}")
    logger.info(f"Mean Reward: {mean_reward}")
