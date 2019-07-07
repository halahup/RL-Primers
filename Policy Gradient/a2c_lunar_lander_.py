# TODO: Implement the A2C and A3C models for the lunar lander
# TODO: Figure out the discounted rewards for A2C
# TODO: Double check all the computational graphs
# TODO: Try normalizing state data
# TODO: Checkout the idea of the not full episodes
# TODO: Investigate how parallel environments result in de-correlated samples


from model import Actor, Critic
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_value_
from collections import deque
import matplotlib.pyplot as plt


ALPHA_ACTOR = 0.001        # learning rate
ALPHA_CRITIC = 0.001
BATCH_SIZE = 25           # how many episodes we want to pack into an epoch
GAMMA = 0.99              # discount rate
HIDDEN_SIZE_ACTOR = 64    # number of hidden nodes we have in the actor network
HIDDEN_SIZE_CRITIC = 64   # number of hidden nodes we have in the critic network
BETA = 0.02               # multiplier for the entropy bonus
STEPS = 4                 # number of steps we want to calculate the discounted rewards forward for


def normalize(array: np.array) -> np.array:
    return (array - np.mean(array)) / np.std(array)


def get_discounted_rewards(rewards, states: np.array, value_network: nn.Module, steps: int = None,) -> np.array:
    discounted_rewards = np.empty_like(rewards, dtype=np.float)
    last_state = torch.tensor(states[-1], dtype=torch.float)
    value_of_the_last_state = value_network(last_state).detach().item()

    for i in range(rewards.shape[0]):
        gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=0.99)
        discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
        discounted_reward = np.sum(rewards[i:] * discounted_gammas)
        if i < rewards.shape[0] - 1:
            discounted_rewards[i] = discounted_reward
        else:
            discounted_rewards[i] = value_of_the_last_state ** discounted_gammas[-1]

    return discounted_rewards


def play_episode(env: gym.Env, policy_network: nn.Module, steps: int = None) -> np.array:
    """
        Plays a single episode (or less) of the environment under policy coming from the policy network
        Args:
            env: environment to get the observations from
            policy_network: the actor network to decide on the actions
            steps: how many steps to play
        Returns: the trajectory of the episode
            rewards: reward sequence obtained in the process of the episode
            states: state sequence obtained during the episode
            actions: action sequence that has been taken
    """
    if not steps:
        rewards = np.empty((0, ), dtype=np.float)
        states = np.empty((0, 8), dtype=np.float)
        actions = np.empty((0, 1), dtype=np.float)
        state = env.reset()
        while True:
            states = np.concatenate((states, state.reshape(1, 8)), axis=0)
            action_logits = policy_network(torch.tensor(state).float().unsqueeze(dim=0))
            action = Categorical(logits=action_logits).sample()

            actions = np.concatenate((actions, np.expand_dims(action.item(), axis=0).reshape((1, 1))), axis=0)

            state, reward, done, _ = env.step(action=action.item())
            rewards = np.concatenate((rewards, np.expand_dims(reward, axis=0)))
            if done:
                states = np.concatenate((states, state.reshape(1, 8)), axis=0)
                return rewards, states, actions.squeeze()


def calculate_entropy_bonus(logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    p = softmax(logits, dim=1)
    log_p = log_softmax(logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    entropy_bonus = -1 * BETA * entropy
    return entropy_bonus, entropy


def calculate_policy_loss(actions: torch.Tensor, weights: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:

    # create the one hot mask
    masks = one_hot(actions.detach(), num_classes=logits.shape[1])

    # calculate the log-probabilities of the corresponding chosen action
    # and sum them up across the first dimension to for a vector
    log_probs = torch.sum(masks.float() * log_softmax(logits, dim=1), dim=1)
    policy_loss = -1 * torch.mean(log_probs * weights.squeeze())
    return policy_loss


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter(comment=f'_Gamma={GAMMA},'
                           f'LR={ALPHA_ACTOR}/{ALPHA_CRITIC},BS={BATCH_SIZE},'
                           f'NHA={HIDDEN_SIZE_ACTOR},NHC={HIDDEN_SIZE_CRITIC},'
                           f'STEPS={STEPS}')

    # create the environment
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v1')

    # actor approximates the policy -> number of actions logits are the output
    actor = Actor(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE_ACTOR)

    # critic approximates the value of the state -> 1 logit is the output -> regression task
    critic = Critic(observation_space_size=env.observation_space.shape[0],
                    hidden_size=HIDDEN_SIZE_CRITIC)

    # define the optimizer
    actor_adam = optim.Adam(params=actor.parameters(), lr=ALPHA_ACTOR, eps=1e-8)
    critic_adam = optim.Adam(params=critic.parameters(), lr=ALPHA_CRITIC, eps=1e-8)

    # deque to calculate the average rewards over 100 steps
    rewards_over_100_episodes = deque([], maxlen=100)

    # every epoch
    while True:
        for episode in range(BATCH_SIZE):
            rewards_episode, states_episode, actions_episode = play_episode(env=env, policy_network=actor)

            discounted_rewards = get_discounted_rewards(rewards=rewards_episode,
                                                        states=states_episode,
                                                        steps=STEPS,
                                                        value_network=critic)
            print(rewards_episode)
            print(discounted_rewards)

            raise NotImplementedError


    # close the environment
    env.close()

    # close the writer
    writer.close()


if __name__ == "__main__":
    main()
