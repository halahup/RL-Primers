import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax, mse_loss
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_value_
from collections import deque


ALPHA = 0.001              # learning rate for the actor
BETA = 0.001               # learning rate for the critic
GAMMA = 0.99               # discount rate
HIDDEN_SIZE = 32           # number of hidden nodes we have in our approximation
PSI = 0.1                  # the entropy bonus multiplier

NUM_EPISODES = 5000
NUM_STEPS = 4

RENDER_EVERY = 100

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


# Q-table is replaced by a neural network
class Actor(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Critic(nn.Module):
    def __init__(self, observation_space_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        return x


def get_discounted_returns(rewards: np.array, gamma: float, state_values: torch.Tensor, n: int):
    """
        Computes the array of discounted rewards [Gt:t+1] for the episode. See reference on p.143 S&B.
        Args:
            rewards: the sequence of the rewards obtained from running the episode
            gamma: the discounting factor
            state_values: teh values of the states calculated by the critic network
            n: the horizon of the bootstrapping
        Returns:
            discounted_rewards: the sequence of the discounted returns from time step t
    """
    discounted_rewards = np.empty_like(rewards, dtype=np.float)
    gamma_array = np.full(shape=(n,), fill_value=gamma)
    power_gamma_array = np.power(gamma_array, np.arange(n))

    # turn the state values torch tensor into the numpy array
    state_values = state_values.numpy()

    # define the end of sequence
    T = rewards.shape[0]

    # for every time step in the sequence
    for t in range(T):

        # check if we can bootstrap
        if t+n < T:

            # calculate the bootstrapped return
            Gt = np.sum(power_gamma_array[:-1] * rewards[t:(t+n)-1]) + power_gamma_array[-1] * state_values[t+n]

        # if we can't bootstrap anymore
        else:

            # check if we can discount
            if t < T-1:

                # compute the monte carlo return
                Gt = np.sum(power_gamma_array[:rewards[t:T].shape[0]] * rewards[t:T])

            else:

                # the last reward
                Gt = rewards[T-1]

        discounted_rewards[t] = Gt

    return discounted_rewards


def get_entropy_bonus(logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):

    # calculate the probabilities
    p = softmax(logits, dim=1)

    # calculate the log probabilities
    log_p = log_softmax(logits, dim=1)

    # calculate the entropy
    entropy = -1 * torch.sum(p * log_p, dim=1)

    # calculate the mean entropy for the episode
    mean_entropy = torch.mean(entropy, dim=0)

    # calculate the entropy bonus
    entropy_bonus = -1 * BETA * mean_entropy

    return entropy_bonus, mean_entropy


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter(comment=f'_A2C_Gamma={GAMMA},LRA={ALPHA},LRC={BETA},NH={HIDDEN_SIZE},NS={NUM_STEPS}')

    # create the environment
    env = gym.make('LunarLander-v2')

    # Q-table is replaced by the agent driven by a neural network architecture
    actor = Actor(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE)

    actor = actor.to(DEVICE)

    critic = Critic(observation_space_size=env.observation_space.shape[0],
                    hidden_size=HIDDEN_SIZE)

    critic = critic.to(DEVICE)

    adam_actor = optim.Adam(params=actor.parameters(), lr=ALPHA)
    adam_critic = optim.Adam(params=critic.parameters(), lr=BETA)

    total_rewards = deque([], maxlen=100)

    # run for K episodes
    for episode in range(NUM_EPISODES):

        # initialize the environment state
        current_state = env.reset()

        logits = torch.empty(size=(0, env.action_space.n), dtype=torch.float)
        action_log_probs = torch.empty(size=(0,), dtype=torch.float)
        state_values = torch.empty(size=(0,), dtype=torch.float)
        rewards = torch.empty(size=(0,), dtype=torch.float)
        states = np.empty(shape=(0, env.observation_space.shape[0]), dtype=np.float)

        # set the done flag to false
        done = False

        # init the total reward
        episode_total_reward = 0

        # accumulate data for 1 episode
        while not done:

            if episode % RENDER_EVERY == 0:
                env.render()

            # get the action logits from the agent - (preferences)
            action_logits = actor(torch.tensor(current_state).float().unsqueeze(dim=0).to(DEVICE)).squeeze()

            # append the logits
            logits = torch.cat((logits, action_logits.unsqueeze(dim=0)), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # compute the log-probabilities of the actions
            log_probs = log_softmax(action_logits, dim=0)

            # get the log-probability of the chosen action
            action_log_probs = torch.cat((action_log_probs, log_probs[action.item()].unsqueeze(dim=0)), dim=0)

            # get the current state value
            current_state_value = critic(torch.tensor(current_state).float().unsqueeze(dim=0).to(DEVICE))
            state_values = torch.cat((state_values, current_state_value), dim=0)

            # take the action
            new_state, reward, done, _ = env.step(action.item())

            episode_total_reward += reward

            # save the reward
            rewards = torch.cat((rewards, torch.tensor(reward, dtype=torch.float).unsqueeze(dim=0)), dim=0)

            # save the state
            states = np.concatenate((states, np.expand_dims(current_state, axis=0)), axis=0)

            # if the episode is over
            if done:
                total_rewards.append(episode_total_reward)
                break

            # update the state
            current_state = new_state

        # zero the gradient in both actor and the critic networks
        actor.zero_grad()
        critic.zero_grad()

        # calculate the sequence of the discounted returns Gt
        discounted_returns = get_discounted_returns(rewards=rewards.numpy(),
                                                    gamma=0.99,
                                                    state_values=state_values.detach().squeeze(),
                                                    n=NUM_STEPS)

        # turn the discounted returns array into torch tensor
        discounted_returns = torch.tensor(discounted_returns, dtype=torch.float)

        # calculate the advantage for time t: Q(s,a) - V(s)
        advantages = discounted_returns - state_values.detach().squeeze()

        # calculate the policy loss
        policy_loss = -1 * torch.mean(action_log_probs * advantages)

        # get the enrtopy bonus
        entropy_bonus, mean_entropy = get_entropy_bonus(logits=logits)

        # add the entopy bonus
        policy_loss += (PSI * entropy_bonus)

        # calculate the policy gradient
        policy_loss.backward()

        # calculate the critic loss
        critic_loss = mse_loss(input=state_values.squeeze(), target=discounted_returns)

        # calculate the gradient of the critic loss
        critic_loss.backward()

        # clip the gradients in the policy gradients and the critic loss gradients
        clip_grad_value_(parameters=actor.parameters(), clip_value=0.1)
        clip_grad_value_(parameters=critic.parameters(), clip_value=0.1)

        # print("#" * 20, "Actor Parameters")
        # for param in actor.parameters():
        #     print(param.grad)
        #
        # print("#" * 20, "Critic Parameters")
        # for param in critic.parameters():
        #     print(param.grad)

        # update the actor and critic parameters
        adam_actor.step()
        adam_critic.step()

        print("\r", f"Episode: {episode}, Avg Return per Epoch: {np.mean(total_rewards):.3f}", end="", flush=True)

        writer.add_scalar(tag='Average Return over 100 episodes',
                          scalar_value=np.mean(total_rewards),
                          global_step=episode)

        writer.add_scalar(tag='Mean Entropy',
                          scalar_value=mean_entropy.item(),
                          global_step=episode)

        # check if solved
        if np.mean(total_rewards) > 200:
            print('\nSolved!')
            break

    # close the environment
    env.close()

    # close the writer
    writer.close()


if __name__ == "__main__":
    main()
