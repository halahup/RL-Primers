# TODO: implement the A2C and A3C models for the lunar lander
# TODO: add gradient clipping
# TODO: investigate why scale of the advantage is so large
# TODO: implement writing the gradients and weights to TB
# TODO: try separating the actor and the critic
# TODO: add the gradient and weight TB logging

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


ALPHA = 0.01              # learning rate
BATCH_SIZE = 50           # how many episodes we want to pack into an epoch
GAMMA = 0.99              # discount rate
HIDDEN_SIZE_ACTOR = 128   # number of hidden nodes we have in the actor network
HIDDEN_SIZE_CRITIC = 128  # number of hidden nodes we have in the critic network
BETA = 0.02               # multiplier for the entropy bonus
STEPS = 100               # number of steps we want to calculate the discounted rewards forward for


# the Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class Critic(nn.Module):
    def __init__(self, observation_space_size: int, hidden_size: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=True)
        )

    def forward(self, x):
        logit = self.net(x)
        return logit


def calculate_loss(actions: torch.Tensor, weights: torch.Tensor, logits: torch.Tensor):

    # create the one hot mask
    masks = one_hot(actions, num_classes=logits.shape[1])

    # calculate the log-probabilities of the corresponding chosen action
    # and sum them up across the first dimension to for a vector
    log_probs = torch.sum(masks.float() * log_softmax(logits, dim=1), dim=1)
    loss = -1 * torch.mean(log_probs * weights.squeeze())

    # add the entropy bonus
    p = softmax(logits, dim=1)
    log_p = log_softmax(logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    entropy_bonus = -1 * BETA * entropy

    return loss, entropy_bonus, entropy


def get_discounted_rewards(rewards, steps: int = None):
    discounted_rewards = np.empty_like(rewards, dtype=np.float)
    if not steps:
        for i in range(rewards.shape[0]):
            gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=GAMMA)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
            discounted_reward = np.sum(rewards[i:] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
    else:
        for i in range(rewards.shape[0]):
            j = i + steps if i + steps < rewards.shape[0] else rewards.shape[0]
            gammas = np.full(shape=(rewards[i:j].shape[0]), fill_value=0.99)
            discounted_gammas = np.power(gammas, np.arange(rewards[i:j].shape[0]))
            discounted_reward = np.sum(rewards[i:j] * discounted_gammas)
            discounted_rewards[i] = discounted_reward
    return discounted_rewards


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter(comment=f'_Gamma={GAMMA},'
                           f'LR={ALPHA},BS={BATCH_SIZE},NHA={HIDDEN_SIZE_ACTOR},NHC={HIDDEN_SIZE_CRITIC}')

    # create the environment
    env = gym.make('LunarLander-v2')

    # actor approximates the policy -> number of actions logits are the output
    actor = Agent(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE_ACTOR)

    # critic approximates the value of the state -> 1 logit is the output -> regression task
    critic = Critic(observation_space_size=env.observation_space.shape[0],
                    hidden_size=HIDDEN_SIZE_CRITIC)

    # define the optimizer
    adam = optim.Adam(params=list(actor.parameters()) + list(critic.parameters()), lr=ALPHA, eps=1e-8)

    # deque to calculate the average rewards over 100 steps
    rewards_over_100_episodes = deque([], maxlen=100)

    # init the epoch and global episode counters
    epoch = 1
    episode_global = 1

    # every epoch
    while True:

        # reset the environment to a random initial state every epoch
        state = env.reset()

        # initialize the structures for the batch collection
        # We collect the observations and actions we performed over the epoch to form the trajectory.
        # We also collect the rewards every step in the episode.
        # We collect the logits for computation of the loss

        # actions we took over the epoch
        epoch_actions = torch.empty(size=(0, ), dtype=torch.long)

        # the advantage weights for A2C
        epoch_reward_weights = torch.empty(size=(0, ), dtype=torch.float)

        # initialize the arrays to hold the state value logits and the rewards to go over an epoch
        epoch_state_value_logits = torch.empty(size=(0, ), dtype=torch.float)
        epoch_discounted_rewards_to_go = torch.empty(size=(0,), dtype=torch.float)

        # logits of actions we have taken over an epoch
        epoch_action_logits = torch.empty(size=(0, env.action_space.n))

        # array to store the values of the states over an episode
        episode_state_value_logits = torch.empty(size=(0, 1), dtype=torch.float)

        # rewards over an episode
        episode_rewards = np.empty(shape=(0, ))

        # flag for environment rendering
        finished_rendering_this_epoch = False

        # counter for the episode within the epoch
        episode = 0

        # episode loop
        while True:

            # There are multiple episodes in a single epoch
            # During these episodes we accumulate the training data
            # We train after we collect enough training data to approximate the policy (performance) gradient
            # render the environment for the first episode in the epoch
            if not finished_rendering_this_epoch:
                env.render()

            # get the action logits from the neural network
            # save the logits to the pool for further loss calculation
            action_logits = actor(torch.tensor(state).float().unsqueeze(dim=0))

            # get the logit out of the critic network for baseline calculation
            value_logit = critic(torch.tensor(state).float().unsqueeze(dim=0))
            
            # collect the action logits and state value approximation logit for every state in the episode
            epoch_action_logits = torch.cat((epoch_action_logits, action_logits), dim=0)
            episode_state_value_logits = torch.cat((episode_state_value_logits, value_logit), dim=0)

            # sample an action according to the action distribution of the actor network
            action = Categorical(logits=action_logits).sample()

            # append the action to the batch of actions for trajectory
            epoch_actions = torch.cat((epoch_actions, action), dim=0)

            # perform the action
            state, reward, done, _ = env.step(action=action.item())

            # append the reward to the rewards pool that we collect during the episode
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])))

            # if the episode is over
            if done:

                # increment the episode counters
                episode_global += 1
                episode += 1

                # Calculate the episode's total reward and append it to the batch of returns.
                # Here we turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more than the later taken actions.
                # We only get the discounted rewards over the next K steps to avoid large
                # values in the advantage functions.
                discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards, steps=STEPS)
                discounted_rewards_to_go = torch.tensor(discounted_rewards_to_go, dtype=torch.float)
                epoch_discounted_rewards_to_go = torch.cat((epoch_discounted_rewards_to_go, discounted_rewards_to_go),
                                                           dim=0)

                # baseline -> Advantage function -> Q = V + A, so Q - V = A
                # Discounted rewards-to-go are the rewards we actually have observed (Q(s,a)-values).
                # While the output of the critic network is the approximation to the values of the state (V(s)-values)
                # Approximated advantage is A'(s,a) = Q(s,a) - V(s)
                advantage = discounted_rewards_to_go - episode_state_value_logits.detach().squeeze()
                # We would like to detach the state value approximation from the computational graph because
                # the approximated advantage is used in the computation of the policy gradient, and
                # we don't want the policy gradient to propagate into the critic network

                # get the total episode reward and append it to the running 100 rewards
                total_episode_reward = np.sum(episode_rewards)
                rewards_over_100_episodes.append(total_episode_reward)

                # write the total episode reward to TB per global episode step
                writer.add_scalar(tag='Total Episode Reward',
                                  scalar_value=np.sum(episode_rewards),
                                  global_step=episode_global)

                # The weights (\Phi_t) for the policy gradient are the advantages that we approximated
                # with the help of our critic network
                epoch_reward_weights = torch.cat((epoch_reward_weights, advantage.detach()), dim=0)
                epoch_state_value_logits = torch.cat((epoch_state_value_logits, episode_state_value_logits), dim=0)

                # reset the episode
                # since there are potentially more episodes left in the epoch to run
                state = env.reset()

                # episode rewards
                episode_rewards = np.empty(shape=(0, ))
                episode_state_value_logits = torch.empty(size=(0, 1), dtype=torch.float)

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # if the epoch is over
                # end the loop if we have enough observations
                if episode >= BATCH_SIZE:
                    break

        # increment the epoch counter
        epoch += 1

        # calculate the policy network (actor network) loss
        policy_loss, entropy_bonus, entropy = calculate_loss(actions=epoch_actions,
                                                             weights=epoch_reward_weights,
                                                             logits=epoch_action_logits)

        # calculate the value network (critic network) loss
        # Approximated V(s) vs Q(s,a) just like in DQN
        value_loss = F.mse_loss(input=epoch_state_value_logits.squeeze(), target=epoch_discounted_rewards_to_go)

        # if the agent is extremely confident in his action -> we want to penalize him to encourage exploration
        # if the agent is not confident the total loss is lower as the entropy loss is higher
        total_loss = policy_loss + value_loss + entropy_bonus

        # zero the gradients in both networks
        actor.zero_grad()
        critic.zero_grad()

        # backprop
        total_loss.backward()

        # clip the gradients of the parameters
        clip_grad_value_(actor.parameters(), clip_value=0.1)
        clip_grad_value_(critic.parameters(), clip_value=0.1)

        # writer.add_histogram(tag='Actor Gradients',
        #                      values=actor.parameters().grad,
        #                      global_step=epoch)
        #
        # writer.add_histogram(tag='Critic Gradients',
        #                      values=critic.parameters().grad,
        #                      global_step=epoch)

        # update the parameters
        adam.step()

        # feedback
        print("\r", "Epoch: {}, "
                    "Avg Return over 100 epsisodes: {:.3f}"
              .format(epoch, np.mean(rewards_over_100_episodes)), end="", flush=True)

        writer.add_scalar(tag='Average Return over 100 episodes',
                          scalar_value=np.mean(rewards_over_100_episodes),
                          global_step=epoch)
        writer.add_scalar(tag='Entropy', scalar_value=entropy.item(), global_step=epoch)
        writer.add_scalar(tag='Advantage', scalar_value=advantage.mean().item(), global_step=epoch)

        # check if solved
        if np.mean(rewards_over_100_episodes) > 200:
            print('\nSolved!')
            break

    total_episode_rewards = list()

    # run an extra 100 episode with the agent to check out the performance
    for _ in range(100):

        # reset the environment to a random initial state every epoch
        state = env.reset()

        # empty list for the episode rewards
        episode_rewards = list()

        # episode loop
        while True:

            # get action logits from the agent based on the state
            action_logits = actor(torch.tensor(state).float().unsqueeze(dim=0))

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # perform the action
            state, reward, done, _ = env.step(action=action.item())

            episode_rewards.append(reward)

            # if the episode is over
            if done:

                # calculate the total episode rewards
                total_episode_rewards.append(np.sum(episode_rewards))
                break

    plt.rcParams['figure.figsize'] = (20, 10)
    plt.plot(total_episode_rewards)
    plt.title('Rewards over 100 episodes of the trained agent')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.grid(linestyle='--')
    plt.savefig('./figures/100_trained.png')

    # close the environment
    env.close()

    # close the writer
    writer.close()


if __name__ == "__main__":
    main()
