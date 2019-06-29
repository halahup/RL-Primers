import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque


# hyper-parameters for the Q-learning algorithm
NUM_EPOCHS = 400          # number of episodes to run
ALPHA = 0.005             # learning rate
BATCH_SIZE = 50           # how many episodes we want to pack into an epoch
GAMMA = 0.99              # discount rate
HIDDEN_SIZE = 512         # number of hidden nodes we have in our approximation


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
        x = self.net(x)
        return x

# TODO: Write extensive comments, describing what happens and why
# TODO: Add the entropy penalty to improve the exploration
# TODO: Figure out the loss function


def calculate_loss(actions: torch.Tensor, weights: torch.Tensor, logits: torch.Tensor):

    # create the one hot mask
    masks = one_hot(actions, num_classes=4)

    # calculate the log-probabilities of the corresponding chosen action
    # and sum them up across the first dimension to for a vector
    log_probs = torch.sum(masks.float() * log_softmax(logits, dim=1), dim=1)
    loss = -1 * torch.mean(weights.squeeze() * log_probs)

    return loss


def get_discounted_rewards(rewards):
    discounted_rewards = np.empty_like(rewards, dtype=np.float)
    for i in range(rewards.shape[0]):
        gammas = np.full(shape=(rewards[i:].shape[0]), fill_value=GAMMA)
        discounted_gammas = np.power(gammas, np.arange(rewards[i:].shape[0]))
        discounted_reward = np.sum(rewards[i:] * discounted_gammas)
        discounted_rewards[i] = discounted_reward
    return discounted_rewards


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter(comment=f'_Gamma={GAMMA},LR={ALPHA},BS={BATCH_SIZE},NH={HIDDEN_SIZE}')

    # create the environment
    env = gym.make('LunarLander-v2')

    # Q-table is replaced by the agent driven by a neural network architecture
    agent = Agent(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE)

    adam = optim.Adam(params=agent.parameters(), lr=ALPHA)

    total_rewards = deque([], maxlen=100)
    epoch = 1
    episode_total = 1

    # epoch loop
    while True:

        # reset the environment to a random initial state every epoch
        state = env.reset()

        # initialize the structures for the batch collection
        # We collect the observations and actions we performed over the epoch to form the trajectory.
        # We also collect the rewards every step in the episode.
        # We collect the logits for computation of the loss
        epoch_observations = torch.empty(size=(0, env.observation_space.shape[0]), dtype=torch.float)
        epoch_actions = torch.empty(size=(0, ), dtype=torch.long)
        epoch_weights = torch.empty(size=(0, ), dtype=torch.float)
        epoch_logits = torch.empty(size=(0, env.action_space.n))
        epoch_returns = np.empty(shape=(0, ))
        episode_rewards = np.empty(shape=(0, ))

        finished_rendering_this_epoch = False

        episode = 0

        # episode loop
        while True:

            # There are many episodes in a single epoch
            # During these episodes we accumulate the training data
            # We train after we collect enough training data

            # render the environment for the first episode in the epoch
            if not finished_rendering_this_epoch:
                env.render()

            # append the observation to the global pool of observations that we are collecting over the epoch
            epoch_observations = torch.cat((epoch_observations, torch.Tensor(state).unsqueeze(dim=0)), dim=0)

            # get the action logits from the neural network
            # save the logits to the pool for further loss calculation
            action_logits = agent(torch.Tensor(state).unsqueeze(dim=0))
            epoch_logits = torch.cat((epoch_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the batch of actions for trajectory
            epoch_actions = torch.cat((epoch_actions, action), dim=0)

            # perform the action
            state, reward, done, _ = env.step(action=action.item())

            # append the reward to the rewards pool that we collect during the episode
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])))

            # if the episode is over
            if done:

                # increment the episode
                episode_total += 1
                episode += 1

                # calculate the episode's total reward and append it to the batch of returns
                # this is the epoch's return - sum of the episodes' rewards
                # here we turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more than the later taken actions
                discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards)
                discounted_rewards_to_go -= np.mean(discounted_rewards_to_go)
                total_rewards.append(np.sum(episode_rewards))

                # write the total episode reward to TB
                writer.add_scalar(tag='Total Episode Reward',
                                  scalar_value=np.sum(episode_rewards),
                                  global_step=episode_total)

                epoch_returns = np.concatenate((epoch_returns, np.array([np.sum(episode_rewards)])))

                # calculate the episode's length
                episode_length = episode_rewards.shape[0]

                # the weights are the rewards-to-go per episode
                # accumulate the weights across the epoch
                epoch_weights = torch.cat((epoch_weights,
                                           torch.ones(size=(episode_length, )) *
                                           torch.tensor(discounted_rewards_to_go.copy()).float()))

                # reset the episode
                # since there are potentially more episodes left in the epoch to run
                state = env.reset()
                episode_rewards = np.empty(shape=(0, ))

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # if the epoch is over
                # end the loop if we have enough observations
                if episode >= BATCH_SIZE:
                    break

        epoch += 1

        # check if solved
        if np.mean(total_rewards) > 200:
            print('\nSolved!')
            break

        # calculate the loss
        loss = calculate_loss(actions=epoch_actions, weights=epoch_weights, logits=epoch_logits)

        # zero the gradient
        adam.zero_grad()

        # backprop
        loss.backward()

        # update the parameters
        adam.step()

        # feedback
        print("\r", "Epoch: {}, "
                    "Avg Return per Epoch: {:.3f}"
              .format(epoch, np.mean(epoch_returns)), end="", flush=True)

        # write to tensorboard
        writer.add_scalar(tag='Average Epoch Return',
                          scalar_value=np.mean(epoch_returns),
                          global_step=epoch)
        writer.add_scalar(tag='Average Return over 100 episodes',
                          scalar_value=np.mean(total_rewards),
                          global_step=epoch)

    # close the environment
    env.close()

    # close the writer
    writer.close()


if __name__ == "__main__":
    main()
