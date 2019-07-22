import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque


ALPHA = 0.005             # learning rate
BATCH_SIZE = 50           # how many episodes we want to pack into an epoch
GAMMA = 0.99              # discount rate
HIDDEN_SIZE = 64         # number of hidden nodes we have in our approximation
BETA = 1.0

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')


# Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

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


def calculate_loss(epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor) -> (torch.Tensor, torch.Tensor):

    policy_loss = -1 * torch.mean(weighted_log_probs)

    # add the entropy bonus
    p = softmax(epoch_logits, dim=1)
    log_p = log_softmax(epoch_logits, dim=1)
    entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
    entropy_bonus = -1 * BETA * entropy

    return policy_loss + entropy_bonus, entropy


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
    writer = SummaryWriter(comment=f'_Gamma={GAMMA},LR={ALPHA},BS={BATCH_SIZE},NH={HIDDEN_SIZE},BETA={BETA}')

    # create the environment
    env = gym.make('LunarLander-v2')

    # Q-table is replaced by the agent driven by a neural network architecture
    agent = Agent(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=HIDDEN_SIZE)
    agent = agent.to(DEVICE)
    adam = optim.Adam(params=agent.parameters(), lr=ALPHA)

    total_rewards = deque([], maxlen=100)

    epoch = 1
    episode_total = 1

    # epoch
    while True:

        # reset the environment to a random initial state every epoch
        state = env.reset()

        episode_actions = torch.empty(size=(0, ), dtype=torch.long, device=DEVICE)
        episode_logits = torch.empty(size=(0, env.action_space.n), device=DEVICE)
        epoch_logits = torch.empty(size=(0, env.action_space.n), device=DEVICE)  # used for entropy calculation
        epoch_weighted_log_probs = torch.empty(size=(0, ), dtype=torch.float, device=DEVICE)

        average_rewards = np.empty(shape=(0, ), dtype=np.float)
        episode_rewards = np.empty(shape=(0, ), dtype=np.float)

        finished_rendering_this_epoch = False

        episode = 0

        # episode
        while True:

            # render the environment for the first episode in the epoch
            if not finished_rendering_this_epoch:
                env.render()

            # get the action logits from the neural network
            action_logits = agent(torch.tensor(state).float().unsqueeze(dim=0).to(DEVICE))

            # append the logits to the episode logits list
            episode_logits = torch.cat((episode_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action = Categorical(logits=action_logits).sample()

            # append the action to the episode action list to obtain the trajectory
            episode_actions = torch.cat((episode_actions, action), dim=0)

            # perform the action
            state, reward, done, _ = env.step(action=action.cpu().item())

            # append the reward to the rewards pool that we collect during the episode
            # we use the episode rewards for the discounted rewards to go calculation
            # and the baseline of average
            episode_rewards = np.concatenate((episode_rewards, np.array([reward])), axis=0)
            average_rewards = np.concatenate((average_rewards,
                                              np.expand_dims(np.mean(episode_rewards), axis=0)),
                                             axis=0)

            # the episode is over
            if done:

                # increment the episode
                episode_total += 1
                episode += 1

                # here we turn the rewards we accumulated during the episode into the rewards-to-go:
                # earlier actions are responsible for more than the later taken actions
                discounted_rewards_to_go = get_discounted_rewards(rewards=episode_rewards)
                discounted_rewards_to_go -= average_rewards  # baseline
                total_rewards.append(np.sum(episode_rewards))

                # set the mask for the actions taken in the episode
                mask = one_hot(episode_actions, num_classes=env.action_space.n)
                episode_log_probs = torch.sum(mask.float() * log_softmax(episode_logits, dim=1), dim=1)
                episode_weighted_log_probs = episode_log_probs * \
                    torch.tensor(discounted_rewards_to_go).float().to(DEVICE)
                epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs,
                                                      torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)),
                                                     dim=0)

                epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)

                # reset the episode
                # since there are potentially more episodes left in the epoch to run
                state = env.reset()
                episode_rewards = np.empty(shape=(0, ), dtype=np.float)
                average_rewards = np.empty(shape=(0,), dtype=np.float)
                episode_actions = torch.empty(size=(0,), dtype=torch.long, device=DEVICE)
                episode_logits = torch.empty(size=(0, env.action_space.n), device=DEVICE)

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # if the epoch is over
                # end the loop if we have enough observations
                if episode >= BATCH_SIZE:
                    break

        epoch += 1

        # calculate the loss
        loss, entropy = calculate_loss(epoch_logits=epoch_logits, weighted_log_probs=epoch_weighted_log_probs)

        # zero the gradient
        adam.zero_grad()

        # backprop
        loss.backward()

        # update the parameters
        adam.step()

        # feedback
        print("\r", "Epoch: {}, "
                    "Avg Return per Epoch: {:.3f}"
              .format(epoch, np.mean(total_rewards)), end="", flush=True)

        writer.add_scalar(tag='Average Return over 100 episodes',
                          scalar_value=np.mean(total_rewards),
                          global_step=epoch)

        writer.add_scalar(tag='Entropy',
                          scalar_value=entropy,
                          global_step=epoch)

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
