import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# hyper-parameters for the Q-learning algorithm
NUM_EPISODES = 100000   # number of episodes to run
GAMMA = 0.9             # discount factor
ALPHA = 0.01            # learning rate


# the Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.fc_in = nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=64)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)

    def forward(self, x):
        x = self.lrelu(self.gn(self.fc_in(x)))
        x = self.fc_out(x)
        return x


def scalar_to_one_hot(scalar: int, size: int):
    """
        Turns a scalar into torch one-hot tensor
    :param scalar: a scalar to turn into one-hot
    :param size: dimensionality of the one-hot vector
    :return: the one-hot torch tensor
    """
    one_hot = torch.zeros(size=(1, size), dtype=torch.float)
    one_hot[0][scalar] = 1
    return one_hot


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter()

    # create the environment
    env = gym.make('Taxi-v2')

    # Q-table is replaced by the agent driven by a neural network architecture
    agent = Agent(observation_space_size=env.observation_space.n,
                  action_space_size=env.action_space.n,
                  hidden_size=64)

    criterion = nn.MSELoss(reduction='sum')

    adam = optim.Adam(params=agent.parameters(), lr=ALPHA)

    # define total reward and total number of steps for feedback
    total_reward = 0
    total_steps = 0

    # define the epsilon for the exploration-exploitation balance
    # we start out with the fully exploratory behavior
    epsilon = 1.0
    min_epsilon = 0.1

    # capture the steps and the average rewards
    plt_avg_steps, plt_avg_rewards = list(), list()

    for episode in range(1, NUM_EPISODES):

        # reset the environment to a random initial state
        current_state = env.reset()

        # default value for the episode termination value
        done = False

        # define the counter for the number of steps per episode
        steps = 1

        while not done:

            # get the action values
            action_values = agent(scalar_to_one_hot(scalar=current_state, size=env.observation_space.n))

            # define the target values for the actions by detaching and deep copying the action vector
            action_values_target = action_values.detach().clone()

            # explore with probability epsilon
            if np.random.uniform(low=0, high=1, size=1) < epsilon:
                action = env.action_space.sample()

            # act greedily with probability (1 - epsilon)
            else:
                # chose the action that results in the largest action value
                action = torch.argmax(action_values, dim=1).item()

            # print(action)

            # perform the action
            new_state, reward, done, _ = env.step(action=action)

            # pass the new state through the agent to get the action values of the next state
            new_state_action_values = agent(scalar_to_one_hot(scalar=new_state, size=env.observation_space.n))

            # define the target for the taken action
            if done:
                action_value_target = reward
            else:
                action_value_target = reward + GAMMA * torch.max(new_state_action_values).detach().item()

            # print(action_value_target)
            # print(action_values_target)

            # define the target vector for the agent
            action_values_target[0][action] = action_value_target

            # print(action_values_target)

            # calculate the loss
            loss = criterion(action_values, action_values_target)

            # zero the gradient
            agent.zero_grad()

            # backprop
            loss.backward()

            # update the parameters
            adam.step()

            # increment the total reward
            total_reward += reward

            # proceed to the next state
            current_state = new_state

            # increment the steps per episode counter
            steps += 1
            total_steps += 1

        # decay the exploration factor
        if epsilon > min_epsilon:
            epsilon -= 1 / (0.25 * NUM_EPISODES)

        # append the average steps and average rewards
        avg_steps = total_steps / float(episode)
        plt_avg_steps.append(avg_steps)

        avg_reward = total_reward / float(episode)
        plt_avg_rewards.append(avg_reward)

        writer.add_scalar(tag='Average Reward', scalar_value=avg_reward, global_step=episode)
        writer.add_scalar(tag='Average Steps', scalar_value=avg_steps, global_step=episode)
        writer.add_scalar(tag='Neural Network Loss', scalar_value=loss.item() * 10e5, global_step=episode)
        writer.add_histogram(tag='FC_IN Weights', values=agent.fc_in.weight.data.detach().numpy(),
                             global_step=episode)
        writer.add_histogram(tag='FC_OUT Weights', values=agent.fc_out.weight.data.detach().numpy(),
                             global_step=episode)
        writer.add_histogram(tag='FC_IN Grads', values=agent.fc_in.weight.grad.numpy(),
                             global_step=episode)
        writer.add_histogram(tag='FC_OUT Grads', values=agent.fc_out.weight.grad.numpy(),
                             global_step=episode)

        print("\r", "Epsilon: {:.3f}, Episode: {}, "
                    "Avg Steps Taken per Episode: {:.3f}, "
                    "Avg Reward per Episode: {:.3f}"
              .format(epsilon, episode, avg_steps, avg_reward), end="", flush=True)

    # close the environment
    env.close()

    # close the writer
    writer.close()

    # save the plots
    plt.plot(plt_avg_steps)
    plt.savefig('./average_steps.png')
    plt.clf()

    plt.plot(plt_avg_rewards)
    plt.savefig('./average_rewards.png')
    plt.clf()


if __name__ == "__main__":
    main()
