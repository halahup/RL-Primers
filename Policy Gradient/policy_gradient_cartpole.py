import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.nn.functional import one_hot, log_softmax
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# hyper-parameters for the Q-learning algorithm
NUM_EPOCHS = 1000        # number of episodes to run
ALPHA = 0.01             # learning rate


# the Q-table is replaced by a neural network
class Agent(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, hidden_size: int):
        super(Agent, self).__init__()

        self.fc_in = nn.Linear(in_features=observation_space_size, out_features=hidden_size, bias=True)
        self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=action_space_size, bias=True)

    def forward(self, x):
        x = self.tanh(self.fc_in(x))
        x = self.fc_out(x)
        return x


# TODO: Implement the policy gradient

def calculate_loss(actions: torch.Tensor, weights: torch.Tensor, logits: torch.Tensor):

    # create the one hot mask
    masks = one_hot(actions, num_classes=2)

    # calculate the log-probabilities of the corresponding chosen action
    # and sum them up across
    log_probs = torch.sum(masks.float() * log_softmax(logits, dim=0), dim=0)


    return -1 * torch.mean(weights * log_probs)


def main():

    # instantiate the tensorboard writer
    writer = SummaryWriter()

    # create the environment
    env = gym.make('CartPole-v1')

    # Q-table is replaced by the agent driven by a neural network architecture
    agent = Agent(observation_space_size=env.observation_space.shape[0],
                  action_space_size=env.action_space.n,
                  hidden_size=32)

    adam = optim.Adam(params=agent.parameters(), lr=ALPHA)

    # define total reward and total number of steps for feedback
    total_steps = 0
    batch_size = 1024

    for epoch in range(1, NUM_EPOCHS):

        # reset the environment to a random initial state
        current_state = env.reset()

        # initialize the lists for the batch collection
        batch_observations = torch.empty(size=(0, env.observation_space.shape[0]), dtype=torch.float)
        batch_actions = torch.empty(size=(0, ), dtype=torch.long)

        batch_returns = list()
        batch_logits = torch.empty(size=(0, env.action_space.n))
        episode_rewards = list()

        finished_rendering_this_epoch = False

        while True:

            # render the environment
            if not finished_rendering_this_epoch:
                env.render()

            # append the observation to the batch of the observations
            batch_observations = torch.cat((batch_observations, torch.Tensor(current_state).unsqueeze(dim=0)), dim=0)

            # get the action logits from the neural network
            action_logits = agent(torch.Tensor(current_state).unsqueeze(dim=0))
            batch_logits = torch.cat((batch_logits, action_logits), dim=0)

            # sample an action according to the action distribution
            action_distribution = Categorical(logits=action_logits)
            action = action_distribution.sample()

            # append the action to the batch of actions
            batch_actions = torch.cat((batch_actions, action), dim=0)

            # perform the action
            new_state, reward, done, _ = env.step(action=action.item())

            # append the reward to the rewards that we collect during the full episode
            episode_rewards.append(reward)

            # increment the steps per episode counter
            total_steps += 1

            # if the episode is done
            if done:

                # calculate the episode total reward and append it to the batch of returns
                epoch_return = np.sum(episode_rewards)
                batch_returns.append(epoch_return)

                # calculate the episode length and append it to the batch of lengths
                epoch_length = len(episode_rewards)

                # the weights are the epoch return multiplied across the epoch length
                batch_weights = torch.zeros(size=(epoch_length, 1)).fill_(epoch_return)

                # reset the episode
                current_state = env.reset()
                done = False
                episode_rewards = list()

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end the loop if we have enough of the observations
                if batch_observations.shape[0] > batch_size:
                    break

            # update the step
            current_state = new_state

        loss = calculate_loss(actions=batch_actions, weights=batch_weights, logits=batch_logits)

        # raise NotImplementedError

        # zero the gradient
        agent.zero_grad()

        # backprop
        loss.backward()

        # clip the gradient values
        clip_grad_value_(parameters=agent.parameters(), clip_value=1)

        # update the parameters
        adam.step()

        # append the average steps and average rewards
        avg_steps = total_steps / batch_size

        writer.add_scalar(tag='Average Steps', scalar_value=avg_steps, global_step=epoch)
        writer.add_scalar(tag='Average Reward', scalar_value=epoch_return, global_step=epoch)
        writer.add_scalar(tag='Neural Network Loss', scalar_value=loss.item(), global_step=epoch)
        writer.add_histogram(tag='FC_IN Weights', values=agent.fc_in.weight.data.detach().numpy(), global_step=epoch)
        writer.add_histogram(tag='FC_OUT Weights', values=agent.fc_out.weight.data.detach().numpy(), global_step=epoch)
        writer.add_histogram(tag='FC_IN Grads', values=agent.fc_in.weight.grad.numpy(), global_step=epoch)
        writer.add_histogram(tag='FC_OUT Grads', values=agent.fc_out.weight.grad.numpy(), global_step=epoch)

        print("\r", "Epoch: {}, "
                    "Avg Steps Taken per Epoch: {:.3f}, "
                    "Reward per Epoch: {:.3f}"
              .format(epoch, avg_steps, epoch_return), end="", flush=True)

    # close the environment
    env.close()

    # close the writer
    writer.close()


if __name__ == "__main__":
    main()
