import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax
from scipy.stats import entropy


# hyper-parameters for the Q-learning algorithm
NUM_EPISODES = 100000  # number of episodes to run
GAMMA = 0.9             # discount factor
ALPHA = 0.5             # learning rate


def main():

    tb_writer = SummaryWriter()

    # create the environment
    env = gym.make('Taxi-v2')

    # initialize the q_learning table
    # In simple temporal difference methods the action value function can be represented by a table;
    # hence, sometimes they call these methods tabular or tabular TD
    # We initialize a table with zeros. The table has height of number of states in the state space
    # and width of number of possible actions in the action space.
    # We can lookup a value of a particular action in a specific state in the Q-table
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float32)

    # define total reward and total number of steps for feedback
    total_reward = 0
    total_steps = 0

    for episode in range(1, NUM_EPISODES):

        # reset the environment to a random initial state
        current_state = env.reset()

        # default value for the episode termination value
        done = False

        # define the counter for the number of steps per episode
        steps = 1

        episode_entropy = list()

        while not done:

            preferences = q_table[current_state]
            probs = softmax(preferences)
            episode_entropy.append(entropy(probs))

            action = np.random.choice(a=np.arange(preferences.shape[0]), p=probs)

            # perform the action
            new_state, reward, done, _ = env.step(action=action)

            # Now we are in a new state, and we got a reward for proceeding into that state.
            # Potentially we can be done, if we reach a terminal state.
            # We need to update the Q-table now with the new observed value of reward:
            # we use the q-learning update rule to perform the update.
            # We blend the current Q-value for the state-action pair with the target of the newly obtained
            # reward for the new state and discounted maximum action value for the new state.
            # This is called bootstrapping - we use the next state's maximum action value to define the current
            # state's action value.
            # q_table[current_state, action] = (1 - ALPHA) * q_table[current_state, action] +\
            #     ALPHA * (reward + GAMMA * np.max(q_table[new_state]))

            q_table[current_state, action] = (1 - ALPHA) * q_table[current_state, action] + \
                ALPHA * (reward + GAMMA * np.max(q_table[new_state]))

            # increment the total reward
            total_reward += reward

            # proceed to the next state
            current_state = new_state

            # increment the steps per episode counter
            steps += 1
            total_steps += 1

        print("\r", "Episode: {}, Avg Steps Taken per Episode: {:.3f}, Avg Reward per Episode: {:.3f}"
              .format(episode, total_steps / float(episode), total_reward / float(episode)),
              end="", flush=True)

        tb_writer.add_scalar(tag='Average Reward per Episode',
                             scalar_value=total_reward / float(episode),
                             global_step=episode)

        tb_writer.add_scalar(tag='Average Steps Taken Per Episode',
                             scalar_value=total_steps / float(episode),
                             global_step=episode)

        tb_writer.add_scalar(tag='Average Entropy per Episode',
                             scalar_value=np.mean(episode_entropy),
                             global_step=episode)

    # close the environment
    env.close()
    tb_writer.close()


if __name__ == "__main__":
    main()
