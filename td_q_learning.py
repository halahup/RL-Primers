import gym
import numpy as np


# hyper-parameters for the Q-learning algorithm
NUM_EPISODES = 1000000  # number of episodes to run
GAMMA = 0.9             # discount factor
ALPHA = 0.3             # learning rate


def main():

    # create the environment
    env = gym.make('Taxi-v2')

    # initialize the q_learning table
    q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float32)

    # define total reward
    total_reward = 0
    total_steps = 0

    # define the epsilon for the exploration-exploitation balance
    epsilon = 1.0

    for episode in range(1, NUM_EPISODES):

        # reset the environment to a random state
        current_state = env.reset()

        # default value for the episode termination value
        done = False

        # define the number of steps per episode
        steps = 1

        while not done:

            # Explore with probability epsilon
            if np.random.uniform(low=0, high=1, size=1) < epsilon:

                action = env.action_space.sample()

            # Else act greedily
            else:

                # take the action that yields the largest Q-value for the state-action pait
                action = np.argmax(q_table[current_state], axis=0)

            # perform the action
            new_state, reward, done, _ = env.step(action=action)

            # now we are in a new state, and we got a reward for proceeding into that state
            # potentially we can be done, if we reach a terminal state
            # we need to update the Q-table now with the new observed value of reward
            q_table[current_state, action] = (1 - ALPHA) * q_table[current_state, action] +\
                ALPHA * (reward + GAMMA * np.max(q_table[new_state]))

            # increment the total reward
            total_reward += reward

            # proceed to the next state
            current_state = new_state

            # increment the steps per episode counter
            steps += 1
            total_steps += 1

        # decay the exploration factor
        if epsilon > 0.1:
            epsilon -= 1 / (0.3 * NUM_EPISODES)

        print("\r", "Epsilon: {:.3f}, Episode: {}, Avg Steps Taken per Episode: {:.3f}, Avg Reward per Episode: {:.3f}"
              .format(epsilon, episode, total_steps / float(episode), total_reward / float(episode)),
              end="", flush=True)

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
