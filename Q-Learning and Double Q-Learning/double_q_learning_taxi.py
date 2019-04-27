import gym
import numpy as np


# hyper-parameters for the Q-learning algorithm
NUM_EPISODES = 1000000   # number of episodes to run
GAMMA = 0.9              # discount factor
ALPHA = 0.5              # learning rate


def main():

    # create the environment
    env = gym.make('Taxi-v2')

    # initialize the q_learning table
    q_table_1 = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float32)
    q_table_2 = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float32)

    # define total reward and total number of steps for feedback
    total_reward = 0
    total_steps = 0

    # define the epsilon for the exploration-exploitation balance
    # we start out with the fully exploratory behavior
    epsilon = 1.0

    for episode in range(1, NUM_EPISODES):

        # reset the environment to a random initial state
        current_state = env.reset()

        # default value for the episode termination value
        done = False

        # define the counter for the number of steps per episode
        steps = 1

        while not done:

            # explore with probability epsilon
            if np.random.uniform(low=0, high=1, size=1) < epsilon:

                action = env.action_space.sample()

            # act greedily with probability (1 - epsilon)
            else:

                # Take the action that yields the largest Q-value for the state-action pair.
                # This time we use 2 functions to figure out what action to take.
                action = np.argmax(q_table_1[current_state] +
                                   q_table_2[current_state], axis=0)

            # perform the action
            new_state, reward, done, _ = env.step(action=action)

            # Now we are in a new state, and we got a reward for proceeding into that state.
            # We use 2 functions to avoid the maximum bias. Now, we look for the action that results in
            # the largest Q-value according to the function we are updating.
            # Next, however, we use the Q-Value from the second function to estimate the Q-value.
            # This system avoids being over-positive and taking np.max(q_table[new_state]).
            # Half of the times we update the first function and half of the times the other one.
            if np.random.uniform(low=0, high=1, size=1) < 0.5:
                q_table_1[current_state, action] = (1 - ALPHA) * q_table_1[current_state, action] +\
                    ALPHA * (reward + GAMMA * q_table_2[new_state, np.argmax(q_table_1[new_state], axis=0)])

            else:
                q_table_2[current_state, action] = (1 - ALPHA) * q_table_2[current_state, action] + \
                    ALPHA * (reward + GAMMA * q_table_1[new_state, np.argmax(q_table_2[new_state], axis=0)])

            # increment the total reward
            total_reward += reward

            # proceed to the next state
            current_state = new_state

            # increment the steps per episode counter
            steps += 1
            total_steps += 1

        # decay the exploration factor
        if epsilon > 0.01:
            epsilon -= 1 / (0.01 * NUM_EPISODES)

        print("\r", "Epsilon: {:.3f}, Episode: {}, Avg Steps Taken per Episode: {:.3f}, Avg Reward per Episode: {:.3f}"
              .format(epsilon, episode, total_steps / float(episode), total_reward / float(episode)),
              end="", flush=True)

    # close the environment
    env.close()


if __name__ == "__main__":
    main()
