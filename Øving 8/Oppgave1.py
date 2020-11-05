"""
This is inspired from Jose Maynez' tutorial on medium.com
https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d
and Richard Brooker's video about the topic (https://www.youtube.com/watch?v=JNKvJEzuNsc)
"""

import gym
import math
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class CartPoleQAgent():
    def __init__(self, buckets=(6, 12),
                 num_episodes=500, min_lr=0.1,
                 min_epsilon=0.1, discount=1.0, decay=25):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon  # The percentage you want to explore
        self.discount = discount
        self.decay = decay
        self.buckets = buckets

        self.env = gym.make('CartPole-v1')
        self.steps = np.zeros(self.num_episodes)

        # This is the action-value function being initialized to 0's
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

        # Relevant for discretizer, with the observation on the form [position, velocity, angle, angular velocity]
        # self.upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        # self.lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        self.upper_bounds = [
            self.env.observation_space.high[2], math.radians(50)]
        self.lower_bounds = [
            self.env.observation_space.low[2], -math.radians(50)]

    def get_epsilon(self, t):
        # Ensures that there's almost at least a min_epsilon chance of randomly exploring, declines as we advance in episodes
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        # Learning rate declines as we add more episodes
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def discretize_state(self, observation):
        """Convert continous state intro a discrete state"""
        position, velocity, angle, angle_velocity = observation
        est = KBinsDiscretizer(n_bins=self.buckets,
                               encode='ordinal', strategy='uniform')
        est.fit([self.lower_bounds, self.upper_bounds])
        # return tuple(map(int, est.transform([[position, velocity, angle, angle_velocity]])[0]))
        return tuple(map(int, est.transform([[angle, angle_velocity]])[0]))

    def choose_action(self, state):
        # Chooses either a random action (exploration) or the action based on the best Q value from the Q-table (exploiting)
        if (np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update_q(self, state, action, reward, new_state):
        # Updates Q-table using the Q-formula.
        self.Q_table[state][action] += (self.learning_rate *
                                        (reward
                                         + self.discount *
                                         np.max(self.Q_table[new_state])
                                         - self.Q_table[state][action]))

    def train(self, print_steps=False, render=False):
        '''
        Trains agent making it go through the environment and choose actions based on its Q-table
        Builds the Q-table as it trains
        '''
        # Looping through each episode
        for e in range(self.num_episodes):
            # Initializes the state
            current_state = self.discretize_state(self.env.reset())
            self.learning_rate = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            # Looping through each step
            while not done:
                if render:
                    self.env.render()
                self.steps[e] += 1
                # Choose Action (A) from State (S)
                action = self.choose_action(current_state)
                # Run the action in the enviroment
                observation, reward, done, _ = self.env.step(action)
                # Create new S
                new_state = self.discretize_state(observation)
                # Update Q(S,A)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
                # When done is true we break out and continue to the next episode
            if print_steps:
                print(f"Trainingsession {e+1}:", int(self.steps[e]), "steps")
        print('Finished training!')

    def run(self):
        # Runs an episode while displaying the cartpole enviroment
        steps = 0
        done = False
        current_state = self.discretize_state(self.env.reset())
        while not done:
            self.env.render()
            steps += 1
            action = self.choose_action(current_state)
            observation, reward, done, _ = self.env.step(action)
            new_state = self.discretize_state(observation)
            current_state = new_state
        return steps


def main():
    agent = CartPoleQAgent(num_episodes=300)
    # Training
    print(f"Trains for {agent.num_episodes} episodes")
    agent.train(print_steps=True, render=False)
    # Showcase
    input("Press Enter to showcase one episode with the trained model")
    print("Number of steps:", agent.run())

    # Cartpole v0: max_episode_steps=200
    # Cartpole v1: max_episode_steps=500


if __name__ == "__main__":
    main()
