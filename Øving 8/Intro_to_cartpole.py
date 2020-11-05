""" import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random aciton
env.close()
 """

# Agent
""" import gym
env = gym.make("CartPole-v0")
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close() """

# Play it yourself
import gym
from pyglet.window import key
import time
bool_do_not_quit = True
restart = False
scores = []
a = 0


def key_press(k, mod):
    global bool_do_not_quit, a, restart
    if k == key.SPACE:
        restart = True
    if k == key.ESCAPE:
        bool_do_not_quit = False
    if k == key.Q:
        bool_do_not_quit = False
    if k == key.LEFT:
        a = 0
    if k == key.RIGHT:
        a = 1


def play_CartPole_yourself():
    env = gym.make("CartPole-v0")
    env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    while bool_do_not_quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        t1 = time.time()
        while bool_do_not_quit:
            observation, reward, done, info = env.step(a)
            time.sleep(0.1)
            total_reward += reward
            steps += 1
            env.render()
            # if done or restart:
            if restart:
                t1 = time.time() - t1
                scores.append(total_reward)
                print("Trial", len(scores), "| Score:", total_reward,
                      "|", steps, "steps | %0.2fs." % t1)
                break
    env.close()


play_CartPole_yourself()
