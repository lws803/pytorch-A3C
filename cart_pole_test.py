import gym
import numpy as np
import time

env = gym.make('Pendulum-v0')

print ("==============================")
print (env.observation_space)
print (env.action_space)

s = env.reset()

while True:
    a = np.array([0.15253448])
    s, r, done, _ = env.step(a.clip(-2, 2))
    print (s, r, done)

    # time.sleep(0.5)
    env.render()
