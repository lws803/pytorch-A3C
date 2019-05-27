import gym
import numpy as np

class Simulation:
    def __init__ (self):
        self.env_wrapped = gym.make('CartPole-v0') # Get the env
        self.env = self.env_wrapped.unwrapped
        self.state_space = self.env_wrapped.observation_space.shape[0]
        self.action_space = self.env_wrapped.action_space.n

        self.state = self.env.reset()
        self.reward = None
        self.done = False


    def show (self):
        self.env.render()

    def reset_env (self):
        self.state = self.env.reset()
        return self.state

    def move (self, action):
        return self.env.step(action)


if __name__ == "__main__":
    sim = Simulation()
    s = sim.reset_env()
    while(1):
        sim.show()
        s = s_
