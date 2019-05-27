import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import argparse
import matplotlib.pyplot as plt
from simulations.cartpole_sim import Simulation

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 40
GAMMA = 0.9
MAX_EP = 3000

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='run testing')
args = parser.parse_args()


class DiscreteNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(DiscreteNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 200)
        self.pi2 = nn.Linear(200, a_dim)
        self.v1 = nn.Linear(s_dim, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = F.relu6(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu6(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = Simulation()
        self.lnet = DiscreteNet(self.env.state_space, self.env.action_space) # local network

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset_env()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w0':
                    # self.env.show()
                    pass
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.move(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

def run_test (gnet, opt):
    env = Simulation()
    lnet = gnet    
    s = env.reset_env() # Reset the env

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    total_step = 1

    while True:
        env.show()

        a = lnet.choose_action(v_wrap(s[None, :])) # Choose next action to perform, left or right by what magnitude
        s_, r, done, _ = env.move(a) # Perform the action and record the state and rewards
        # Also take the boolean of whether the sim is done

        ep_r += r
        buffer_a.append(a) # Buffer for action
        buffer_s.append(s) # Buffer for state
        buffer_r.append(r) # Buffer for rewards

        if total_step % UPDATE_GLOBAL_ITER == 0 or done:
            # TODO: Test if we really need the feedback training, maybe can remove this
            # push_and_pull(opt, lnet, gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
            buffer_s, buffer_a, buffer_r = [], [], []
            if done:
                print (total_step)
                s = env.reset_env() # Reset the env
                total_step = 0
                
        s = s_ # Set current state to the new state caused by action
        total_step += 1


if __name__ == "__main__":
    sim = Simulation()
    gnet = DiscreteNet(sim.state_space, sim.action_space)        # global network

    if args.test:
        gnet.load_state_dict(torch.load("model_discrete.pth")) # Load the previously trained network
    
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    if args.test:
        run_test(gnet, opt)

    else:
        # parallel training
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
        [w.start() for w in workers]
        res = []                    # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]

        print ("Saving model...")
        torch.save(gnet.state_dict(), "model_discrete.pth")

        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()
