import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os
import argparse
import matplotlib.pyplot as plt
from simulations.pendulum_sim import Simulation

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 100

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='run testing')
parser.add_argument('--model_path', type=str, default='models/model_1.pth', help='path to the model')
args = parser.parse_args()
print ("================ args ================")
print (args)
print ("======================================")


class ContinuousNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ContinuousNet, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu6(self.a1(x))
        mu = 2 * F.tanh(self.mu(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001      # avoid 0
        c1 = F.relu6(self.c1(x))
        values = self.v(c1)
        return mu, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)  # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.sim = Simulation()
        self.lnet = ContinuousNet(self.sim.state_space, self.sim.action_space) # local network

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.sim.reset_env() # Reset the env

            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(MAX_EP_STEP):
                if self.name == 'w0':
                    self.sim.show() # Render the gym env for worker 0 only
                    pass

                a = self.lnet.choose_action(v_wrap(s[None, :])) # Choose next action to perform, left or right by what magnitude
                s_, r, done, _ = self.sim.move(a.clip(-2, 2)) # Perform the action and record the state and rewards
                # Also take the boolean of whether the sim is done

                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a) # Buffer for action
                buffer_s.append(s) # Buffer for state
                buffer_r.append((r+8.1)/8.1) # normalize buffer for reward
                # TODO: Find out what is 8.1?

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_ # Set current state to the new state caused by action
                total_step += 1

        self.res_queue.put(None)


def run_test (gnet, opt):
    sim = Simulation()
    lnet = gnet    
    s = sim.reset_env() # Reset the env

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    total_step = 1

    while True:
        sim.show()

        a = lnet.choose_action(v_wrap(s[None, :])) # Choose next action to perform, left or right by what magnitude
        s_, r, done, _ = sim.move(a.clip(-2, 2)) # Perform the action and record the state and rewards
        # Also take the boolean of whether the sim is done

        ep_r += r
        buffer_a.append(a) # Buffer for action
        buffer_s.append(s) # Buffer for state
        buffer_r.append((r+8.1)/8.1)    # normalize buffer for reward
        # TODO: Find out what is 8.1?

        if total_step % UPDATE_GLOBAL_ITER == 0 or done:
            # TODO: Test if we really need the feedback training, maybe can remove this
            push_and_pull(opt, lnet, gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
            buffer_s, buffer_a, buffer_r = [], [], []
            if done:
                print ("SUCCESS", total_step)
                return
        s = s_ # Set current state to the new state caused by action
        total_step += 1


if __name__ == "__main__":
    sim = Simulation()
    gnet = ContinuousNet(sim.state_space, sim.action_space) # global network
    
    if args.test:
        gnet.load_state_dict(torch.load(args.model_path)) # Load the previously trained network

    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0002)  # global optimizer
    
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    if args.test:
        run_test(gnet, opt)

    else:
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
        torch.save(gnet.state_dict(), args.model_path)

        plt.plot(res)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.show()
