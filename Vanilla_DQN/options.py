print("****************************")
print("Loading Vanilla Q Learning Options")
print("****************************")

import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        

    def initialize(self):
        # experiment specifics
        #Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=64, help='batch size to be used')
        self.parser.add_argument('--memory_size', type=int, nargs='?', default=1000000, help='size of replay memory')
        self.parser.add_argument('--update_freq', type=int, nargs='?', default=32, help='how often to update the model')
        self.parser.add_argument('--lr', type=int, nargs='?', default=0.0001, help='learning rate')
        self.parser.add_argument('--discount_rate', type=int, nargs='?', default=0.9, help='rewards discount rate')
        self.parser.add_argument('--transfer_rate', type=int, nargs='?', default=0.001, help='transfer rate for soft update')

        #Env Options
        self.parser.add_argument('--env', type=str, nargs='?', default='Unity_Banana', help='Name of the OpenAI Env')
        self.parser.add_argument('--env_seed', type=int, nargs='?', default=0, help='random seed for the environment')

        #Training Options
        self.parser.add_argument('--num_episodes', type=int, nargs='?', default=3000, help='total number of training episodes')#3000
        self.parser.add_argument('--max_iteration', type=int, nargs='?', default=1000, help='max number of iterations per episodes')
        self.parser.add_argument('--min_epsilon', type=int, nargs='?', default=0.1, help='min value for epsilon')
        self.parser.add_argument('--decay', type=int, nargs='?', default=0.995, help='decay rate of epsilon per episode')
        self.parser.add_argument('--win_cond', type=int, nargs='?', default=13, help='Condition where the env is considered solved')

        #render
        self.parser.add_argument('--render', type=bool, nargs='?', default=True, help='Renders an episode at the end of training')

    def parse(self):
        self.initialize()
        self.parser.add_argument('-f')
        self.opt = self.parser.parse_args()

        return self.opt



"""

options = options()

opts = options.parse()
batch = opts.batch
"""