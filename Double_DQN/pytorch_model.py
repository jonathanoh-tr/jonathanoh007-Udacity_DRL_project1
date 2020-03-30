
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F


class DQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, layer_1=128, layer_2=64):
        '''
        :param state_size:
        :param action_size:
        :param seed:
        :param layer_1:
        :param layer_2:
        '''

        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        '''specify the network'''
        self.model = nn.Sequential(
            nn.Linear(state_size, layer_1),
            nn.ReLU(True),
            nn.Linear(layer_1, layer_1),
            nn.ReLU(True),
            nn.Linear(layer_1, layer_1),
            nn.ReLU(True),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(True),
            nn.Linear(layer_2, layer_2),
            nn.ReLU(True),
            nn.Linear(layer_2, action_size),
        )

    def forward(self, state):
        '''
        overrides the function forward
        :param state:
        :return:
        '''

        x = self.model(state)

        return x