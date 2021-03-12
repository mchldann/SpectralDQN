import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, std_init=0.5):

        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()


    def reset_parameters(self):

        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))


    def _scale_noise(self, size):

        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())


    def reset_noise(self):

        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)


    def forward(self, input):

        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):

    def __init__(self, gpu, in_channels, num_actions, num_freqs, noisy, dueling, zero_init_final_layer):
        super(DQN, self).__init__()

        self.num_actions = num_actions
        self.num_freqs = num_freqs

        self.noisy = noisy
        self.dueling = dueling

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        if self.dueling:
            if self.noisy:
                self.fc1_v = NoisyLinear(in_features=7*7*64, out_features=512)
                self.fc1_a = NoisyLinear(in_features=7*7*64, out_features=512)
                self.fc2_v = NoisyLinear(in_features=512, out_features=1)
                self.fc2_a = NoisyLinear(in_features=512, out_features=num_actions*self.num_freqs)
            else:
                self.fc1_v = nn.Linear(in_features=7*7*64, out_features=512)
                self.fc1_a = nn.Linear(in_features=7*7*64, out_features=512)
                self.fc2_v = nn.Linear(in_features=512, out_features=1)
                self.fc2_a = nn.Linear(in_features=512, out_features=num_actions*self.num_freqs)
        else:
            if self.noisy:
                self.fc1 = NoisyLinear(in_features=7*7*64, out_features=512)
                self.fc2 = NoisyLinear(in_features=512, out_features=num_actions*self.num_freqs)
            else:
                self.fc1 = nn.Linear(in_features=7*7*64, out_features=512)
                self.fc2 = nn.Linear(in_features=512, out_features=num_actions*self.num_freqs)

        self.relu = nn.ReLU()

        self.conv1.bias.data.fill_(0.0)
        self.conv2.bias.data.fill_(0.0)
        self.conv3.bias.data.fill_(0.0)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.bias.data.fill_(0.0)

        if zero_init_final_layer:
            self.fc2.weight.data.fill_(0.0)

        if gpu >= 0:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.relu.cuda()

            if self.dueling:
                self.fc1_v.cuda()
                self.fc1_a.cuda()
                self.fc2_v.cuda()
                self.fc2_a.cuda()
            else:
                self.fc1.cuda()
                self.fc2.cuda()

        else:
            self.conv1.cpu()
            self.conv2.cpu()
            self.conv3.cpu()
            self.relu.cpu()

            if self.dueling:
                self.fc1_v.cpu()
                self.fc1_a.cpu()
                self.fc2_v.cpu()
                self.fc2_a.cpu()
            else:
                self.fc1.cpu()
                self.fc2.cpu()


    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        if self.dueling:
            v = self.fc2_v(self.relu(self.fc1_v(x))) # Value stream
            a = self.fc2_a(self.relu(self.fc1_a(x))) # Advantage stream
            x = v + a - a.mean(1, keepdim=True)  # Combine streams
        else:
            x = self.fc2(self.relu(self.fc1(x)))

        if self.num_freqs > 1:
            x = x.view(-1, self.num_freqs, self.num_actions)
            x = x.permute(0, 2, 1)

        return x


    def reset_noise(self):

        if self.noisy:
            for name, module in self.named_children():
                if 'fc' in name:
                    module.reset_noise()

