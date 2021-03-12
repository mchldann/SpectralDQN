import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from random import randrange
from dqn import DQN

class AgentDQN(object):

    def __init__(self, manager, agent_params):

        self.manager = manager

        # Rainbow enhancements
        self.noisy_nets = agent_params["noisy_nets"]
        self.dueling = agent_params["dueling"]
        self.double_dqn = agent_params["double_dqn"]
        self.prioritized = agent_params["prioritized"]

        self.discount = agent_params["discount"]
        
        self.ep_start = agent_params["ep_start"]
        self.ep = self.ep_start
        self.ep_end = agent_params["ep_end"]
        self.ep_endt = agent_params["ep_endt"]

        self.eval_ep = agent_params["eval_ep"]

        # For error magnitude tracking
        self.error_mag_beta = 0.001
        self.error_mag_updates = 0.0
        self.error_mag_biased = 0.0
        self.error_mag = 0.0

        self.adam_lr = agent_params["adam_lr"]
        self.adam_eps = agent_params["adam_eps"]
        self.adam_beta1 = agent_params["adam_beta1"]
        self.adam_beta2 = agent_params["adam_beta2"]

        self.huber_clip = agent_params["huber_clip"]

        self.network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, 1, self.noisy_nets, self.dueling, False, False, False)
        self.target_network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, 1, self.noisy_nets, self.dueling, False, False, False)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.adam_lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)


    def learn(self):

        assert self.manager.transitions.size() > self.manager.minibatch_size, 'Not enough transitions stored to learn'

        # TODO: Make compatible with n-step returns by reinstating ret_partial and performing the clipping in neural_q_learner instead
        s, a, r, _, _, _, s_plus_n, term_under_n, tree_idx, weight = self.manager.transitions.sample(self.manager.minibatch_size)

        # Normalise by max importance-sampling weight from batch. (Weights default to 1 if self.prioritized == False.)
        if self.prioritized:
            weight /= np.amax(weight)
            if self.manager.gpu >= 0:
                weight = torch.from_numpy(weight).float().cuda()
            else:
                weight = torch.from_numpy(weight).float().cuda()

        if self.manager.gpu >= 0:
            r = torch.from_numpy(r).float().cuda()
            term_under_n = torch.from_numpy(term_under_n).float().cuda()
            a_tens = torch.from_numpy(a).cuda().unsqueeze(1).long()
        else:
            r = torch.from_numpy(r).float().cpu()
            term_under_n = torch.from_numpy(term_under_n).float().cpu()
            a_tens = torch.from_numpy(a).cpu().unsqueeze(1).long()

        if self.noisy_nets:
            self.target_network.reset_noise()

        q_tpn_values_main = self.target_network.forward(s_plus_n).detach()

        if self.double_dqn:
            q_tpn_values_live = self.network.forward(s_plus_n).detach()

        # Calculate q-values at time t
        q_values = self.network.forward(s).gather(1, a_tens).squeeze()

        if self.double_dqn:
            q_vals_for_greedy_action = q_tpn_values_live
        else:
            q_vals_for_greedy_action = q_tpn_values_main

        _, greedy_act = q_vals_for_greedy_action.max(1)
        greedy_act = greedy_act.unsqueeze(1)

        value_tp1 = q_tpn_values_main.gather(1, greedy_act).squeeze()

        target_overall = torch.ones_like(term_under_n).sub(term_under_n).mul(self.discount ** self.manager.n_step_n).mul(value_tp1).add(r)

        error = q_values - target_overall

        # Huber loss
        error.clamp_(-self.huber_clip, self.huber_clip)

        # Prioritized weights
        if self.prioritized:
            error.mul_(weight.unsqueeze(1).expand(-1, 1))

        error.div_(self.manager.minibatch_size)

        # Track error magnitude (just for debugging purposes)
        error_mag = error.pow(2.0).sum().sqrt().item()
        self.error_mag_biased = (1.0 - self.error_mag_beta) * self.error_mag_biased + self.error_mag_beta * error_mag
        self.error_mag_updates = self.error_mag_updates + 1.0
        self.error_mag = self.error_mag_biased / (1.0 - (1.0 - self.error_mag_beta) ** self.error_mag_updates)

        self.optimizer.zero_grad()
        q_values.backward(error.data)
        self.optimizer.step()

        # Update priorities of sampled transitions (using the error for the main net)
        if self.prioritized:
            self.manager.transitions.update_priorities(tree_idx, error_for_priority_update)


    def increment_timestep(self):

        if self.prioritized:
            self.priority_weight = self.priority_weight_start + (1.0 - self.priority_weight_start) * (float(self.manager.numSteps) / self.priority_weight_anneal_endt)


    def reset_live_networks_noise(self):

        if self.noisy_nets:
            self.network.reset_noise()


    def refresh_target(self):

        self.target_network.load_state_dict(self.network.state_dict())


    def act(self, state, is_eval):

        if is_eval:
            self.ep = self.eval_ep
        else:
            self.ep = self.ep_end + np.maximum(0, (self.ep_start - self.ep_end) * (self.ep_endt - np.maximum(0, self.manager.numSteps - self.manager.learn_start)) / self.ep_endt)

        a = self.greedy(state)

        # Epsilon greedy
        if np.random.uniform() < self.ep:
            return randrange(self.manager.n_actions)
        else:
            return a


    def greedy(self, state):

        # Turn single state into minibatch.  Needed for convolutional nets.
        assert state.dim() >= 3, 'Input must be at least 3D'

        q = self.network.forward(state).detach().squeeze()

        # TODO: Add random tiebreaking
        best_q, action_selected = q.max(0)
        best_q = best_q.item()
        action_selected = action_selected.item()

        self.manager.bestq[0] = best_q

        return action_selected


    def save_model(self):

        path = self.manager.log_dir + 'model.chk'
        print 'Saving model to ' + path + '...'

        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, path)


    def load_model(self):

        path = self.manager.log_dir + 'model.chk'
        print 'Loading model from ' + path + '...'

        checkpoint = torch.load(path)
        
        self.network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, 1, self.noisy_nets, self.dueling, False, False, False)
        self.target_network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, 1, self.noisy_nets, self.dueling, False, False, False)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.adam_lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

