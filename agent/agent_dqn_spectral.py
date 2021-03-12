import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import time
from random import randrange
from dqn import DQN

class AgentDQNSpectral(object):

    def __init__(self, manager, agent_params):

        self.manager = manager

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.debug_print_freq = 999999999 # 25
        self.debug_print_i = 0

        self.verbose = agent_params["verbose"]

        self.scaling_eps = agent_params["scaling_eps"]

        self.use_targ_compress = agent_params["use_targ_compress"]

        self.enable_mmc = agent_params["enable_mmc"]
        self.mmc_sigma_cutoff = agent_params["mmc_sigma_cutoff"]
        self.mmc_max = agent_params["mmc_max"]
        self.mmc_divisor = agent_params["mmc_divisor"]

        # Rainbow enhancements
        self.noisy_nets = agent_params["noisy_nets"]
        self.dueling = agent_params["dueling"]
        self.double_dqn = agent_params["double_dqn"]
        self.prioritized = agent_params["prioritized"]

        if self.prioritized:
            self.priority_exponent = agent_params["priority_exponent"]
            self.priority_weight_start = agent_params["priority_weight_start"]
            self.priority_weight_anneal_endt = agent_params["priority_weight_anneal_endt"]
            self.priority_weight = self.priority_weight_start

        self.discount = agent_params["discount"]
        self.max_theo_output = 1.0 / (1.0 - self.discount)

        self.ep_start = agent_params["ep_start"]
        self.ep = self.ep_start
        self.ep_end = agent_params["ep_end"]
        self.ep_endt = agent_params["ep_endt"]

        self.eval_ep = agent_params["eval_ep"]

        self.num_freqs = agent_params["num_freqs"]
        self.spectrum_base = agent_params["spectrum_base"]

        # For running gradient correction
        self.max_error_magnitude = agent_params["max_error_magnitude"]
        self.min_error_divisor = agent_params["min_error_divisor"]
        self.error_mag_beta = agent_params["error_mag_beta"]
        self.error_mag_updates = 0.0

        self.error_mag_biased = 0.0
        self.error_mag = 0.0
        self.error_mag_frequencies_biased = torch.zeros(self.num_freqs).to(self.device)
        self.error_mag_frequencies = torch.zeros(self.num_freqs).to(self.device)

        self.error_scale_style = agent_params["error_scale_style"]

        self.mag_constant = agent_params["mag_constant"]
        self.mag_scale = agent_params["mag_scale"]

        self.adam_lr = agent_params["adam_lr"]
        self.adam_eps = agent_params["adam_eps"]
        self.adam_beta1 = agent_params["adam_beta1"]
        self.adam_beta2 = agent_params["adam_beta2"]

        self.split_pos_neg_freqs = agent_params["split_pos_neg_freqs"]

        if self.split_pos_neg_freqs:
            self.head_count_mul = 2
        else:
            self.head_count_mul = 1

        self.active_freqs = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)
        self.active_freqs[0] = 1.0
        if self.split_pos_neg_freqs:
            self.active_freqs[self.num_freqs] = 1.0

        # For calculating running standard deviations
        self.stats_update_beta = agent_params["stats_update_beta"]
        self.freq_normalisation_eps = agent_params["freq_normalisation_eps"]

        self.stats_updates = 0.0
        self.sigma_updates = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)

        self.v_pre_mmc_biased = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)
        self.mu_pre_mmc_biased = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)
        self.sigma_pre_mmc = torch.ones(self.num_freqs * self.head_count_mul).to(self.device)

        self.v_post_mmc_biased = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)
        self.mu_post_mmc_biased = torch.zeros(self.num_freqs * self.head_count_mul).to(self.device)
        self.sigma_post_mmc = torch.ones(self.num_freqs * self.head_count_mul).to(self.device)

        self.multipliers = torch.pow(self.spectrum_base, torch.linspace(0, self.num_freqs - 1, self.num_freqs)).to(self.device)
        if self.split_pos_neg_freqs:
            self.multipliers = torch.cat((-self.multipliers, self.multipliers), 0)

        self.network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, self.num_freqs * self.head_count_mul, self.noisy_nets, self.dueling, True)
        self.target_network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, self.num_freqs * self.head_count_mul, self.noisy_nets, self.dueling, True)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.adam_lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)
        self.max_error = 0.0


    def h(self, x):

        x = x.mul(self.mag_constant)
        return x.abs().add_(1.0).sqrt_().sub_(1.0).mul_(x.sign()).add_(x.mul(self.scaling_eps)).mul_(self.mag_scale)


    def h_inverse(self, x):

        x = x.div(self.mag_scale)
        tmp = x.abs().add_(1.0 + self.scaling_eps).mul_(self.scaling_eps * 4.0).add_(1.0).sqrt_().sub_(1.0).div_(2.0 * self.scaling_eps).pow_(2.0).sub_(1.0).mul_(x.sign())
        return tmp.div_(self.mag_constant) 


    def learn(self):

        assert self.manager.transitions.size() > self.manager.minibatch_size, 'Not enough transitions stored to learn'

        s, a, _, _, ret_spectral, _, ret_partial_pos_spectral, ret_partial_neg_spectral, s_plus_n, term_under_n, timed_out_under_n, tree_idx, weight = self.manager.transitions.sample(self.manager.minibatch_size)

        if not self.split_pos_neg_freqs:
            # Merge positive and negative parts of the overall return and n-step return
            ret_spectral_neg = ret_spectral[:, 0:self.num_freqs]
            ret_spectral_pos = ret_spectral[:, self.num_freqs:(2 * self.num_freqs)]
            ret_spectral = ret_spectral_pos - ret_spectral_neg

        if self.split_pos_neg_freqs:
            ret_partial_spectral = np.concatenate((ret_partial_neg_spectral, ret_partial_pos_spectral), axis=1)
        else:
            ret_partial_spectral = ret_partial_pos_spectral - ret_partial_neg_spectral

        # Update active frequencies
        ret_partial_spectral_summed = np.sum(np.absolute(ret_partial_spectral), axis=0)

        for i in range(0, self.num_freqs * self.head_count_mul):

            if ret_partial_spectral_summed[i] > 0.0:

                # Initialise new frequency
                if self.active_freqs[i].item() < 1:
                    self.active_freqs[i] = 1

        # Normalise by max importance-sampling weight from batch. (Weights default to 1 if self.prioritized == False.)
        if self.prioritized:
            weight /= np.amax(weight)
            weight = torch.from_numpy(weight).to(self.device)

        ret_partial_spectral = torch.from_numpy(ret_partial_spectral).to(self.device)
        ret_spectral = torch.from_numpy(ret_spectral).to(self.device)
        term_under_n = torch.from_numpy(term_under_n).to(self.device).unsqueeze(1).expand(-1, 1)
        timed_out_under_n = torch.from_numpy(timed_out_under_n).to(self.device)
        a_tens = torch.from_numpy(a).to(self.device).unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_freqs * self.head_count_mul).long()

        self.stats_updates = self.stats_updates + 1.0

        if self.noisy_nets:
            self.target_network.reset_noise()

        q_tpn_values_main = self.target_network.forward(s_plus_n).detach()

        if self.double_dqn:
            q_tpn_values_live = self.network.forward(s_plus_n).detach()

            if self.use_targ_compress:
                q_tpn_values_live = self.h_inverse(q_tpn_values_live)

            q_tpn_values_live.clamp_(-self.max_theo_output, self.max_theo_output)

        # Calculate q-values at time t
        q_values = self.network.forward(s).gather(1, a_tens).squeeze()
        if self.verbose:
            print 'q_values[0] = ', q_values[0]
            if self.use_targ_compress:
                print 'q_values[0] (uncompressed) = ', self.h_inverse(q_values[0])

        # Uncompress the target net's output
        if self.use_targ_compress:
            q_tpn_values_main = self.h_inverse(q_tpn_values_main)

        q_tpn_values_main.clamp_(-self.max_theo_output, self.max_theo_output)

        if self.double_dqn:
            q_vals_for_greedy_action = q_tpn_values_live
        else:
            q_vals_for_greedy_action = q_tpn_values_main

        q_tmp = q_vals_for_greedy_action.clone()
        q_tmp = q_tmp.mul(self.multipliers.unsqueeze(0).unsqueeze(0).expand(self.manager.minibatch_size, self.manager.n_actions, -1))
        q_tmp = q_tmp.sum(2)

        _, greedy_act = q_tmp.max(1)
        greedy_act = greedy_act.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_freqs * self.head_count_mul)

        value_tp1 = q_tpn_values_main.gather(1, greedy_act).squeeze()

        target_overall = torch.ones_like(term_under_n).sub(term_under_n).mul(self.discount ** self.manager.n_step_n).mul(value_tp1).add(ret_partial_spectral)

        # Clamp to valid range
        target_overall.clamp_(-self.max_theo_output, self.max_theo_output)

        self.sigma_updates.add_(self.active_freqs)
        sig_updates = torch.max(self.sigma_updates, torch.ones_like(self.sigma_updates))

        if self.enable_mmc:

            # Calculate mu_batch and sigma_batch before MMC is applied.
            # This is used to determine how much of the Monte Carlo return should be used.
            sum_target_value = target_overall.sum(0)
            sum_sq_target_value = target_overall.pow(2.0).sum(0)

            mean_target_value = sum_target_value.div(self.manager.minibatch_size)
            mean_sq_target_value = sum_sq_target_value.div(self.manager.minibatch_size)

            # Update running averages
            self.mu_pre_mmc_biased.mul_(1.0 - self.stats_update_beta).add_(mean_target_value.mul(self.stats_update_beta))
            self.v_pre_mmc_biased.mul_(1.0 - self.stats_update_beta).add_(mean_sq_target_value.mul(self.stats_update_beta))

            # Debias
            v_new = self.v_pre_mmc_biased.div(torch.pow(torch.ones_like(sig_updates).sub(self.stats_update_beta), sig_updates).sub(1.0).mul(-1.0))
            mu_new = self.mu_pre_mmc_biased.div(torch.pow(torch.ones_like(sig_updates).sub(self.stats_update_beta), sig_updates).sub(1.0).mul(-1.0))

            self.sigma_pre_mmc = ((v_new - mu_new.pow(2.0)).clamp_(min=0.0)).sqrt_()

        # Compress the target
        if self.use_targ_compress:
            target_overall = self.h(target_overall)

        if self.enable_mmc:

            # Mixed Monte Carlo
            mmc_amt = self.sigma_pre_mmc.sub(self.mmc_sigma_cutoff).mul(-1.0).div(self.mmc_divisor).clamp(min=0.0, max=self.mmc_max)
            if self.verbose:
                print 'self.sigma_pre_mmc = ', self.sigma_pre_mmc
                print 'mmc_amt = ', mmc_amt
            mmc_amt = mmc_amt.unsqueeze(0).expand(self.manager.minibatch_size, -1)

            if self.use_targ_compress:
                target_overall = torch.ones_like(mmc_amt).sub(mmc_amt).mul(target_overall).add(self.h(ret_spectral).mul(mmc_amt))
            else:
                target_overall = torch.ones_like(mmc_amt).sub(mmc_amt).mul(target_overall).add(ret_spectral.mul(mmc_amt))

        # Don't learn from episode time-out terminations
        timed_out_mask = timed_out_under_n.unsqueeze(1).expand(-1, self.num_freqs * self.head_count_mul)
        target_overall = torch.ones_like(timed_out_mask).sub(timed_out_mask).mul(target_overall).add(timed_out_mask.mul(q_values.detach()))

        # Calculate mu_batch and sigma_batch *after* MMC (and target compression) is applied.
        sum_target_value = target_overall.sum(0)
        sum_sq_target_value = target_overall.pow(2.0).sum(0)

        mean_target_value = sum_target_value.div(self.manager.minibatch_size)
        mean_sq_target_value = sum_sq_target_value.div(self.manager.minibatch_size)

        # Update running averages
        self.mu_post_mmc_biased.mul_(1.0 - self.stats_update_beta).add_(mean_target_value.mul(self.stats_update_beta))
        self.v_post_mmc_biased.mul_(1.0 - self.stats_update_beta).add_(mean_sq_target_value.mul(self.stats_update_beta))

        # Debias
        v_new = self.v_post_mmc_biased.div(torch.pow(torch.ones_like(sig_updates).sub(self.stats_update_beta), sig_updates).sub(1.0).mul(-1.0))
        mu_new = self.mu_post_mmc_biased.div(torch.pow(torch.ones_like(sig_updates).sub(self.stats_update_beta), sig_updates).sub(1.0).mul(-1.0))

        self.sigma_post_mmc = ((v_new - mu_new.pow(2.0)).clamp_(min=0.0)).sqrt_()

        if self.verbose:
            print 'self.sigma_post_mmc = ', self.sigma_post_mmc

        error = q_values.detach() - target_overall

        # Prioritized weights
        if self.prioritized:
            error.mul_(weight.unsqueeze(1).expand(-1, 1))

        # Normalise error (across spectra)
        if self.verbose:
            print 'error[0] pre scale = ', error[0]

        if self.error_scale_style == "none":
            error_scale_fac = torch.ones_like(self.multipliers)
        elif self.error_scale_style == "normalize_by_var":
            error_scale_fac = torch.max(self.sigma_post_mmc, torch.ones_like(self.sigma_post_mmc).mul(self.freq_normalisation_eps)).pow(-2.0)
        else:
            sys.exit('Error scale style (' + self.error_scale_style + ') not recognised!')

        error.mul_(error_scale_fac.unsqueeze(0).expand(self.manager.minibatch_size, -1))

        if self.verbose:
            print 'error[0] post scale = ', error[0]

        max_post_scale = error.abs().max().item()
        self.max_error = np.maximum(self.max_error, max_post_scale)

        if self.verbose:
            print 'max_post_scale = ', max_post_scale, ', self.max_error = ', self.max_error

        error.div_(self.manager.minibatch_size)

        # Track error magnitude (purely for the stats)
        self.error_mag_updates = self.error_mag_updates + 1.0
        error_mag = error.pow(2.0).sum(1).sqrt().mean().item()
        self.error_mag_biased = (1.0 - self.error_mag_beta) * self.error_mag_biased + self.error_mag_beta * error_mag
        self.error_mag = self.error_mag_biased / (1.0 - (1.0 - self.error_mag_beta) ** self.error_mag_updates)

        self.optimizer.zero_grad()
        q_values.backward(error.data)

        # Adjust final layer gradients
        for name, p in self.network.named_parameters():

            if name == 'fc2.weight':
                for i in range(0, self.num_freqs):
                    p.grad.data[(self.manager.n_actions * i):(self.manager.n_actions * (i + 1)), :].div_(error_scale_fac[i].item())

            if name == 'fc2.bias':
                for i in range(0, self.num_freqs):
                    p.grad.data[(self.manager.n_actions * i):(self.manager.n_actions * (i + 1))].div_(error_scale_fac[i].item())

        self.optimizer.step()

        # Update priorities of sampled transitions (using the error for the main net)
        if self.prioritized:
            self.manager.transitions.update_priorities(tree_idx, error_for_priority_update)

        self.debug_print_i += 1
        if self.debug_print_i % self.debug_print_freq == 0:
            print 'torch.norm(self.network.conv1.weight.data, p=2) = ', torch.norm(self.network.conv1.weight.data, p=2)
            print 'torch.norm(self.network.conv2.weight.data, p=2) = ', torch.norm(self.network.conv2.weight.data, p=2)
            print 'torch.norm(self.network.conv3.weight.data, p=2) = ', torch.norm(self.network.conv3.weight.data, p=2)
            print 'torch.norm(self.network.fc1.weight.data, p=2) = ', torch.norm(self.network.fc1.weight.data, p=2)
            print 'torch.norm(self.network.fc2.weight.data, p=2) = ', torch.norm(self.network.fc2.weight.data, p=2)
            print 'self.network.conv1.weight.data.abs().mean() = ', self.network.conv1.weight.data.abs().mean()
            print 'self.network.conv2.weight.data.abs().mean() = ', self.network.conv2.weight.data.abs().mean()
            print 'self.network.conv3.weight.data.abs().mean() = ', self.network.conv3.weight.data.abs().mean()
            print 'self.network.fc1.weight.data.abs().mean() = ', self.network.fc1.weight.data.abs().mean()
            print 'self.network.fc2.weight.data.abs().mean() = ', self.network.fc2.weight.data.abs().mean()
            #print '-----'
            #for name, param in self.network.named_parameters():
            #    if param.requires_grad:
            #        print name + ": ", param.grad.data.abs().mean()

        if self.verbose:
            print ''


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

        # Uncompress
        if self.use_targ_compress:
            q = self.h_inverse(q)

        #q = self.get_conservative_q(q)

        q.clamp_(-self.max_theo_output, self.max_theo_output)

        q = q.mul(self.multipliers)

        q_cumsum = torch.cumsum(q, 1)

        q = q.sum(1)

        # TODO: Add random tiebreaking
        best_q, action_selected = q.max(0)
        best_q = best_q.item()
        action_selected = action_selected.item()

        q_cumsum = q_cumsum.narrow(0, action_selected, 1).squeeze()
        q_cumsum = q_cumsum.cpu().numpy()

        self.manager.bestq = np.flip(q_cumsum, axis=0)

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
        
        self.network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, self.num_freqs * self.head_count_mul, self.noisy_nets, self.dueling, True)
        self.target_network = DQN(self.manager.gpu, self.manager.in_channels, self.manager.n_actions, self.num_freqs * self.head_count_mul, self.noisy_nets, self.dueling, True)

        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.adam_lr, betas=(self.adam_beta1, self.adam_beta2), eps=self.adam_eps)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

