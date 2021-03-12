import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2
from PIL import Image
import torchvision as tv
from torchvision import transforms
from random import randrange
from agent_dqn import AgentDQN
from agent_dqn_squash import AgentDQNSquash
from agent_dqn_popart import AgentDQNPopArt
from agent_dqn_spectral import AgentDQNSpectral
from transition_table import TransitionTable
from preproc import Preproc
from dialog import Dialog

class NeuralQLearner(object):

    def __init__(self, agent_params, transition_params):

        self.agent_params = agent_params
        self.transition_params = transition_params

        # Rainbow enhancements
        self.noisy_nets = agent_params["noisy_nets"]
        self.dueling = agent_params["dueling"]

        self.agent_type = agent_params["agent_type"]
        self.log_dir = agent_params["log_dir"]
        self.ale = agent_params["ale"]
        self.gpu = agent_params["gpu"]
        self.n_actions = agent_params["n_actions"]
        self.hist_len = agent_params["hist_len"]
        self.downsample_w = agent_params["downsample_w"]
        self.downsample_h = agent_params["downsample_h"]
        self.max_reward = agent_params["max_reward"]
        self.min_reward = agent_params["min_reward"]

        self.num_freqs = agent_params["num_freqs"]
        self.spectrum_base = agent_params["spectrum_base"]

        self.discount = agent_params["discount"]
        self.learn_start = agent_params["learn_start"]
        self.update_freq = agent_params["update_freq"]
        self.n_replay = agent_params["n_replay"]
        self.n_step_n = agent_params["n_step_n"]
        self.minibatch_size = agent_params["minibatch_size"]
        self.target_refresh_steps = agent_params["target_refresh_steps"]
        self.show_graphs = agent_params["show_graphs"]
        self.graph_save_freq = agent_params["graph_save_freq"]

        self.in_channels = self.hist_len

        self.numSteps = 0

        # For inserting complete episodes into the experience replay cache
        self.current_episode = []
        self.lastState = None
        self.lastAction = None
        self.lastTerminal = False
        self.lastTimedOut = False

        if self.agent_type == 'dqn':
            self.agent = AgentDQN(self, agent_params)

        elif self.agent_type == 'dqn_squash':
            self.agent = AgentDQNSquash(self, agent_params)

        elif self.agent_type == 'dqn_popart':
            self.agent = AgentDQNPopArt(self, agent_params)

        elif self.agent_type == 'dqn_spectral':
            self.agent = AgentDQNSpectral(self, agent_params)

        else:
            sys.exit('Agent type (' + self.agent_type + ') not recognised!')

        self.bestq = np.zeros((1), dtype=np.float32)
            
        self.preproc = Preproc(self.agent_params, self.downsample_w, self.downsample_h)
        self.transitions = TransitionTable(self.transition_params, self.agent)

        self.q_values_plot = Dialog()
        self.error_mag_plot = Dialog()

        # For debugging
        self.image_dump_counter = 0
        self.experience_dump = False


    def add_episode_to_cache(self):

        IDX_STATE = 0
        IDX_ACTION = 1
        IDX_EXTRINSIC_REWARD = 2
        IDX_TERMINAL = 3
        IDX_TIMED_OUT = 4

        ep_length = len(self.current_episode)
        ret_partial = np.zeros((ep_length), dtype=np.float32)
        ret_partial_pos_spectral = np.zeros((ep_length, self.num_freqs), dtype=np.float32)
        ret_partial_neg_spectral = np.zeros((ep_length, self.num_freqs), dtype=np.float32)

        r_spectral_pos = np.zeros((ep_length, self.num_freqs), dtype=np.float32)
        r_spectral_neg = np.zeros((ep_length, self.num_freqs), dtype=np.float32)

        ret_spectral_pos = np.zeros((ep_length, self.num_freqs), dtype=np.float32)
        ret_spectral_neg = np.zeros((ep_length, self.num_freqs), dtype=np.float32)

        last_n_rewards_discounted = np.zeros((self.n_step_n), dtype=np.float32)
        last_n_rewards_idx = 0

        last_n_pos_spectral_rewards_discounted = np.zeros((self.n_step_n, self.num_freqs), dtype=np.float32)
        last_n_pos_spectral_rewards_idx = 0

        last_n_neg_spectral_rewards_discounted = np.zeros((self.n_step_n, self.num_freqs), dtype=np.float32)
        last_n_neg_spectral_rewards_idx = 0

        i = ep_length - 1

        for j in range(0, self.num_freqs):
            r_spectral = np.sign(self.current_episode[i][IDX_EXTRINSIC_REWARD]) * np.clip((np.absolute(self.current_episode[i][IDX_EXTRINSIC_REWARD]) - (self.spectrum_base ** j - 1.0) / (self.spectrum_base - 1.0)) / (self.spectrum_base ** j), 0.0, 1.0)
            r_spectral_pos[i][j] = np.clip(r_spectral, 0.0, 1.0)
            r_spectral_neg[i][j] = np.clip(-r_spectral, 0.0, 1.0)

        np.copyto(ret_spectral_pos[i], r_spectral_pos[i])
        np.copyto(ret_spectral_neg[i], r_spectral_neg[i])

        np.copyto(last_n_pos_spectral_rewards_discounted[last_n_pos_spectral_rewards_idx], r_spectral_pos[i])
        last_n_pos_spectral_rewards_idx = (last_n_pos_spectral_rewards_idx + 1) % self.n_step_n

        np.copyto(last_n_neg_spectral_rewards_discounted[last_n_neg_spectral_rewards_idx], r_spectral_neg[i])
        last_n_neg_spectral_rewards_idx = (last_n_neg_spectral_rewards_idx + 1) % self.n_step_n

        last_n_rewards_discounted[last_n_rewards_idx] = self.current_episode[i][IDX_EXTRINSIC_REWARD]
        last_n_rewards_idx = (last_n_rewards_idx + 1) % self.n_step_n

        ret_partial[i] = np.sum(last_n_rewards_discounted)

        np.copyto(ret_partial_pos_spectral[i], np.sum(last_n_pos_spectral_rewards_discounted, 0))
        np.copyto(ret_partial_neg_spectral[i], np.sum(last_n_neg_spectral_rewards_discounted, 0))

        i = ep_length - 2
        while i >= 0:

            for j in range(0, self.num_freqs):
                r_spectral = np.sign(self.current_episode[i][IDX_EXTRINSIC_REWARD]) * np.clip((np.absolute(self.current_episode[i][IDX_EXTRINSIC_REWARD]) - (self.spectrum_base ** j - 1.0) / (self.spectrum_base - 1.0)) / (self.spectrum_base ** j), 0.0, 1.0)
                r_spectral_pos[i][j] = np.clip(r_spectral, 0.0, 1.0)
                r_spectral_neg[i][j] = np.clip(-r_spectral, 0.0, 1.0)

            np.copyto(ret_spectral_pos[i], self.discount * ret_spectral_pos[i + 1] + r_spectral_pos[i])
            np.copyto(ret_spectral_neg[i], self.discount * ret_spectral_neg[i + 1] + r_spectral_neg[i])

            last_n_pos_spectral_rewards_discounted = last_n_pos_spectral_rewards_discounted * self.discount
            np.copyto(last_n_pos_spectral_rewards_discounted[last_n_pos_spectral_rewards_idx], r_spectral_pos[i])
            last_n_pos_spectral_rewards_idx = (last_n_pos_spectral_rewards_idx + 1) % self.n_step_n

            last_n_neg_spectral_rewards_discounted = last_n_neg_spectral_rewards_discounted * self.discount
            np.copyto(last_n_neg_spectral_rewards_discounted[last_n_neg_spectral_rewards_idx], r_spectral_neg[i])
            last_n_neg_spectral_rewards_idx = (last_n_neg_spectral_rewards_idx + 1) % self.n_step_n

            last_n_rewards_discounted = last_n_rewards_discounted * self.discount
            last_n_rewards_discounted[last_n_rewards_idx] = self.current_episode[i][IDX_EXTRINSIC_REWARD]
            last_n_rewards_idx = (last_n_rewards_idx + 1) % self.n_step_n

            ret_partial[i] = np.sum(last_n_rewards_discounted)

            np.copyto(ret_partial_pos_spectral[i], np.sum(last_n_pos_spectral_rewards_discounted, 0))
            np.copyto(ret_partial_neg_spectral[i], np.sum(last_n_neg_spectral_rewards_discounted, 0))

            i -= 1

        r_spectral = np.concatenate((r_spectral_neg, r_spectral_pos), axis=1)
        ret_spectral = np.concatenate((ret_spectral_neg, ret_spectral_pos), axis=1)

        # Add episode to the cache
        i = 0
        while i < ep_length:
            self.transitions.add(self.current_episode[i][IDX_STATE], self.current_episode[i][IDX_ACTION], self.current_episode[i][IDX_EXTRINSIC_REWARD], r_spectral[i], ret_spectral[i], ret_partial[i], ret_partial_pos_spectral[i], ret_partial_neg_spectral[i], self.current_episode[i][IDX_TERMINAL], self.current_episode[i][IDX_TIMED_OUT], ep_length - 1 - i)
            i += 1


    def handle_last_terminal(self):

        self.add_episode_to_cache()
        self.current_episode = []


    def handle_terminal(self, is_eval):

        if not is_eval:
            self.error_mag_plot.add_data_point("errorMag", self.numSteps, [self.agent.error_mag], False, self.show_graphs, self.log_dir)


    def perceive(self, reward, rawstate, terminal, properly_terminal, is_eval, timed_out):

        #rawstate = torch.from_numpy(rawstate)
        #state = self.preproc.forward(rawstate)

        state = cv2.resize(rawstate, (84, 84), interpolation=cv2.INTER_AREA)
        state = torch.from_numpy(state).float()

        self.transitions.add_recent_state(state, terminal)
        curState = self.transitions.get_recent().reshape(1, self.hist_len, self.downsample_w, self.downsample_h)

        if self.gpu >= 0:
            curState = curState.cuda()
        else:
            curState = curState.cpu()

        # Clip the reward
        reward = np.minimum(reward, self.max_reward)
        reward = np.maximum(reward, self.min_reward)

        if not is_eval:
            # Store transition s, a, r, s'
            if self.lastState is not None:
                self.current_episode.append([self.lastState, self.lastAction, reward, self.lastTerminal, self.lastTimedOut])

        if properly_terminal:
            self.handle_terminal(is_eval)

        if not is_eval:
            # Necessary to process episode once lastTerminal is True so that each experience in the cache has an endpoint
            if self.lastTerminal:
                self.handle_last_terminal()

        # In the noisy nets paper it appears that the noise is reset every step, but in Kaixhin's Rainbow implementation
        # the reset frequency is set equal to the training frequency. Here we're following Kaixhin.
        if self.noisy_nets and (self.numSteps % self.update_freq == 0):
            self.agent.reset_live_networks_noise()

        # Select action
        actionIndex = 0
        if not terminal:
            actionIndex = self.agent.act(curState, is_eval)

        self.q_values_plot.add_data_point("bestq", self.numSteps, self.bestq, True, self.show_graphs, None)
        if self.ale.isUpdatingScreenImage():
            #self.q_values_plot.add_data_point("bestq", self.numSteps, self.bestq, True, self.show_graphs, None)
            self.q_values_plot.update_image('eps = ' + str(self.agent.ep))

        if not is_eval:

            if self.numSteps % self.graph_save_freq == 0:
                self.q_values_plot.update_image('eps = ' + str(self.agent.ep))
                self.q_values_plot.save_image(self.log_dir)
                self.error_mag_plot.save_image(self.log_dir)

            self.numSteps += 1
            self.agent.increment_timestep()

            # Do some Q-learning updates
            if (self.numSteps % self.update_freq == 0) and (self.numSteps > self.learn_start):
                for i in range(0, self.n_replay):
                    self.agent.learn()

            self.lastState = state.clone()
            self.lastAction = actionIndex
            self.lastTerminal = terminal
            self.lastTimedOut = timed_out

            if self.numSteps % self.target_refresh_steps == 0:
                self.agent.refresh_target()

        return actionIndex

