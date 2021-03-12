#!/usr/bin/env python
# This is a direct port to python of the shared library example from
# ALE provided in doc/examples/sharedLibraryInterfaceWithModesExample.cpp
import sys
import numpy as np
import os
import torch
import torchvision as tv
sys.path.append("..")

from random import randrange
from ale_python_interface import ALEInterface
from neural_q_learner import NeuralQLearner
from dialog import Dialog

if len(sys.argv) < 2:
  print 'Usage:', sys.argv[0], 'rom_file'
  sys.exit()

score_plot = Dialog()
perceived_score_plot = Dialog()

ale = ALEInterface()
agent_params = {}
agent_params["agent_type"] = "dqn_spectral" # dqn, dqn_squash, dqn_popart, dqn_spectral

#### ENVIRONMENT SETTINGS #####

experiment_type = "main" # main, exponential_pong, ponglantis
pong_score_exponent = 2.0
ponglantis_transition_score = 10

random_seed = 123

use_nature_paper_settings = True

if use_nature_paper_settings:
    use_minimal_actions = True
    terminal_on_life_loss = True
    repeat_action_prob = 0.0
    no_ops = 30
else:
    use_minimal_actions = False
    terminal_on_life_loss = False
    repeat_action_prob = 0.25
    no_ops = 0

frame_pooling_style = "max_pool" # color_averaging, max_pool

###############################

agent_params["verbose"] = False # True

agent_params["log_dir"] = os.path.dirname(os.path.realpath(__file__))
agent_params["log_dir"] = agent_params["log_dir"] + "/results_by_game/" + sys.argv[1] + '_' + agent_params["agent_type"] + "/"
if not os.path.exists(agent_params["log_dir"]):
    os.makedirs(agent_params["log_dir"])

eval_freq = 250000 # As per Rainbow paper
eval_steps = 125000 # As per Rainbow paper
eval_start_time = 999999999 # Set to 0 to enable periodic evaluations
max_frames_per_episode = 108000 # As per Rainbow paper
agent_params["eval_ep"] = 0.001

# Rainbow enhancements
agent_params["noisy_nets"] = False
agent_params["dueling"] = False
agent_params["double_dqn"] = False
agent_params["prioritized"] = False
agent_params["n_step_n"] = 3 # 1

agent_params["split_pos_neg_freqs"] = False

# Mixed Monte Carlo settings
agent_params["enable_mmc"] = False
agent_params["mmc_sigma_cutoff"] = 0.0
agent_params["mmc_max"] = 0.0
agent_params["mmc_divisor"] = 1.0 # agent_params["mmc_sigma_cutoff"] / agent_params["mmc_max"]

agent_params["error_scale_style"] = "normalize_by_var" # "normalize_by_var, none"

agent_params["mag_constant"] = 1.0 # 50.0 # 20.0 # 100.0 # 100.0
agent_params["mag_scale"] = 1.0 # 0.02

# For estimating the variance of the targets (used by Pop-Art and the main agent)
agent_params["stats_update_beta"] = 3E-4 # 0.01

agent_params["freq_normalisation_eps"] = np.sqrt(10.0) * 0.01 # 0.01 # 0.0001 # 1E-4 # 0.1 # 0.01
agent_params["popart_eps"] = 0.001

# For prioritized experience replay (***not fully implemented yet*** -- left here to potentially experiment with Rainbow modifications down the track)
agent_params["priority_exponent"] = 0.5 # "alpha" in the prioritized experience replay paper.
agent_params["priority_weight_start"] = 0.4 # "beta" in the prioritized experience replay paper. Annealed to 1 over the course of training.
agent_params["priority_weight_anneal_endt"] = 200000000

agent_params["use_targ_compress"] = False # True
agent_params["scaling_eps"] = 10E-3 # Agent57 and Never Give Up both use 10E-3. Pohlen et al. ("Observe and Look Further") use 10E-2.

agent_params["num_freqs"] = 21
agent_params["spectrum_base"] = 2.0

agent_params["correct_error_magnitude"] = True
agent_params["max_error_magnitude"] = 0.01
agent_params["min_error_divisor"] = 1E-4
agent_params["error_mag_beta"] = 0.001

agent_params["huber_clip"] = 1.0 # Only applies to dqn and dqn_spectral

# Adam optimizer settings
agent_params["adam_lr"] = 0.000025
agent_params["adam_beta1"] = 0.9
agent_params["adam_beta2"] = 0.999
agent_params["adam_eps"] = 0.005 / 32.0

agent_params["ale"] = ale
agent_params["gpu"] = 0 # Set to -1 for CPU processing
agent_params["frame_skip"] = 4
agent_params["use_rgb_for_raw_state"] = False
agent_params["hist_len"] = 4
agent_params["downsample_w"] = 84
agent_params["downsample_h"] = 84

if agent_params["agent_type"] == "dqn":
    agent_params["max_reward"] = 1.0
    agent_params["min_reward"] = -1.0
else:
    agent_params["max_reward"] = float("inf")
    agent_params["min_reward"] = float("-inf")

if agent_params["noisy_nets"]:
    agent_params["ep_start"] = 0.0
    agent_params["ep_end"] = 0.0
    agent_params["ep_endt"] = 1000000
else:
    agent_params["ep_start"] = 1
    agent_params["ep_end"] = 0.01
    agent_params["ep_endt"] = 1000000

agent_params["discount"] = (0.99 ** (1.0 / 3.0)) # 0.99
agent_params["learn_start"] = 50000
agent_params["update_freq"] = 4
agent_params["n_replay"] = 1
agent_params["minibatch_size"] = 32
agent_params["target_refresh_steps"] = 10000
agent_params["show_graphs"] = False
agent_params["graph_save_freq"] = 1000

if frame_pooling_style == "color_averaging":
    ale.setBool('color_averaging', True)
else:
    ale.setBool('color_averaging', False)

transition_params = {}
transition_params["agent_params"] = agent_params
transition_params["replay_size"] = 1000000
transition_params["hist_spacing"] = 1
transition_params["bufferSize"] = 512

# Get & Set the desired settings
ale.setInt('random_seed', random_seed)
ale.setFloat("repeat_action_probability", repeat_action_prob);
#ale.setInt("max_num_frames_per_episode", max_frames_per_episode); # Now handling this manually

ale.setBool('sound', False)

# If we're using color averaging then there's no need to process every individual frame (since the ALE handles color averaging).
# If we're taking the max pixel intensity of successive frames then we need every individual frame so that we can perform the max manually.
if frame_pooling_style == "color_averaging":
    ale.setInt('frame_skip', agent_params["frame_skip"])

# Set USE_SDL to true to display the screen. ALE must be compilied
# with SDL enabled for this to work. On OSX, pygame init is used to
# proxy-call SDL_main.
USE_SDL = False # True
if USE_SDL:
  ale.setBool('display_screen', True)
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX

# Load the ROM file
ale.loadROM('../roms/' + sys.argv[1] + '.bin')

#Get the list of available modes and difficulties
avail_modes = ale.getAvailableModes()
avail_diff  = ale.getAvailableDifficulties()

print 'Number of available modes: ', len(avail_modes)
print 'Number of available difficulties: ', len(avail_diff)

# Get the list of legal actions
if use_minimal_actions:
    action_set = ale.getMinimalActionSet()
else:
    action_set = ale.getLegalActionSet()

agent_params["n_actions"] = len(action_set)

agent_manager = NeuralQLearner(agent_params, transition_params)

ale.reset_game()
total_episode_steps = 0

training_frame_num = 0
is_eval = False
steps_since_eval_ran = 0
steps_since_eval_began = 0
eval_total_score = 0
eval_total_episodes = 0
best_eval_average = float("-inf")
reward = 0
perceived_reward = 0
total_reward = 0
total_perceived_reward = 0
total_reward_all_lives = 0
total_perceived_reward_all_lives = 0
total_positive_reward_all_lives = 0
total_negative_reward_all_lives = 0
terminal = False
lives = -1
terminal = False
properly_terminal = False
timed_out = False
prev_rawstate = None
act_reps = agent_params["frame_skip"]
loaded_atlantis = False

moving_average_score = 0
moving_average_perceived_score = 0

moving_average_score_mom = 0.98
moving_average_score_updates = 0

learning_curves_csv_filename = 'learning_curve.csv'
with open(agent_params["log_dir"] + learning_curves_csv_filename,'w') as fd:
    fd.write('num_training_steps,eval_score\n')

with open(agent_params["log_dir"] + 'plot_movingAverageScore.csv','w') as fd:
    fd.write('num_training_steps,movingAverageScore\n')

with open(agent_params["log_dir"] + 'plot_movingAveragePerceivedScore.csv','w') as fd:
    fd.write('num_training_steps,movingAveragePerceivedScore\n')

while training_frame_num < 999999999:

    if training_frame_num % (agent_params["graph_save_freq"] * agent_params["frame_skip"]) == 0 and not is_eval:
        score_plot.save_image(agent_params["log_dir"])
        perceived_score_plot.save_image(agent_params["log_dir"])

    if agent_params["use_rgb_for_raw_state"]:
        rawstate = ale.getScreenRGB()
    else:
        rawstate = ale.getScreenGrayscale()

    if prev_rawstate is None:
        perceived_state = rawstate
    else:
        perceived_state = np.maximum(rawstate, prev_rawstate)

    a = agent_manager.perceive(perceived_reward, perceived_state, terminal, properly_terminal, is_eval, timed_out)

    if frame_pooling_style == "color_averaging":
        reward = ale.act(action_set[a])
        total_episode_steps += 1
        # Fix for score overflow in video pinball and asterix
        if (sys.argv[1] == "video_pinball" or sys.argv[1] == "asterix") and reward < 0:
            reward += 1000000
    else:
        i = 0
        reward = 0
        while i < act_reps:

            # Store the second last frame for max pooling
            if i == (act_reps - 1):
                if agent_params["use_rgb_for_raw_state"]:
                    prev_rawstate = ale.getScreenRGB()
                else:
                    prev_rawstate = ale.getScreenGrayscale()

            # Fix for score overflow in video pinball and asterix
            temp_reward = ale.act(action_set[a])
            total_episode_steps += 1
            if (sys.argv[1] == "video_pinball" or sys.argv[1] == "asterix") and temp_reward < 0:
                temp_reward += 1000000
            reward += temp_reward
            i += 1

    perceived_reward = reward

    if experiment_type == "exponential_pong":
        perceived_reward = np.power(pong_score_exponent, total_positive_reward_all_lives) * reward

    lives_temp = ale.lives()
    properly_terminal = ale.game_over()

    timed_out = False
    if total_episode_steps >= max_frames_per_episode:
        timed_out = True
        properly_terminal = True

    # Hack to fix issue where game isn't considered terminal because m_started == False
    if (sys.argv[1] == "breakout" or sys.argv[1] == "chopper_command") and (lives_temp == 0):
        properly_terminal = True

    # Check if we need to terminate from losing a life
    terminal = properly_terminal
    if terminal_on_life_loss and (lives_temp < lives):
        terminal = True

        # Terminate after one life lost on the second phase of ponglantis
        if experiment_type == "ponglantis" and loaded_atlantis:
            properly_terminal = True

    lives = lives_temp

    total_reward += reward
    total_perceived_reward += perceived_reward

    if reward < 0:
        total_negative_reward_all_lives += reward
    else:
        total_positive_reward_all_lives += reward

    if experiment_type == "ponglantis":
        if total_positive_reward_all_lives >= ponglantis_transition_score and not loaded_atlantis:
            ale.loadROM('../roms/atlantis.bin')
            ale.reset_game()
            lives_temp = ale.lives()
            loaded_atlantis = True

    if is_eval:
        steps_since_eval_began += 1
    else:
        steps_since_eval_ran += 1
        training_frame_num += 1

    if terminal:
        if is_eval:
            print 'Evaluation frame: ' + str(steps_since_eval_began) + ', episode ended with score: ' + str(total_reward)
        else:
            print 'Training frame: ' + str(training_frame_num) + ', episode ended with score: ' + str(total_reward)

        total_reward_all_lives += total_reward
        total_perceived_reward_all_lives += total_perceived_reward

        total_reward = 0
        total_perceived_reward = 0

    if properly_terminal:

        if is_eval:
            eval_total_score += total_reward_all_lives
            eval_total_episodes += 1
        else:

            moving_average_score = moving_average_score_mom * moving_average_score + (1.0 - moving_average_score_mom) * total_reward_all_lives
            moving_average_perceived_score = moving_average_score_mom * moving_average_perceived_score + (1.0 - moving_average_score_mom) * total_perceived_reward_all_lives

            moving_average_score_updates = moving_average_score_updates + 1
            zero_debiased_score = moving_average_score / (1.0 - moving_average_score_mom ** moving_average_score_updates)
            zero_debiased_perceived_score = moving_average_perceived_score / (1.0 - moving_average_score_mom ** moving_average_score_updates)

            score_plot.add_data_point("movingAverageScore", training_frame_num, [zero_debiased_score], False, agent_params["show_graphs"], agent_params["log_dir"])
            perceived_score_plot.add_data_point("movingAveragePerceivedScore", training_frame_num, [zero_debiased_perceived_score], False, agent_params["show_graphs"], agent_params["log_dir"])
            score_plot.update_image('Training frame = ' + str(training_frame_num))
            perceived_score_plot.update_image('Training frame = ' + str(training_frame_num))

        print 'Total score (all lives): ' + str(total_reward_all_lives)
        print 'Total perceived score (all lives): ' + str(total_perceived_reward_all_lives)
        print 'Total positive score (all lives): ' + str(total_positive_reward_all_lives)
        print 'Total negative (all lives): ' + str(total_negative_reward_all_lives)
        print ''
        total_reward_all_lives = 0
        total_perceived_reward_all_lives = 0
        total_positive_reward_all_lives = 0
        total_negative_reward_all_lives = 0

        if experiment_type == "ponglantis":
            ale.loadROM('../roms/pong.bin')
            loaded_atlantis = False

        ale.reset_game()
        total_episode_steps = 0
        prev_rawstate = None

        # Perform up to 30 random no-ops before starting
        if no_ops > 0:
            for _ in range(randrange(no_ops)):
                ale.act(0)
                total_episode_steps += 1
                if ale.game_over():
                    ale.reset_game()
                    total_episode_steps = 0
                    prev_rawstate = None

        if training_frame_num >= eval_start_time and steps_since_eval_ran >= eval_freq:
            is_eval = True
            eval_total_score = 0
            eval_total_episodes = 0

            while steps_since_eval_ran >= eval_freq:
                steps_since_eval_ran -= eval_freq

        elif steps_since_eval_began >= eval_steps:

            ave_eval_score = float(eval_total_score) / eval_total_episodes
            print 'Evaluation ended with average score of ' + str(ave_eval_score)

            with open(agent_params["log_dir"] + learning_curves_csv_filename,'a') as fd:
                fd.write(str(agent_manager.numSteps) + ',' + str(ave_eval_score) + '\n')

            if ave_eval_score > best_eval_average:
                best_eval_average = ave_eval_score
                print 'New best eval average of ' + str(best_eval_average)
                agent_manager.agent.save_model()
            else:
                print 'Did not beat best eval average of ' + str(best_eval_average)

            print ''

            is_eval = False
            steps_since_eval_began = 0

