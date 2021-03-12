# Spectral DQN

[Adapting to Reward Progressivity via Spectral Reinforcement Learning](https://openreview.net/forum?id=dyjPVUc2KB), ICLR'21.

## Abstract

We consider reinforcement learning tasks with *progressive rewards*; that is, tasks where the rewards tend to increase in magnitude over time. We hypothesise that this property may be problematic for value-based deep reinforcement learning agents, particularly if the agent must first succeed in relatively unrewarding regions of the task in order to reach more rewarding regions. To address this issue, we propose *Spectral DQN*, which decomposes the reward into frequencies such that the high frequencies only activate when large rewards are found. This allows the training loss to be balanced so that it gives more even weighting across small and large reward regions. In two domains with extreme reward progressivity, where standard value-based methods struggle significantly, Spectral DQN is able to make much farther progress. Moreover, when evaluated on a set of six standard *Atari* games that do not overtly favour the approach, Spectral DQN remains more than competitive: While it underperforms one of the benchmarks in a single game, it comfortably surpasses the benchmarks in three games. These results demonstrate that the approach is not overfit to its target problem, and suggest that Spectral DQN may have advantages beyond addressing reward progressivity.

## Requirements

Blah...

## Installing
First install the Arcade Learning Environment (ALE) by following the instructions in README_ALE.md

conda env create --file environment.yml

## Running the main Atari experiments (game = ms_pacman)

## Settings

Change to CPU if necessary in python_agent.py

## Actually running
cd agent
source activate spectral_dqn
python2 ./python_agent.py ms_pacman

