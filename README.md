# Spectral DQN

[Adapting to Reward Progressivity via Spectral Reinforcement Learning](https://openreview.net/forum?id=dyjPVUc2KB), ICLR'21.

## Abstract

We consider reinforcement learning tasks with *progressive rewards*; that is, tasks where the rewards tend to increase in magnitude over time. We hypothesise that this property may be problematic for value-based deep reinforcement learning agents, particularly if the agent must first succeed in relatively unrewarding regions of the task in order to reach more rewarding regions. To address this issue, we propose *Spectral DQN*, which decomposes the reward into frequencies such that the high frequencies only activate when large rewards are found. This allows the training loss to be balanced so that it gives more even weighting across small and large reward regions. In two domains with extreme reward progressivity, where standard value-based methods struggle significantly, Spectral DQN is able to make much farther progress. Moreover, when evaluated on a set of six standard *Atari* games that do not overtly favour the approach, Spectral DQN remains more than competitive: While it underperforms one of the benchmarks in a single game, it comfortably surpasses the benchmarks in three games. These results demonstrate that the approach is not overfit to its target problem, and suggest that Spectral DQN may have advantages beyond addressing reward progressivity.

## Requirements

The installation instructions below assume that you have Anaconda installed. You will also need to source some Atari ROMs and place them in the /roms folder.

## Installing
First, install the Arcade Learning Environment (ALE) by following the instructions in README_ALE.md

Next, create a Conda environment with the necessary packages installed by running:
```console
conda env create --file environment.yml
```

## Running

```console
source activate spectral_dqn
cd agent
python2 ./python_agent.py game_title
```

The agent's configuration settings are stored in /agent/python_agent.py.

To see the agent playing the game, set USE_SDL = False then select the SDL window that appears and press 'd' to toggle the display.

To see performance and debugging graphs, set agent_params\["show_graphs"\] = True.

If you don't have a CUDA enabled GPU, you can set agent_params\["gpu"\] = -1 to train on the CPU.
