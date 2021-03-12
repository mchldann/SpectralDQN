# Requirements

Blah...

# Installing
First install the Arcade Learning Environment (ALE) by following the instructions in README_ALE.md

conda env create --file environment.yml

# Running the main Atari experiments (game = ms_pacman)

## Settings

Change to CPU if necessary in python_agent.py

## Actually running
cd agent
source activate spectral_dqn
python2 ./python_agent.py ms_pacman

