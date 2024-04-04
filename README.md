# Flappy Bird AI Project

This project implements a reinforcement learning algorithm to train an AI agent to play the game Flappy Bird. The AI agent is trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Project Components

### 1. Flappy Bird Game

The game environment is implemented using the Pygame library. The game consists of a bird that can flap its wings to avoid colliding with pipes that appear on the screen. The goal of the game is to navigate the bird through the gaps between the pipes for as long as possible.

### 2. Reinforcement Learning Algorithms

Two reinforcement learning algorithms are implemented for training the AI agent:

- **NEAT Algorithm (NeuroEvolution of Augmenting Topologies)**: This algorithm evolves neural networks to control the bird's actions (flap or not flap) based on the current game state. NEAT is a genetic algorithm that evolves neural network architectures and weights over multiple generations.

- **Q-Learning**: An attempt has been made to implement Q-learning, but it is still a work in progress and has not been successfully implemented yet.

### 3. Training and Evaluation

- The `eval_gen` function evaluates the fitness of each AI agent (bird) in a generation by simulating the Flappy Bird game and calculating a fitness score based on the performance of each agent.
- The `run_neat` function trains AI agents using the NEAT algorithm for a specified number of generations.
- The `replay_genome` function allows replaying a trained AI agent (genome) without further training.

## How to Use

1. Ensure you have Python installed on your system along with the required dependencies listed in `requirements.txt`.

2. Run the script `main.py` with appropriate command-line arguments:

``` shell
python gameNEAT.py [-h] [--config CONFIG] [--color COLOR] [--nGens NGENS]
                   [--checkpoint CHECKPOINT] [--replay REPLAY]
```

- `--config`: Path to the NEAT configuration file (default: `config-ff.txt`).
- `--color`: Color of the bird (`random` or `red`) (default: `random`).
- `--nGens`: Number of generations to run training (default: `100`).
- `--checkpoint`: Checkpoint number to restore training from (default: `None`).
- `--replay`: Path to the genome file to replay (default: `None`).

## Work in Progress

- The Q-learning algorithm is still a work in progress and has not been successfully implemented yet. Further work is needed to debug and refine the Q-learning implementation.
- Additional optimizations and improvements may be made to enhance the performance and effectiveness of the NEAT algorithm.

Feel free to contribute to the project and experiment with different reinforcement learning algorithms to train the Flappy Bird AI agent!