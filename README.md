# Autonomous Taxi RL Agent 🚕🤖

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange.svg)](https://gymnasium.farama.org/)


An implementation of a Reinforcement Learning agent that learns to optimally navigate a city grid, pick up passengers, and deliver them to their destinations using **Tabular Q-Learning**.

Built using the [Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) `Taxi-v3` environment, this project demonstrates the fundamentals of sequential decision-making, the Explore vs. Exploit dilemma, and the Bellman Equation without relying on deep neural networks.

## 📌 Project Overview

In the `Taxi-v3` environment, there are 4 designated locations in the grid world indicated by Red, Green, Yellow, and Blue. 
* **The Goal:** The agent must drive to the passenger's starting location, pick them up, drive to the destination, and drop them off in the fewest steps possible.
* **The Challenge:** The agent starts with zero knowledge of the map. It must explore the environment randomly, receive rewards (or penalties), and update its internal map of "good" and "bad" actions (the Q-Table).

## ✨ Features
* **Q-Learning Algorithm:** Built from scratch using NumPy.
* **Epsilon Decay:** Gradually shifts the agent from random exploration to intelligent exploitation over 10,000 training episodes.
* **CLI Visualization:** Renders the trained agent's performance directly in the terminal using ANSI escape codes for smooth, dependency-light playback.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `gymnasium`, `numpy`, `os`, `time`, `random`



This project utilizes a **Q-Table**, a matrix of 500 rows (possible states) and 6 columns (possible actions). 

1. **State Space (500):** A combination of 25 taxi positions, 5 possible passenger locations (including in the taxi), and 4 destination locations.
2. **Action Space (6):** Move South, North, East, West, Pickup, Dropoff.
3. **The Bellman Equation:** After every step, the agent updates its Q-Table using the formula:
   ```python
   new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
   ```
   * `alpha` (Learning Rate): How much new information overrides old information.
   * `gamma` (Discount Factor): How much the agent values long-term rewards vs. immediate rewards.

## 📈 Future Improvements
* Export and save the trained Q-Table as a `.npy` file so the agent doesn't need to retrain on every run.
* Implement a hyperparameter tuning loop to find the optimal combinations of `alpha`, `gamma`, and `epsilon_decay`.
* Transition the algorithm from Tabular Q-Learning to a Deep Q-Network (DQN) using PyTorch for environments with continuous state spaces.
