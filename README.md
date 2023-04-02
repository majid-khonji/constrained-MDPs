

# Toolkit for (Chance) Constrained MDP Algorithms
## Graph-based Constrained MDP and RL Environment Visualization

This repository contains Python code to visualize the environment of a Reinforcement Learning (RL) problem as a graph. The code creates a directed graph representing the state transitions and actions based on the given environment. The purpose of this visualization is to help understand the structure of the environment and improve the RL algorithms.

### Dependencies

- gym
- networkx
- matplotlib
- numpy
- gym_mazeS

### Usage

1. Create the desired environment using OpenAI's `gym` library.
2. Set the `horizon` variable to the desired depth of the graph.
3. Call the `build_graph_sim` function with the environment and initial state to create the graph. Alternatively, you can use the `build_graph` function for a more deterministic approach.
4. Use the `visualize_graph` function to display the graph.

#### Example

```python
import gym
import networkx as nx
from visualization import build_graph_sim, visualize_graph

env = gym.make('FrozenLake-v1')
horizon = 100
initial_state = env.reset()

graph = nx.DiGraph()
build_graph_sim(env, initial_state, graph, horizon)

visualize_graph(graph)
```
### Features

- Builds a directed graph representing the environment's state transitions and actions.
- Visualizes the graph using `networkx` and `matplotlib`.
- Groups nodes with similar descendant nodes.
- Calculates the number of nodes and edges in the graph.

### Functions

- `build_graph`: Builds a deterministic graph based on the environment's transition probabilities.
- `build_graph_sim`: Builds a graph using simulations, exploring the environment by performing random actions.
- `visualize_graph`: Draws the graph with nodes labeled by state numbers and edges labeled by actions and probabilities.
- `get_node_groups`: Groups nodes by depth based on their descendant nodes.

### Limitations

- The graph can become very large for complex environments with many states and actions.
- The visualization may not be suitable for very deep graphs with many nodes at each depth.

### Tested Environments

- `FrozenLake-v1`
- `CliffWalking-v0`
- `Taxi-v3`
- `maze-v0`
- `maze-sample-100x100-v0`
- `maze-random-100x100-v0`
- `maze-sample-10x10-v0`
- `maze-random-30x30-plus-v0`

Feel free to experiment with other environments from the `gym` library, but keep in mind that the visualization may not be suitable for all types of environments.

