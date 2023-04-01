import gym
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
import gym_maze


def build_graph(env, state, graph, visited, depth, horizon):
    if depth >= horizon or (state, depth) in visited:
        return

    visited.add((state, depth))

    for action in range(env.action_space.n):
        for prob, next_state, _, done in env.P[state][action]:
            if prob > 0:
                graph.add_edge((state, depth), (next_state, depth + 1), action=action, prob=prob)
                if not done:
                    build_graph(env, next_state, graph, visited, depth + 1, horizon)


""" The code below has an issue with setting the initial state for some environment. For example, the following code will not work for the FrozenLake-v0 environment"""
# def build_graph_sim(env, state, graph, visited, depth, horizon, num_simulations=100, state_var_name='s'):
#     if isinstance(state, (list, np.ndarray)):  # Check if the state is a list or a NumPy array
#         state = tuple(state)  # Convert the state to a tuple

#     if depth >= horizon or (state, depth) in visited:
#         return

#     visited.add((state, depth))

#     for action in range(env.action_space.n):
#         next_state_counts = defaultdict(int)
#         for _ in range(num_simulations):
#             env.reset()                            # Reset the environment
#             setattr(env.unwrapped, state_var_name, state)  # Set the current state using the state_var_name
#             next_state, _, done, _, info = env.step(action)
#             next_state_bkup = next_state
#             if isinstance(next_state, (list, np.ndarray)):  # Check if the next state is a list or a NumPy array
#                 next_state = tuple(next_state)  # Convert the next state to a tuple
            
#             next_state_counts[next_state] += 1

#         for next_state, count in next_state_counts.items():
#             prob = count / num_simulations
#             graph.add_edge((state, depth), (next_state, depth + 1), action=action, prob=prob)
#             build_graph_sim(env, next_state_bkup, graph, visited, depth + 1, horizon)


def build_graph_sim(env, init_state, graph, horizon, num_simulations=10000):
    if isinstance(init_state, (list, np.ndarray)):  # Check if the state is a list or a NumPy array
        init_state = tuple(init_state)  # Convert the state to a tuple


    next_state_counts = defaultdict(int)
    for _ in range(num_simulations):
        # env.seed()  # Ensure different outcomes for each simulation
        env.reset()  # Reset the environment
        current_state = init_state
        current_depth = 0
        done = False

        # Run the simulation until the end or reaching the desired depth
        while not done and current_depth < horizon:
            action = env.action_space.sample()  # Sample a random action
            next_state, _, done,_, _ = env.step(action)
            if isinstance(next_state, (list, np.ndarray)):  # Check if the next state is a list or a NumPy array
                next_state = tuple(next_state)  # Convert the next state to a tuple
            graph.add_edge((current_state, current_depth), (next_state, current_depth + 1), action=action, prob=0)
            current_depth += 1
            current_state = next_state





def visualize_graph(graph):
    # Create positions for the nodes based on depth
    depths = {node: node[1] for node in graph.nodes()}
    level_count = {depth: 0 for depth in range(horizon + 1)}
    pos = {}

    for node in graph.nodes():
        depth = depths[node]
        pos[node] = (depth, level_count[depth])
        level_count[depth] += 1

    # Calculate the number of nodes at each level
    num_nodes_at_level = [0] * (horizon + 1)
    for depth in depths.values():
        num_nodes_at_level[depth] += 1

    # Distribute nodes evenly within each level
    for node in pos:
        depth = depths[node]
        pos[node] = (depth, pos[node][1] / (num_nodes_at_level[depth] - 1) if num_nodes_at_level[depth] > 1 else 0.5)

    # Relabel nodes to display only state number
    node_labels = {node: node[0] for node in graph.nodes()}

    # Visualize the graph
    nx.draw(graph, pos, with_labels=False, node_color='lightblue', font_weight='bold', node_size=500)
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    edge_labels = {(u, v): f"A: {d['action']} P: {d['prob']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

    plt.show()

def _get_descendants(graph, node, depth, horizon, memo):
    if depth >= horizon:
        return {node}

    if node in memo:
        return memo[node]

    descendants = set()
    for successor in graph.successors(node):
        descendants |= _get_descendants(graph, successor, depth + 1, horizon, memo)

    memo[node] = descendants
    return descendants

def get_node_groups(graph, horizon):
    node_groups = {depth: [] for depth in range(horizon + 1)}

    memo = {}
    for node in graph.nodes():
        depth = node[1]
        node_descendants = _get_descendants(graph, node, depth, horizon, memo)

        if depth == 0:
            if len(node_groups[depth]) == 0:
                node_groups[depth].append({node})
            continue

        added_to_existing_group = False
        for group in node_groups[depth]:
            group_descendants = set.union(*(_get_descendants(graph, n, n[1], horizon, memo) for n in group))
            if not group_descendants.isdisjoint(node_descendants):
                group.add(node)
                added_to_existing_group = True
                break

        if not added_to_existing_group:
            node_groups[depth].append({node})

    return node_groups



env = gym.make('FrozenLake-v1') # 16 states, 4 actions
# env = gym.make('CliffWalking-v0') # 48 states, 4 actions
# env = gym.make("Taxi-v3") # 500 states, 6 actions
# env = gym.make('maze-v0')
# env = gym.make('maze-sample-100x100-v0')
# env = gym.make('maze-random-100x100-v0')
# env = gym.make('maze-sample-10x10-v0')
env = gym.make('maze-random-30x30-plus-v0')




print("Action space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

horizon = 10
initial_state = env.reset()

graph = nx.DiGraph()
visited = set()
# build_graph(env, initial_state[0], graph, visited, 0, horizon)
build_graph_sim(env, initial_state[0], graph, horizon, num_simulations=3000)

# count the number of edges
print(f"Number of edges: {len(graph.edges())}")
print(f"Number of nodes: {len(graph.nodes())}")

node_groups  = get_node_groups(graph, horizon)
for depth, groups in node_groups.items():
    group_sizes = [len(group) for group in groups]
    print(f"Depth {depth}: {group_sizes}")

    # print(f"Depth {depth}: {group_sizes}   {node_groups[depth]}")


# visualize_graph(graph)
