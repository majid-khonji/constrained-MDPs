import gym
import networkx as nx
import matplotlib.pyplot as plt

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

def get_descendants(graph, node, depth, horizon, memo):
    if depth >= horizon:
        return {node}

    if node in memo:
        return memo[node]

    descendants = set()
    for successor in graph.successors(node):
        descendants |= get_descendants(graph, successor, depth + 1, horizon, memo)

    memo[node] = descendants
    return descendants

def get_node_groups(graph, horizon):
    node_groups = {depth: [] for depth in range(horizon + 1)}

    memo = {}
    for node in graph.nodes():
        depth = node[1]
        node_descendants = get_descendants(graph, node, depth, horizon, memo)

        if depth == 0:
            if len(node_groups[depth]) == 0:
                node_groups[depth].append({node})
            continue

        added_to_existing_group = False
        for group in node_groups[depth]:
            group_descendants = set.union(*(get_descendants(graph, n, n[1], horizon, memo) for n in group))
            if not group_descendants.isdisjoint(node_descendants):
                group.add(node)
                added_to_existing_group = True
                break

        if not added_to_existing_group:
            node_groups[depth].append({node})

    return node_groups



# env = gym.make('FrozenLake-v1') # 16 states, 4 actions
env = gym.make('CliffWalking-v0') # 48 states, 4 actions
# env = gym.make("Taxi-v3") # 500 states, 6 actions
graph = nx.DiGraph()
visited = set()
horizon = 20
initial_state = env.reset()
build_graph(env, initial_state[0], graph, visited, 0, horizon)
node_groups  = get_node_groups(graph, horizon)
for depth, groups in node_groups.items():
    group_sizes = [len(group) for group in groups]
    print(f"Depth {depth}: {group_sizes}")


# print(node_groups)
# visualize_graph(graph)
