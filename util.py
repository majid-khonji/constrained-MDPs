import gym
import numpy as np

def simulate_environment(env, num_steps, seed=None):
    # Initialize the state-action dictionary
    state_dict = {}

    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Reset the environment
    state = env.reset()

    # Loop through the simulation steps
    for step in range(num_steps):
        # Choose a random action
        action = env.action_space.sample()

        # Take the action and get the next state
        next_state, reward, done, info, _ = env.step(action)

        # Get the list of possible next states for this state-action pair
        possible_states = [next_state]

        # Try taking random actions to explore more possible next states
        for _ in range(10):
            random_action = env.action_space.sample()
            random_next_state, _, _, _, _ = env.step(random_action)
            if random_next_state not in possible_states:
                possible_states.append(random_next_state)

        # Add the state-action pair and its possible next states to the dictionary
        state_dict[(state, action)] = possible_states

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            # Reset the environment
            state = env.reset()

    # Return the state-action dictionary
    return state_dict


if __name__ == '__main__':
    # Get the environment
    env = gym.make('FrozenLake-v1')

    # Simulate the environment
    state_dict = simulate_environment(env, 1000)

    # Print the state dictionary
    print(state_dict)
