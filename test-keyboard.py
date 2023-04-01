import  gym
import gym_maze
import pygame
import keyboard
import time

# import gym_classics 
# import warnings
# from gym import logger
# warnings.simplefilter("always")
# logger.set_level(logger.WARN)
# warnings.filterwarnings("ignore", module="gym")
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)



################################################
# Test the environment
################################################

h = 3 # planning horizon

# Classic Gym
# env = gym.make('ClassicGridworld-v0')
# env = gym.make('FourRooms-v0')
# env = gym.make('JacksCarRental-v0')
# env = gym.make('CliffWalk-v0')

# Original gym 

# env = gym.make('FrozenLake-v1') # 16 states, 4 actions
# env = gym.make('CliffWalking-v0') # 48 states, 4 actions
# env = gym.make("Taxi-v3", render_mode="human") # 500 states, 6 actions
# env = gym.make("Blackjack-v1") # 32 states, 2 actions
# env = gym.make("maze-random-10x10-plus-v0") 
# env = gym.make('CartPole-v1', render_mode="human") # 500 states, 6 actions
# env = gym.make('maze-v0') 
# env = gym.make('maze-sample-100x100-v0')
env = gym.make('maze-sample-10x10-v0')


print("Action space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))
print("Reward range: {}".format(env.reward_range))




state = env.reset()
# Render the environment
env.render()
print("Initial state: {}".format(state))

done = False
for episode in range(3):
        env.reset()
        action = -1
        while not done:
    
            if keyboard.is_pressed('w'):
                  print("w")
                  action = 0
            elif keyboard.is_pressed('s'):
                  print("s")
                  action = 1
            elif keyboard.is_pressed('a'):
                  print("a")
                  action = 3
            elif keyboard.is_pressed('d'):
                  print("d")
                  action = 2
            time.sleep(.1)
            if action != -1:
                next_state, reward, done,_, info = env.step(action)
               
            
                # print(_)
                transition = (state, action, next_state, reward, done)
                print("Transition: {}".format(transition))
                action = -1
            env.render()
        done = False



env.close()