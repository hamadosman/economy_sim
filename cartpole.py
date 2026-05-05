import gymnasium as gym 
import numpy as np 

env = gym.make("CartPole-v1")
state, info = env.reset()

print(state)  # 4 values — cart pos, cart vel, pole angle, pole vel
print(env.action_space.n)  # 2 — left or right
