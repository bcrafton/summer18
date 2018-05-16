
import gym

env = gym.make('FrozenLake-v0')
env.reset()

for _ in range(1000):
    env.render()
    action = input("next action: ")
    action = int(action)
    env.step(action)

    
