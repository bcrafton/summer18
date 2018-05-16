
import gym

env = gym.make('Taxi-v2')
env.reset()

for _ in range(1000):
    env.render()

    # action = env.action_space.sample()

    action = input("next action: ")
    action = int(action)

    next_state, reward, done, _ = env.step(action)

    print (next_state, reward)

    
