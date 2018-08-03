
import gym

env = gym.make('CartPole-v0')
state = env.reset()

done = False

action = 0
while not done:
    
    # action = env.action_space.sample()
    action = not action
    
    '''
    if state[3] < 0:
        action = 1
    else:
        action = 0
    '''
        
    next_state, reward, done, _ = env.step(action)
    print (state, action, next_state, reward)
