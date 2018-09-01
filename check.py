import gym

env = gym.make('LunarLander-v2')


pos = env.reset()

print(pos)


for _ in range(1000):

    env.render()

    action = env.action_space.sample()


    print('Action : ',action)


    observation,reward,done,info = env.step(action)

    print('Observation : ',observation)
    print('Reward :',reward)

    if done:
        env.reset()