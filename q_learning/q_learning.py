import gym
import numpy as np
env = gym.make("MountainCar-v0", render_mode="human")
env.reset()

# in a real environment you won't know these values
print(env.observation_space.high)
print(env.observation_space.low)
# actions most of the time in a real world environment you will know
print(env.action_space.n)

# we are using the observation_space.high value so the discrete size works to any environment
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)
print(q_table)
done = False

# 3 actions - push car left, action 1 do nothing, action 2 push car right
while not done:
    action = 2
    # in this case state is two values - position and velocity
    observation, reward, done, info, extra_value = env.step(action)
    print(reward, observation)     
    env.render()

env.close()

