import gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.1
# a measure for how important we find our future actions over current actions, future rewards over current rewards
DISCOUNT = 0.95
EPISODES = 25000

'''# in a real environment you won't know these values
print(env.observation_space.high)
print(env.observation_space.low)
# actions most of the time in a real world environment you will know
print(env.action_space.n)'''

# we are using the observation_space.high value so the discrete size works to any environment
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
#print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
#print(q_table.shape)
#print(q_table)

def get_discrete_state(state):
    discrete_state = (state[0] - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# env.reset() returns the inital state so we can pass it into get_discrete_state
# the first element returned is the observation which is the inital state
discrete_state = get_discrete_state(env.reset())
print('discrete_state', discrete_state)
# these are the init q values
print(q_table[discrete_state])
print(np.argmax(q_table[discrete_state]))

done = False

# 3 actions - push car left, action 1 do nothing, action 2 push car right
while not done:
    action = np.argmax(q_table[discrete_state])
    # in this case state is two values - position and velocity
    observation, reward, done, info, extra_value = env.step(action)
    new_discete_state = get_discrete_state(observation)
    env.render()

env.close()

