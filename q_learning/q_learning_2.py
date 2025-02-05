import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

# a measure for how important we find our future actions over current actions, future rewards over current rewards
LEARNING_RATE = 0.1
# a measure for how important we find our future actions over current actions, future rewards over current rewards
DISCOUNT = 0.95
EPISODES = 1200
SHOW_EVERY = 200

# we are using the observation_space.high value so the discrete size works to any environment
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int32))

# the first part is training and setting q values 
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    # env.reset() returns the inital state so we can pass it into get_discrete_state
    # the first element returned is the observation which is the inital state
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        # in this case state is two values - position and velocity
        new_state, reward, truncated, terminated, _ = env.step(action)
        done = truncated or terminated
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
        # [0] is position 
        elif new_state[0] >= env.goal_position:
            print(f"We reached to the goal! Episode: {episode}")
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

env.close()

# now that we have set q values according to the above we simply rerun using the
# the q_table as it now contains the highest rewards in sequential order to our goal
# %%
env = gym.make("MountainCar-v0", render_mode='human')
env.reset()

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
discrete_state = get_discrete_state(env.reset()[0])

done = False
while not done:
    discrete_state = get_discrete_state(env.state)
    action = np.argmax(q_table[discrete_state])
    new_state, _, done, _, _ = env.step(action)

env.close()

# %%
env.close()
