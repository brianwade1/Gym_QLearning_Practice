"""
Q-learning with Mountain Car gym environment

Created by: Brian Wade
"""
# Standard imports
import os
import csv

# Conda imports
import gym
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

# Constants
ENV_NAME = "MountainCar-v0"
goal_position = 0.5

# Training parameters
BINS = 40  # Number of bins to discretize the x-axis observation space
LEARNING_RATE = 0.1 # alpha 
DISCOUNT = 0.99 # gamma
EPISODES = 30_000 # total number of episodes to train
USE_REWARD_SHAPING = False

# Exploration settings
EPSILON_START = 1  # starting value - will decay later
START_EPSILON_DECAYING = 500
END_EPSILON_DECAYING = int(0.95*EPISODES)

# Folders for results
IMAGE_FOLDER = 'Images'
RESULTS_FOLDER = 'Results'

#Show cartoons during training
SHOW_RENDER = False # 1 = show cartoon, 0 = no show cartoon
SHOW_EVERY = 1000

# For stats
PROGRESS_SHOW = 500  # Show on screen
STATS_EVERY = 100  #save for plotting


##########################################

# Initialize environment
env = gym.make(ENV_NAME)

#Actions space: 0 = left, 1 = stay, 2 = right
print('**** Action Space Size ****')
print(env.action_space.n)

#How large is the observation space? - (x-pos, vel)  
print('**** Observation Space Size ****')     
print(env.observation_space.high)
print(env.observation_space.low)

# Mid point along x-axis - used in reward shaping
mid_point = env.observation_space.low[0] + (env.observation_space.high[0] - env.observation_space.low[0]) / 2

#Bucket observations space into 20 bins.
DISCRETE_OS_SIZE = [BINS] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print('**** Observation Discretization Window Size ****')    
print(discrete_os_win_size)

#make 20x20x3 q-table.  20x20 observation space and 3 actions (left right no input)
#initialize with [-2,0] random values since not getting flag has a reward of 
#-1 and getting flag has a reward of 0.
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(int))

def get_reward(state):
    if state[0] >= goal_position:
        #print("Car has reached the goal")
        return 0
    if state[0] > mid_point:
        return (-1 / (mid_point - goal_position)) * (state[0] - goal_position)
    return -1

#Train the agent
print('**** Start the training! ****')  
epsilon_decay_value = EPSILON_START/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
epsilon = EPSILON_START
ep_rewards = []
episode_history_easy_read = []
episode_reward_history = {'episode': [], 'avg': [], 'max': [], 'min': []}
for episode in range(EPISODES):

    # Initialize Episode
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    if (episode % SHOW_EVERY == 0 and SHOW_RENDER):
        render = True
        print(episode)
    else:
        render = False
    
    # Do Episode
    done = False
    while not done:
        
        # Epsilon greedy policy
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
        
        # Advance environment with chosen action
        new_state, reward_out, done, _ = env.step(action)
        
        # Modify reward if using reward shaping
        if USE_REWARD_SHAPING:
            reward = get_reward(new_state)
        else:
            reward = reward_out
        episode_reward += reward
        
        # Transform continuous state to discrete state
        new_discrete_state = get_discrete_state(new_state)
        
        # Render if watching playback
        if render:
            print(reward, new_state)
            env.render()
        
        # If simulation did not end yet after last step - update Q table
        if not done:
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
    
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
    
            # Find new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
            
        else:
            q_table[discrete_state + (action,)] = reward
            #print(f"Cart made it to flag on episode: {episode}")
                
        discrete_state = new_discrete_state
        
    # Decay epsilon for epsilon greedy policy - transition from exploration to exploitation
    if (END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING) and (epsilon > 0.0):
        epsilon -= epsilon_decay_value
        if epsilon < 0.0:
            epsilon = 0.0

    # Record training history    
    ep_rewards.append(episode_reward)
    if (episode % STATS_EVERY == 0) and (episode != 0):
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        episode_reward_history['episode'].append(episode)
        episode_reward_history['avg'].append(average_reward)
        episode_reward_history['max'].append(max(ep_rewards[-STATS_EVERY:]))
        episode_reward_history['min'].append(min(ep_rewards[-STATS_EVERY:]))
        
    # Print stats to console
    if (episode % PROGRESS_SHOW == 0) and (episode != 0):
        min_reward = episode_reward_history['min'][-1]
        max_reward = episode_reward_history['max'][-1]
        summary_line = f'Episode: {episode:>5d}, ' \
                            f'min reward: {min_reward:>4.1f}, ' \
                            f'average reward: {average_reward:>4.1f}, ' \
                            f'max reward: {max_reward:>4.1f}, ' \
                            f'current epsilon: {epsilon:>1.2f}'
        episode_history_easy_read.append(summary_line)
        print(summary_line)

# Close environment when training is complete        
env.close()

# Run name for saving files
run_name = ENV_NAME + '-USE_REWARD_SHAPING-' + str(USE_REWARD_SHAPING) 

# Save Q-Table
results_file_path = os.path.join(RESULTS_FOLDER, 'Qtable-' + run_name + '.csv')
with open(results_file_path, 'w') as f:
    w = csv.writer(f)
    w.writerows(q_table)

# Save summary of results history
short_history_file_path = os.path.join(RESULTS_FOLDER, 'HistorySummary-' + run_name + '.txt')
with open(short_history_file_path, 'w') as f:
    for row in episode_history_easy_read:
        f.write(row)
        f.write('\n')

# Save all results history
long_history_file_path = os.path.join(RESULTS_FOLDER, 'LongHistory-' + run_name + '.csv')
with open(long_history_file_path, 'w') as f:
    w = csv.DictWriter(f, episode_reward_history.keys())
    w.writeheader()
    w.writerow(episode_reward_history)

# Plot training history
plt.plot(episode_reward_history['episode'], episode_reward_history['avg'], label="average rewards")
plt.plot(episode_reward_history['episode'], episode_reward_history['max'], label="max rewards")
plt.plot(episode_reward_history['episode'], episode_reward_history['min'], label="min rewards")
plt.legend(loc=2)
plt.grid(True)
plt.savefig(os.path.join(IMAGE_FOLDER, 'TrainingHistory-' + run_name + '.png'))
#plt.show()

# Record gif of trained agent in environment
# Original code from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
gif_filename = 'Animation-' + run_name + '.gif'
def save_frames_as_gif(frames, path=IMAGE_FOLDER, filename=gif_filename):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1)
    anim.save(os.path.join(path, filename), fps=240)

#Run the env using only the trained policy using greedy actions
print('**** Start the testing! ****')  
frames = []
discrete_state = get_discrete_state(env.reset())
done = False
while not done:
    #Render to frames buffer
    frames.append(env.render(mode="rgb_array"))
    action = np.argmax(q_table[discrete_state])
    new_state, _, done, _ = env.step(action)
    discrete_state = get_discrete_state(new_state)

env.close()
print('**** Saving animation! ****')  
save_frames_as_gif(frames)
