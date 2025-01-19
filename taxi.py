import gym
import numpy as np
import random
import time
# Ortamı oluşturun
env = gym.make("Taxi-v3", render_mode="ansi")

q_table=np.zeros([env.observation_space.n,env.action_space.n])

#Hyperparametres
episode=50
gamma=0.9
epsilon=0.1
alpha=0.1
reward_list=[]

while True:
    state=env.reset()[0]

    reward_count=0

    for i in range(episode):
        if(random.uniform(0,1)<epsilon):
            action = env.action_space.sample()
        else:
            action=np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)    
        #Old value
        old_value=q_table[state,action]
        #Next max
        next_max=np.max(q_table[next_state])
        #Q Function
        next_value=(1-alpha)*old_value+alpha*(reward+gamma*next_max)
        #Update next value
        q_table[state,action]=next_value
        #Update state
        state=next_state
        #Ödül listesi
        reward_count+=reward
        # Render çıktısını göster
        print(env.render())
        reward_list.append(reward_count)    
        # Eğer 'done' True ise döngüden çık
        if done:
            break

    if i%5 == 0:
        time.sleep(3)
        print(f"{i} kere eğitildim")
        reward_list.append(reward_count)
      
    
# %% visualize
fig ,axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(droputs_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()


env.close()
