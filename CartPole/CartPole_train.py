import tensorflow as tf
import numpy as np
import gym

from cartpoleagent import CartPoleAgent


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    env._max_episode_steps = 1001

    CPagent = CartPoleAgent(env, gamma = 0.9)
    CPagent.summary()

    init_eps = 0.9
    end_eps = 0.001
    eps_decaying = 50

    for episode in range(500):
        obs = env.reset()
        eps = init_eps - (init_eps-end_eps) / max(eps_decaying/(episode+1), 1)
        
        for t in range(1001):
            #env.render()
            
            action = CPagent.action_select(obs, eps)
            next_obs, reward, done, info = env.step(action)
            if done:
                reward = -1
            CPagent.save_replay(obs, action, next_obs, reward, done)
            
            obs = next_obs        
            
            if done: 
                print("Episode {} finished after {} timesteps".format(episode,t+1))
                for i in range(50):
                    CPagent.train(train_size = 64, epochs = 1)
                break
                
            if t == 1000:
                CPagent.save_model()        


        if episode %10 ==0:
            CPagent.target_update()
            print("Target Updated")        
    
    env.close()