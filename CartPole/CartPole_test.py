import tensorflow as tf
import numpy as np
import gym

from cartpoleagent import CartPoleAgent

if __name__ == "__main__":

	env = gym.make("CartPole-v0")
	env._max_episode_steps = 1001

	CPagent = CartPoleAgent(env)
	CPagent.load_model("./model/CartPole")

	obs = env.reset()
	while True:
	    env.render()
	        
	    action = CPagent.action_select(obs, 0)        
	    next_obs, reward, done, info = env.step(action) 
	    obs = next_obs
	    if done:
	        break    
	env.close()