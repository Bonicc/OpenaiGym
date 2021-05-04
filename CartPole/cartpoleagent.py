import tensorflow as tf
import numpy as np

from DQN import DQNagent

class CartPoleAgent(DQNagent):
    def __init__(self, env, alpha = 1, gamma = 0.999, replay_memory_max_size = 2000, name = "CartPole"):
        super().__init__(env, alpha, gamma, replay_memory_max_size, name)
        pass
    
    def generate_model(self):        
        ### make env's model        
        inputs = tf.keras.Input(shape = self.obs_s)
        
        '''make your own model here'''
        # ==========================================================
        x = tf.keras.layers.Dense(10, activation=tf.nn.tanh)(inputs)
        x = tf.keras.layers.Dense(10, activation=tf.nn.tanh)(x)
        # ==========================================================
        '''make your own model here'''

        outputs = tf.keras.layers.Dense(self.act_n, activation = None)(x)            
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.01),
                     loss = tf.losses.MeanSquaredError()
                     )        

        return model

if __name__ == "__main__":
	pass