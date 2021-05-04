import tensorflow as tf
import numpy as np

class DQNagent():
    def __init__(self, env, alpha = 1, gamma = 0.99, replay_memory_max_size = 50000,name = "DQN"):
        ### Main DQN class
        ### make your own DQN model for each ai gym envrionment
        
        self.name = name
        self.env = env
        self.obs_s = env.observation_space.shape
        self.obs_n = np.prod(self.obs_s)
        self.act_n = env.action_space.n
        
        self.model = self.generate_model()
        self.target_model = self.generate_model()
        
        self.replay_memory = np.empty(0).reshape(0,self.obs_n*2+3)
        self.replay_memory_index = 0
        self.replay_memory_max_size = replay_memory_max_size
                
        self.alpha = alpha
        self.gamma = gamma
        
    def generate_model(self):
        inputs = tf.keras.Input(shape = self.obs_s)

        '''make your own model here'''
        # ==========================================================
        # ==========================================================
        '''make your own model here'''

        outputs = tf.keras.layers.Dense(self.act_n, activation = None)(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.1),
                     loss = tf.losses.MeanSquaredError()
                     ) 
        
        return model
    
    ### model summary
    def summary(self):
        return self.model.summary()
    
    ### select action with epsilon greedy 
    def action_select(self, obs, epsilon = 0):
        
        obs = self.make_obs_to_input_shape(obs)
        action = self.env.action_space.sample()
        
        if np.random.random() > epsilon:
            action = np.argmax(self.model(obs)) 
            
        return action                
        
    ### train model with data in replay buffer    
    def train(self, train_size = 128,batch_size = 128, epochs = 50):     
        ### make clone of replay memory and shuffle
        ### extract the train size of shuffled memory 
        ### change the shape of input side of replay memory as shape of input
        ### train the model with shuffled and batched data        
        
        replay_memory_clone = self.replay_memory.copy()        
        np.random.shuffle(replay_memory_clone)        
        
        obs = replay_memory_clone[:train_size,:self.obs_n]
        next_obs = replay_memory_clone[:train_size,self.obs_n:-3]
        action = replay_memory_clone[:train_size,-3]
        reward = replay_memory_clone[:train_size,-2]
        done = replay_memory_clone[:train_size,-1]        
        
        train_size = min(len(self.replay_memory),train_size)
        
        ### x_train is observation(state)
        ### y_train is updated Q-value
        x_train = self.make_obs_to_input_shape(obs)
        
        y_train = self.model(obs)
        y_train = y_train.numpy()

        ### set y_train if done :reward  else: reward + next state's maximum
        y_train[np.arange(train_size),action.astype(int)] = \
            (-self.alpha) * y_train[np.arange(train_size),action.astype(int)]+\
            self.alpha * (reward + self.gamma * ~done.astype(bool) * np.max(self.target_model(next_obs), axis = 1))        

        return self.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 0)
    
    
    ### save the replay in replay buffer, To train model with uncorrelated Values
    def save_replay(self, obs, action, next_obs, reward, done):
                
        o = np.reshape(obs, self.obs_n)
        no = np.reshape(next_obs, self.obs_n)
                
        stack = np.concatenate([o,no,[action],[reward],[done]])
        
        if len(self.replay_memory) <= self.replay_memory_max_size:
            self.replay_memory = np.vstack([self.replay_memory, stack])
        else:
            self.replay_memory[self.replay_memory_index] = stack
            self.replay_memory_index = (self.replay_memory_index+1) % self.replay_memory_max_size
                
        return
    
    ### update target network
    def target_update(self):
        for a, b in zip(self.model.variables, self.target_model.variables):
        	b.assign(a)
        return
    
    ### save the trained model
    def save_model(self, file_path = None):
        if file_path is None:
            file_path = "./model/"+self.name
        tf.keras.models.save_model(self.model, file_path)
        return 
    
    ### load the trained model
    def load_model(self, file_path = None):
        if file_path is None:
            file_path = "./model/"+self.name
        self.model = tf.keras.models.load_model(file_path)
        return 
    
    ### make observation(state)'s shape to input shape
    def make_obs_to_input_shape(self,obs):
        obs_shape = [-1]
        for i in self.obs_s:
            obs_shape.append(i)
        obs = np.reshape(obs, obs_shape)
        return obs    