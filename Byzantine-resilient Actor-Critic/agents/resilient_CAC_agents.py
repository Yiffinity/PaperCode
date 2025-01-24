import numpy as np
import tensorflow as tf
from tensorflow import keras

class BRAC_agent():

    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95, H=0):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.H = H
        self.n_actions = self.actor.output_shape[1]
        self.fast_lr = fast_lr
        # self.optimizer_fast = keras.optimizers.SGD(learning_rate=fast_lr)
        self.optimizer_fast = keras.optimizers.Adam(learning_rate=fast_lr)
        self.mse = keras.losses.MeanSquaredError()
        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.critic_features = keras.Model(self.critic.inputs,self.critic.layers[-2].output)
        self.TR_features = keras.Model(self.TR.inputs,self.TR.layers[-2].output)

    def _resilient_aggregation(self,values_innodes):
        
        n_neighbors = values_innodes.shape[0]
        own_val = values_innodes[0]                  #get own value
        sorted_vals = tf.sort(values_innodes,axis=0)        #sort neighbors' values
        H_small = sorted_vals[self.H]
        H_large = sorted_vals[n_neighbors - self.H - 1]
        lower_bound = tf.math.minimum(H_small,own_val)
        upper_bound = tf.math.maximum(H_large,own_val)
        clipped_vals = tf.clip_by_value(sorted_vals,lower_bound,upper_bound)
        aggregated_values = tf.reduce_mean(clipped_vals,axis=0)

        return aggregated_values

    def critic_update_team(self,s,critic_agg):
        
        phi = self.critic_features(s)
        phi_norm = tf.math.reduce_sum(tf.math.square(phi),axis=1) + 1
        weights = 1 / (2 * self.fast_lr * phi_norm)
        self.critic_features.trainable = False
        self.critic.compile(optimizer=self.optimizer_fast,loss=self.mse)
        self.critic.train_on_batch(s,critic_agg,sample_weight=weights)

    def TR_update_team(self,sa,TR_agg):
        
        f = self.TR_features(sa)
        f_norm = tf.math.reduce_sum(tf.math.square(f),axis=1).numpy() + 1
        weights = 1 / (2 * self.fast_lr * f_norm)
        self.TR_features.trainable = False
        self.TR.compile(optimizer=self.optimizer_fast,loss=self.mse)
        self.TR.train_on_batch(sa,TR_agg,sample_weight=weights)

    def actor_update(self,s,ns,sa,a_local,pretrain=False):
        
        r_team = self.TR(sa)
        V = self.critic(s)
        nV = self.critic(ns)
        global_TD_error = tf.squeeze(r_team + self.gamma * nV - V).numpy()
        training_loss = self.actor.train_on_batch(s,a_local,sample_weight=global_TD_error)

        return training_loss

    def critic_update_local(self,s,ns,r_local):
       
        critic_weights_temp = self.critic.get_weights()
        nV = self.critic(ns)
        local_TD_target = r_local + self.gamma * nV
        self.critic_features.trainable = True
        self.critic.compile(optimizer=self.optimizer_fast,loss=self.mse)
        training_hist = self.critic.fit(s,local_TD_target,batch_size=s.shape[0],epochs=5,verbose=0)
        critic_weights = self.critic.get_weights()
        self.critic.set_weights(critic_weights_temp)

        return critic_weights, training_hist.history['loss'][0]

    def TR_update_local(self,sa,r_local):

        TR_weights_temp = self.TR.get_weights()
        self.TR_features.trainable = True
        self.TR.compile(optimizer=self.optimizer_fast,loss=self.mse)
        training_hist = self.TR.fit(sa,r_local,batch_size=sa.shape[0],epochs=5,verbose=0)
        TR_weights = self.TR.get_weights()
        self.TR.set_weights(TR_weights_temp)

        return TR_weights, training_hist.history['loss'][0]

    def resilient_consensus_critic_hidden(self,critic_weights_innodes):
    
        weights_agg = []
        for layer in zip(*critic_weights_innodes):
            weights = tf.convert_to_tensor(layer)
            weights_agg.append(self._resilient_aggregation(weights).numpy())
        self.critic_features.set_weights(weights_agg[:-2])

    def resilient_consensus_TR_hidden(self,TR_weights_innodes):

        weights_agg = []
        for layer in zip(*TR_weights_innodes):
            weights = tf.convert_to_tensor(layer)
            weights_agg.append(self._resilient_aggregation(weights).numpy())
        self.TR_features.set_weights(weights_agg[:-2])

    def resilient_consensus_critic(self,s,critic_weights_innodes):
        
        critic_weights_temp = self.critic.layers[-1].get_weights()
        critics = []
        for weights in critic_weights_innodes:
            self.critic.layers[-1].set_weights(weights[-2:])
            critics.append(self.critic(s))
        critics = tf.convert_to_tensor(critics)
        critic_agg = self._resilient_aggregation(critics)
        self.critic.layers[-1].set_weights(critic_weights_temp)

        return critic_agg

    def resilient_consensus_TR(self,sa,TR_weights_innodes):
        
        TR_weights_temp = self.TR.layers[-1].get_weights()
        TRs = []
        for weights in TR_weights_innodes:
            self.TR.layers[-1].set_weights(weights[-2:])
            TRs.append(self.TR(sa))
        TRs = tf.convert_to_tensor(TRs)
        TR_agg = self._resilient_aggregation(TRs)
        self.TR.layers[-1].set_weights(TR_weights_temp)

        return TR_agg

    def get_action(self,state,decay,mu=0.1):
       
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        mu = pow(0.995, decay) * mu
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights()]
