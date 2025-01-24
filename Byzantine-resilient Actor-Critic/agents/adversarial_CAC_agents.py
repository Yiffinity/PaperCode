import numpy as np
import tensorflow as tf
from tensorflow import keras


class Malicious_CAC_agent():
    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic_local_weights = self.critic.get_weights()
        self.TR.compile(optimizer=keras.optimizers.Adam(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())

    def actor_update(self,s,ns,r_local,a_local):
        weights_temp = self.critic.get_weights()
        self.critic.set_weights(self.critic_local_weights)
        V = self.critic(s)
        nV = self.critic(ns)
        TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()
        training_stats = self.actor.fit(s,a_local,sample_weight=TD_error,batch_size=200,epochs=1,verbose=0)
        self.critic.set_weights(weights_temp)

        return training_stats.history['loss'][0]

    def critic_update_compromised(self,s,ns,r_compromised):
        nV = self.critic(ns)
        TD_target_compromised = r_compromised + self.gamma * nV
        training_stats = self.critic.fit(s,TD_target_compromised,epochs=10,batch_size=32,verbose=0)

        return self.critic.get_weights(), training_stats.history["loss"][0]

    def critic_update_local(self,s,ns,r_local):
        weights_temp = self.critic.get_weights()
        self.critic.set_weights(self.critic_local_weights)
        nV = self.critic(ns)
        local_TD_target = r_local + self.gamma * nV
        self.critic.fit(s,local_TD_target,epochs=10,batch_size=32,verbose=0)
        self.critic_local_weights = self.critic.get_weights()
        self.critic.set_weights(weights_temp)

    def TR_update_compromised(self,sa,r_compromised):
        training_stats = self.TR.fit(sa,r_compromised,epochs=10,batch_size=32,verbose=0)

        return self.TR.get_weights(), training_stats.history["loss"][0]

    def get_action(self,state,decay,mu=0.1):
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        mu = pow(0.995, decay) * mu
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights(), self.critic_local_weights]

class Greedy_CAC_agent():

    def __init__(self,actor,critic,team_reward,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.TR.compile(optimizer=keras.optimizers.Adam(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())

    def actor_update(self,s,ns,r_local,a_local):


        V = self.critic(s)
        nV = self.critic(ns)
        TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()
        training_stats = self.actor.fit(s,a_local,sample_weight=TD_error,batch_size=200,epochs=1,verbose=0)

        return training_stats.history['loss'][0]

    def critic_update_local(self,s,ns,r_local):

        nV = self.critic(ns)
        local_TD_target = r_local + self.gamma * nV
        training_stats = self.critic.fit(s,local_TD_target,epochs=10,batch_size=32,verbose=0)

        return self.critic.get_weights(), training_stats.history['loss'][0]

    def TR_update_local(self,sa,r_local):

        training_stats = self.TR.fit(sa,r_local,epochs=10,batch_size=32,verbose=0)

        return self.TR.get_weights(), training_stats.history['loss'][0]

    def get_action(self,state,decay,mu=0.1):

        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(state).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        mu = pow(0.995, decay) * mu
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights()]
