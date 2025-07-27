import numpy as np
import tensorflow as tf
from tensorflow import keras

class Faulty_CAC_agent():
    '''
    FAULTY CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the faulty consensus actor-critic (FCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The FCAC agent employs neural networks to approximate the actor, critic, and team-average reward.
    It updates its actor network but does not update the critic and team-average reward parameters. Furthermore, it transmits
    fixed parameter values to the other agents in the network. The FCAC agent does not apply consensus updates. It samples actions
    from the policy approximated by the actor network.

    ARGUMENTS: NN models for actor and critic, and team-average reward
               slow learning rate (for the actor network)
               discount factor gamma
    '''
    def __init__(self,actor,critic,team_reward,args,critic_attention_layer,agent_index,slow_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.args = args
        self.critic_attention_layer=critic_attention_layer
        self.agent_index=agent_index
        self.n_actions=self.actor.output_shape[1]

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        
        self.encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            keras.layers.Dense(self.args['hidden_dim']),            
            keras.layers.LeakyReLU()
            ])
        self.actor_encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            tf.keras.layers.Dense(self.args['hidden_dim']),            
            tf.keras.layers.LeakyReLU()
            ])
    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, local rewards and actions
        RETURNS: training loss
        '''
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:]  # [B, state_dim], 当前智能体的下一个状态
        obs_encoding=self.encoder(current_obs)
        next_obs_encoding=self.encoder(next_obs)
        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([next_obs_encoding, netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]

        V = self.critic(critic_input)
        nV = self.critic(next_critic_input)
        # 这里待确定
        TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()
        actor_in=self.actor_encoder(s)
        training_stats = self.actor.fit(actor_in,a_local,sample_weight=TD_error,batch_size=200,epochs=1,verbose=0)

        return training_stats.history['loss'][0]

    def get_critic_weights(self):
        '''
        Returns critic parameters and average loss
        '''
        return self.critic.get_weights()

    def get_TR_weights(self):
        '''
        Returns team-average reward parameters
        '''
        return self.TR.get_weights()

    def get_action(self,state,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        actor_in=self.actor_encoder(state)
        action_prob = self.actor.predict(actor_in).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights()]

class Malicious_CAC_agent():
    '''
    MALICIOUS CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the malicious consensus actor-critic (MCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The MCAC agent receives both local and compromised team reward, and observes the global state and action.
    The adversary seeks to maximize its own objective function and minimize the average objective function of the remaining agents.
    The MCAC agent employs neural networks to approximate the actor and critic. It trains the actor, local critic, compromised team
    critic, and compromised team reward. For the actor updates, the agents uses local rewards and critic. The MCAC agent does not
    apply consensus updates but transmits the compromised critic and team reward parameters.

    ARGUMENTS: NN models for actor and critic, and team reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
    '''
    def __init__(self,actor,critic,team_reward,args,critic_attention_layer,agent_index,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]
        self.args = args
        self.critic_attention_layer=critic_attention_layer
        self.agent_index=agent_index

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic_local_weights = self.critic.get_weights()
        self.TR.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            keras.layers.Dense(self.args['hidden_dim']),            
            keras.layers.LeakyReLU()
            ])
        self.actor_encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            tf.keras.layers.Dense(self.args['hidden_dim']),            
            tf.keras.layers.LeakyReLU()
            ])

    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, local rewards and actions
        RETURNS: training loss
        s和ns的形状为 [B, n_agents, observation_dim]
        '''
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:]  # [B, state_dim], 当前智能体的下一个状态
        obs_encoding=self.encoder(current_obs)
        # obs_encoding=tf.expand_dims(obs_encoding, axis=1)  # [B, 1, hidden_dim]
        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)

        weights_temp = self.critic.get_weights()
        self.critic.set_weights(self.critic_local_weights)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([self.encoder(next_obs), netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]

        V = self.critic(critic_input)
        nV = self.critic(next_critic_input)
        # TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()
        # 这里待确定
        TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()  # 保留 Tensor 形状
        # actor的输入也改成encoding吧
        # verbose设置为 1，它会在控制台打印训练进度
        actor_in=self.actor_encoder(s)

        # 使用输入 actor_in 和目标输出 a_local，基于 TD-error 的加权损失进行一次前向 + 反向传播，更新网络参数
        training_stats = self.actor.fit(actor_in,a_local,sample_weight=TD_error,batch_size=200,epochs=1,verbose=0)
        self.critic.set_weights(weights_temp)

        return training_stats.history['loss'][0]

    def critic_update_compromised(self,s,ns,r_compromised):
        '''
        Stochastic update of the team critic network
        - performs an update of the team critic network
        - evaluates compromised TD targets with a one-step lookahead
        - applies MSE gradients with TD targets as target values
        ARGUMENTS: visited consecutive states, compromised_rewards
                    boolean to reset parameters to prior values
        RETURNS: updated compromised critic hidden and output layer parameters, training loss
        '''
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:]  # [B, state_dim], 当前智能体的下一个状态
        obs_encoding=self.encoder(current_obs)
        next_obs_encoding=self.encoder(next_obs)

        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([next_obs_encoding, netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        nV = self.critic(next_critic_input)
        TD_target_compromised = r_compromised + self.gamma * nV

        training_stats = self.critic.fit(critic_input,TD_target_compromised,epochs=10,batch_size=32,verbose=0)

        return self.critic.get_weights(), training_stats.history["loss"][0]

    def critic_update_local(self,s,ns,r_local):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        weights_temp = self.critic.get_weights()
        self.critic.set_weights(self.critic_local_weights)
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:]  # [B, state_dim], 当前智能体的下一个状态
        obs_encoding=self.encoder(current_obs)
        next_obs_encoding=self.encoder(next_obs)
        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([next_obs_encoding, netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        nV = self.critic(next_critic_input)
        local_TD_target = r_local + self.gamma * nV
        self.critic.fit(critic_input,local_TD_target,epochs=10,batch_size=32,verbose=0)
        self.critic_local_weights = self.critic.get_weights()
        self.critic.set_weights(weights_temp)

    def TR_update_compromised(self,sa,r_compromised):
        '''
        Stochastic update of the team reward network
        - performs a single batch update of the team reward network
        - applies an MSE gradient with compromised rewards as target values
        ARGUMENTS: visited states, team actions, compromised rewards,
                    boolean to reset parameters to prior values
        RETURNS: updated compromised team reward hidden and output layer parameters, training loss
        '''
        # sa是已经在函数外处理好了的[B, n_agents, observation_dim + 1]
        training_stats = self.TR.fit(sa,r_compromised,epochs=10,batch_size=32,verbose=0)

        return self.TR.get_weights(), training_stats.history["loss"][0]

    def get_action(self,state,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        random_action = np.random.choice(self.n_actions)
        actor_in=self.actor_encoder(state)
        action_prob = self.actor.predict(actor_in).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights(), self.critic_local_weights]

class Greedy_CAC_agent():
    '''
    GREEDY CONSENSUS ACTOR-CRITIC AGENT
    This is an implementation of the greedy consensus actor-critic (GCAC) agent that is trained simultaneously with the resilient
    consensus actor-critic (RCAC) agents. The algorithm is a realization of temporal difference learning with one-step lookahead,
    also known as TD(0). The GCAC agent receives a local reward, and observes the global state and action. The GCAC agent seeks
    to maximize its own objective function and is oblivious to the remaining agents' objectives. It employs neural networks
    to approximate the actor, critic, and estimated reward function. For the actor updates, the agents uses the local rewards
    and critic. The GCAC agent does not apply consensus updates but transmits its critic and reward function parameters.

    ARGUMENTS: NN models for actor and critic, and team reward
               slow learning rate (for the actor network)
               fast learning rate (for the critic and team reward networks)
               discount factor gamma
    '''

    def __init__(self,actor,critic,team_reward,args,critic_attention_layer,agent_index,slow_lr,fast_lr,gamma=0.95):
        self.actor = actor
        self.critic = critic
        self.TR = team_reward
        self.gamma = gamma
        self.n_actions=self.actor.output_shape[1]
        self.args = args
        self.critic_attention_layer=critic_attention_layer
        self.agent_index=agent_index

        self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=slow_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.TR.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=fast_lr),loss=keras.losses.MeanSquaredError())
        self.encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            keras.layers.Dense(self.args['hidden_dim']),            
            keras.layers.LeakyReLU()
            ])
        self.actor_encoder = tf.keras.Sequential([
            # tf.keras.layers.BatchNormalization(axis=-1, center=False, scale=False),
            tf.keras.layers.Dense(self.args['hidden_dim']),            
            tf.keras.layers.LeakyReLU()
            ])

    def actor_update(self,s,ns,r_local,a_local):
        '''
        Stochastic update of the actor network
        - performs a single update of the actor
        - computes TD errors with a one-step lookahead
        - applies the TD errors as sample weights for the cross-entropy gradient
        ARGUMENTS: visited states, local rewards and actions
        RETURNS: training loss
        '''
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:] # [B, state_dim], 当前智能体的下一个状态
        obs_encoding=self.encoder(current_obs)
        next_obs_encoding=self.encoder(next_obs)
        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([next_obs_encoding, netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]

        V = self.critic(critic_input)
        nV = self.critic(next_critic_input)
        TD_error = tf.squeeze(r_local + self.gamma * nV - V).numpy()
        actor_in=self.actor_encoder(s)
        training_stats = self.actor.fit(actor_in,a_local,sample_weight=TD_error,batch_size=200,epochs=1,verbose=0)

        return training_stats.history['loss'][0]

    def critic_update_local(self,s,ns,r_local):
        '''
        Local stochastic update of the critic network
        - performs a stochastic update of the critic network using local rewards
        - evaluates a local TD target with a one-step lookahead
        - applies an MSE gradient with the local TD target as a target value
        ARGUMENTS: visited consecutive states, local rewards
        RETURNS: updated critic parameters
        '''
        current_obs=s[:,self.agent_index,:]  # [B, state_dim], 当前智能体的状态
        next_obs=ns[:,self.agent_index,:]

        obs_encoding=self.encoder(current_obs)
        next_obs_encoding=self.encoder(next_obs)
        attention_output = self.critic_attention_layer(s, self.agent_index)
        netx_attention_output = self.critic_attention_layer(ns, self.agent_index)
        critic_input = tf.concat([obs_encoding, attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        next_critic_input = tf.concat([next_obs_encoding, netx_attention_output], axis=-1)  # [batch_size, 2 * hidden_dim]
        nV = self.critic(next_critic_input)
        local_TD_target = r_local + self.gamma * nV
        training_stats = self.critic.fit(critic_input,local_TD_target,epochs=10,batch_size=32,verbose=0)

        return self.critic.get_weights(), training_stats.history['loss'][0]

    def TR_update_local(self,sa,r_local):
        '''
        Local stochastic update of the team reward network
        - performs a stochastic update of the team-average reward network
        - applies an MSE gradient with a local reward as a target value
        ARGUMENTS: state-action pairs, local rewards
        RETURNS: updated team reward parameters
        '''
        # sa是已经在函数外处理好了的[B, n_agents, observation_dim + 1]
        training_stats = self.TR.fit(sa,r_local,epochs=10,batch_size=32,verbose=0)

        return self.TR.get_weights(), training_stats.history['loss'][0]

    def get_action(self,state,mu=0.1):
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        '''Choose an action at the current state
            - set from_policy to True to sample from the actor
            - set from_policy to False to sample from the random uniform distribution over actions
            - set mu to [0,1] to control probability of choosing a random action
        '''
        actor_in=self.actor_encoder(state)
        random_action = np.random.choice(self.n_actions)
        action_prob = self.actor.predict(actor_in).ravel()
        action_from_policy = np.random.choice(self.n_actions, p = action_prob)
        self.action = np.random.choice([action_from_policy,random_action], p = [1-mu,mu])

        return self.action

    def get_parameters(self):

        return [self.actor.get_weights(), self.critic.get_weights(), self.TR.get_weights()]
