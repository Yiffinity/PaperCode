import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
import pandas as pd

tf.get_logger().setLevel('ERROR')


def train_BRAC(env,agents,args,exp_buffer=None):

    paths = []
    n_agents, n_states = env.n_agents, args['n_states']
    n_coop = args['agent_label'].count('Cooperative')
    gamma = args['gamma']
    in_nodes = args['in_nodes']
    max_ep_len, n_episodes, n_ep_fixed = args['max_ep_len'], args['n_episodes'], args['n_ep_fixed']
    n_epochs, batch_size, buffer_size = args['n_epochs'], args['batch_size'], args['buffer_size']

    if exp_buffer:
        states = exp_buffer[0]
        nstates = exp_buffer[1]
        actions = exp_buffer[2]
        rewards = exp_buffer[3]
    else:
        states, nstates, actions, rewards = [], [], [], []
    #---------------------------------------------------------------------------
    '                                 TRAINING                                 '
    #---------------------------------------------------------------------------
    for t in range(n_episodes):

        j,  ep_returns = 0, 0
        est_returns, mean_true_returns, mean_true_returns_adv = [], 0, 0
        action, actor_loss, critic_loss, TR_loss = np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents), np.zeros(n_agents)
        i = t % n_ep_fixed

        env.reset()
        state, _, _ = env.get_data()

        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                est_returns.append(agents[node].critic(state.reshape(1,state.shape[0],state.shape[1]))[0][0].numpy())

        while j < max_ep_len:
            for node in range(n_agents):
                action[node] = agents[node].get_action(state.reshape(1,state.shape[0],state.shape[1]),t)
            env.step(action)
            nstate, reward, _ = env.get_data()
            ep_returns += reward * (gamma ** j)
            j += 1

            states.append(np.array(state))
            nstates.append(np.array(nstate))
            actions.append(np.array(action).reshape(-1,1))
            rewards.append(np.array(reward).reshape(-1,1))
            state = np.array(nstate)

            if i == n_ep_fixed-1 and j == max_ep_len:

                s = tf.convert_to_tensor(states,tf.float32)
                ns = tf.convert_to_tensor(nstates,tf.float32)
                r = tf.convert_to_tensor(rewards,tf.float32)
                a = tf.convert_to_tensor(actions,tf.float32)
                sa = tf.concat([s,a],axis=-1)

                r_coop = tf.zeros([r.shape[0],r.shape[2]],tf.float32)
                for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):
                    r_coop += r[:,node] / n_coop

                for n in range(n_epochs):
                    critic_weights,TR_weights = [],[]

                    #LOCAL 
                    for node in range(n_agents):
                        r_applied = r_coop if args['common_reward'] else r[:,node]
                        if args['agent_label'][node] == 'Cooperative':
                            x, TR_loss[node] = agents[node].TR_update_local(sa,r_applied)
                            y, critic_loss[node] = agents[node].critic_update_local(s,ns,r_applied)
                        elif args['agent_label'][node] == 'Greedy':
                            x, TR_loss[node] = agents[node].TR_update_local(sa,r[:,node])
                            y, critic_loss[node] = agents[node].critic_update_local(s,ns,r[:,node])
                        elif args['agent_label'][node] == 'Malicious':
                            agents[node].critic_update_local(s,ns,r[:,node])
                            x, TR_loss[node] = agents[node].TR_update_compromised(sa,-r_coop)
                            y, critic_loss[node] = agents[node].critic_update_compromised(s,ns,-r_coop)
                        TR_weights.append(x)
                        critic_weights.append(y)
                    #BR-AC
                    for node in (x for x in range(n_agents) if args['agent_label'][x] == 'Cooperative'):

                        critic_weights_innodes = [critic_weights[i] for i in in_nodes[node]]
                        TR_weights_innodes = [TR_weights[i] for i in in_nodes[node]]

                        agents[node].resilient_consensus_critic_hidden(critic_weights_innodes)
                        agents[node].resilient_consensus_TR_hidden(TR_weights_innodes)

                        critic_agg = agents[node].resilient_consensus_critic(s,critic_weights_innodes)
                        TR_agg = agents[node].resilient_consensus_TR(sa,TR_weights_innodes)

                        agents[node].critic_update_team(s,critic_agg)
                        agents[node].TR_update_team(sa,TR_agg)

                for node in range(n_agents):
                    if args['agent_label'][node] == 'Cooperative':
                        actor_loss[node] = agents[node].actor_update(s[-max_ep_len*n_ep_fixed:],ns[-max_ep_len*n_ep_fixed:],sa[-max_ep_len*n_ep_fixed:],a[-max_ep_len*n_ep_fixed:,node])
                    else:
                        actor_loss[node] = agents[node].actor_update(s[-max_ep_len*n_ep_fixed:],ns[-max_ep_len*n_ep_fixed:],r[-max_ep_len*n_ep_fixed:,node],a[-max_ep_len*n_ep_fixed:,node])


                if len(states) > buffer_size:
                    q = len(states) - buffer_size
                    del states[:q]
                    del nstates[:q]
                    del actions[:q]
                    del rewards[:q]

        for node in range(n_agents):
            if args['agent_label'][node] == 'Cooperative':
                mean_true_returns += ep_returns[node]/n_coop
            else:
                mean_true_returns_adv += ep_returns[node]/(n_agents-n_coop)

        output = '| Episode: {} | Est. returns: {} | Returns: {} | Average critic loss: {} | Average TR loss: {} | Average actor loss: {}'.format(t,est_returns,mean_true_returns,critic_loss,TR_loss,actor_loss)
        print(output)
        with open('./log.txt', 'a') as file:
            file.write(output + '\n')  
            
        path = {
                "True_team_returns":mean_true_returns,
                "True_adv_returns":mean_true_returns_adv,
                "Estimated_team_returns":np.mean(est_returns)
               }
        paths.append(path)

    sim_data = pd.DataFrame.from_dict(paths)
    weights = [agent.get_parameters() for agent in agents]
    return weights,sim_data
