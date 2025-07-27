import os
import numpy as np
import gym
import argparse
import pickle
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from Attention import CriticAttention
from critics import AttentionCritic
from environments.grid_world import Grid_World
from environments.grid_world_obstacles import Grid_World_Obstacles
from environments.grid_world_obs_3D import Grid_World_3D_Obstacles
from agents.resilient_CAC_agents import RPBCAC_agent
from agents.adversarial_CAC_agents import Faulty_CAC_agent, Greedy_CAC_agent, Malicious_CAC_agent
import training.train_agents as training
import test_agents as testing

'''
Cooperative navigation problem with resilient consensus and adversarial actor-critic agents
- This is a main file, where the user selects learning hyperparameters, environment parameters,
  and neural network architecture for the actor, critic, and team reward estimates.
- The script triggers a training process whose results are passed to folder Simulation_results.
'''

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training consensus AC agents')
    parser.add_argument('--n_agents',help='total number of agents',type=int,default=5)
    parser.add_argument('--agent_label', help='classification of each agent (Cooperative,Malicious,Faulty,Greedy)',type=str, default=['Cooperative','Cooperative','Cooperative','Cooperative','Cooperative'])
    parser.add_argument('--in_nodes',help='specify a list of neighbors that transmit values to each agent (include the index of the agent as the first element)',type=int,default=[[0,1,2,3],[1,2,3,4],[2,3,4,0],[3,4,0,1],[4,0,1,2]])
    parser.add_argument('--n_actions',help='size of action space of each agent',type=int,default=5)
    parser.add_argument('--n_states',help='state dimension of each agent',type=int,default=2)
    parser.add_argument('--n_episodes', help='Total number of episodes', type=int, default=20000)
    parser.add_argument('--max_ep_len', help='Number of steps per episode', type=int, default=20)
    parser.add_argument('--n_ep_fixed',help='Number of episodes under a fixed policy',type=int,default=50)
    parser.add_argument('--n_epochs',help='Number of updates in the policy evaluation',type=int,default=20)
    parser.add_argument('--slow_lr', help='actor network learning rate',type=float, default=0.00029)
    parser.add_argument('--fast_lr', help='critic network learning rate',type=float, default=0.003)
    parser.add_argument('--batch_size', help='batch size for policy evaluation',type=int,default=200)
    parser.add_argument('--buffer_size',help='size of experience replay buffer',type=int,default=5000)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--H', help='max number of adversaries in the local neighborhood', type=int, default=0)
    parser.add_argument('--common_reward',help='Set to True if the agents receive the team-average reward',default=False)
    parser.add_argument('--summary_dir',help='Create a directory to save simulation results', default='./simulation_results/')
    parser.add_argument('--pretrained_agents',help='Set to True if the agents have been pretrained',default=False)
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=50)
    parser.add_argument('--attend_heads',help='attend_heads',type=int,default=2)
    parser.add_argument('--hidden_dim',help='hidden dimension of the attention layer',type=int,default=128)
    parser.add_argument('--obstacles', help='Set to True if the environment contains obstacles', default=True)
    parser.add_argument('--3Dobstacles', help='Set to True if the environment contains 3D obstacles', default=True)

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])
    s_desired = np.random.randint(0,10,size=(args['n_agents'],args['n_states']))
    s_initial = np.random.randint(0,10,size=(args['n_agents'],args['n_states']))

    #----------------------------------------------------------------------------------------------------------------------------------------
    if args['pretrained_agents']:
        pretrained_weights = np.load('pretrained_weights.npy', allow_pickle=True)
        s_desired = np.load('desired_state.npy', allow_pickle=True)
    

    '''NEURAL NETWORK ARCHITECTURE'''
    agents = []
    attentions = {}
    observation_dim = args['n_states']*(args['n_agents']+1)

    for node in range(args['n_agents']):
        actor = keras.Sequential([
                                    keras.Input(shape=(args['n_agents'],args['hidden_dim'])),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(args['n_actions'], activation='softmax')
                                  ])

        critic_attention_layer = CriticAttention(state_dim=observation_dim, hidden_dim=args['hidden_dim'], num_heads=args['attend_heads'])
        attentions[node]=critic_attention_layer
        critic = keras.Sequential([
                                    keras.Input(shape=(2*args['hidden_dim'],)),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(128, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(1)
                                  ])

        team_reward = keras.Sequential([
                                    keras.Input(shape=(args['n_agents'],observation_dim + 1)),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(64, activation=keras.layers.LeakyReLU(alpha=0.1)),
                                    keras.layers.Dense(1)
                                  ])
        if args['pretrained_agents']:
            actor.set_weights(pretrained_weights[node][0])
            critic.set_weights(pretrained_weights[node][1])
            team_reward.set_weights(pretrained_weights[node][2])

        if args['agent_label'][node] == 'Malicious':        #create a malicious agent
            print("This is a malicious agent")
            agents.append(Malicious_CAC_agent(actor,critic,team_reward,args,critic_attention_layer,node,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))
            if args['pretrained_agents']:
                agents[node].critic_local_weights = pretrained_weights[node][3]

        elif args['agent_label'][node] == 'Faulty':         #create a faulty agent
            print("This is a faulty agent")
            agents.append(Faulty_CAC_agent(actor,critic,team_reward,args,critic_attention_layer,node,slow_lr = args['slow_lr'],gamma = args['gamma']))

        elif args['agent_label'][node] == 'Greedy':         #create a greedy agent
            print("This is a greedy agent")
            agents.append(Greedy_CAC_agent(actor,critic,team_reward,args,critic_attention_layer,node,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma']))

        else: # args['agent_label'][node] == 'Cooperative': #create a cooperative agent
            print("This is an RPBCAC agent")
            agents.append(RPBCAC_agent(actor,critic,team_reward,args,critic_attention_layer,node,slow_lr = args['slow_lr'],fast_lr = args['fast_lr'],gamma = args['gamma'],H = args['H']))

    print(args,s_desired, s_initial)


    if args['obstacles']:

        if args['3Dobstacles']:
            if args['pretrained_agents']:
                '''TEST AGENTS'''
                env = Grid_World_3D_Obstacles(nrow=10,
                            ncol=10,
                            n_agents=args['n_agents'],
                            desired_state=s_desired,
                            initial_state=s_initial,
                            randomize_state=False,
                            scaling=True
                            )
                testing.test_RPBCAC(env, agents, args)
            else:
            #---------------------------------------------------------------------------------------------------------------------------------------------
                '''TRAIN AGENTS'''
                env = Grid_World_3D_Obstacles(nrow=10, ncol=10, n_agents=args['n_agents'], desired_state=s_desired,
                                initial_state=s_initial, randomize_state=True, scaling=True)
                agent_weights,sim_data = training.train_RPBCAC(env,agents,args)
            #----------------------------------------------------------------------------------------------------
                sim_data.to_pickle("/home/icdm/yjt/(may attention)Resilient-consensus/Resilient-consensus-based-MARL-main/20250630/sim_data.pkl")
                # np.save('pretrained_weights.npy', agent_weights, allow_pickle=True)
                np.save("/home/icdm/yjt/(may attention)Resilient-consensus/Resilient-consensus-based-MARL-main/20250630/pretrained_weights.npy", np.array(agent_weights, dtype=object), allow_pickle=True)
                np.save('/home/icdm/yjt/(may attention)Resilient-consensus/Resilient-consensus-based-MARL-main/20250630/desired_state.npy',s_desired,allow_pickle=True)

        else:
            obstacles_list = [
                    [1, 6],
                    [3, 2],
                    [8, 3],
                    [4, 5],
                    [6, 8]
                ]

            if args['pretrained_agents']:
                '''TEST AGENTS'''
                env = Grid_World_Obstacles(nrow=10,
                            ncol=10,
                            n_agents=args['n_agents'],
                            obstacles=obstacles_list,
                            desired_state=s_desired,
                            initial_state=s_initial,
                            randomize_state=False,
                            scaling=True
                            )
                testing.test_RPBCAC(env, agents, args)
            else:
            #---------------------------------------------------------------------------------------------------------------------------------------------
                '''TRAIN AGENTS'''
                env = Grid_World_Obstacles(nrow=10, ncol=10, n_agents=args['n_agents'], obstacles_list=obstacles_list,desired_state=s_desired,
                                initial_state=s_initial, randomize_state=True, scaling=True)
                agent_weights,sim_data = training.train_RPBCAC(env,agents,args)
            #----------------------------------------------------------------------------------------------------
                sim_data.to_pickle("/sim_data.pkl")
                # np.save('pretrained_weights.npy', agent_weights, allow_pickle=True)
                np.save("/pretrained_weights.npy", np.array(agent_weights, dtype=object), allow_pickle=True)
                np.save('/desired_state.npy',s_desired,allow_pickle=True)

    else:
        if args['pretrained_agents']:
            '''TEST AGENTS'''
            env = Grid_World(nrow=5,
                        ncol=5,
                        n_agents=args['n_agents'],
                        desired_state=s_desired,
                        initial_state=s_initial,
                        randomize_state=False,
                        scaling=True
                        )
            testing.test_RPBCAC(env, agents, args)
        else:
        #---------------------------------------------------------------------------------------------------------------------------------------------
            '''TRAIN AGENTS'''
            env = Grid_World(nrow=10,
                        ncol=10,
                        n_agents=args['n_agents'],
                        desired_state=s_desired,
                        initial_state=s_initial,
                        randomize_state=True,
                        scaling=True
                        )
            agent_weights,sim_data = training.train_RPBCAC(env,agents,args)
        #----------------------------------------------------------------------------------------------------
            sim_data.to_pickle("/sim_data.pkl")
            # np.save('pretrained_weights.npy', agent_weights, allow_pickle=True)
            np.save("/pretrained_weights.npy", np.array(agent_weights, dtype=object), allow_pickle=True)
            np.save('/desired_state.npy',s_desired,allow_pickle=True)
