import numpy as np
import tensorflow as tf


tf.get_logger().setLevel('ERROR')

def test_BRAC(env,agents,args):
    n_agents = env.n_agents
    action = np.zeros(n_agents)
    print("start: ")
    state = env.initial_state
    
    while(1):
        sclae_state, reward, done = env.get_data()
        print("state: \n", state, "\n  done:  ", done)

        if np.all(done==1):
            break
        for node in range(n_agents):
            # action[node] = agents[node].get_action(sclae_state.reshape(1,sclae_state.shape[0],sclae_state.shape[1]), decay=0, mu=0)
            if done[node]:
                action[node] = 0
            else:
                action[node] = agents[node].get_action(sclae_state.reshape(1,sclae_state.shape[0],sclae_state.shape[1]), decay=0, mu=0)
        print("action: ", action)
        state = env.step(action)
        # env.render()
        
    env.close()
