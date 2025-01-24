import numpy as np
import gym
from gym import spaces

class Grid_World(gym.Env):

    metadata = {'render.modes': ['console']}

    def __init__(self, size = 5, n_agents = 1, desired_state = None,initial_state = None,randomize_state = True,scaling = False):
        self.lwh = size              # length*width*height
        self.n_agents = n_agents
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.n_states = 3
        self.actions_dict = {0:np.array([0,0,0]), 1:np.array([-1,-1,1]), 2:np.array([0,-1,1]), 3:np.array([1,-1,1]), 4:np.array([-1,0,1]),
                             5:np.array([0,0,1]), 6:np.array([1,0,1]), 7:np.array([-1,1,1]), 8:np.array([0,1,1]), 9:np.array([1,1,1]),
                             10:np.array([-1,-1,0]), 11:np.array([0,-1,0]), 12:np.array([1,-1,0]), 13:np.array([-1,0,0]), 14:np.array([1,0,0]),
                             15:np.array([-1,1,0]), 16:np.array([0,1,0]), 17:np.array([1,1,0]), 18:np.array([-1,-1,-1]), 19:np.array([0,-1,-1]),
                             20:np.array([1,-1,-1]), 21:np.array([-1,0,-1]), 22:np.array([0,0,-1]), 23:np.array([1,0,-1]), 24:np.array([-1,1,-1]),
                             25:np.array([0,1,-1]), 26:np.array([1,1,-1])}
        

        # self.actions_dict = {0:np.array([0,0,0]), 1:np.array([0,0,1]), 2:np.array([0,0,-1]), 3:np.array([0,1,0]), 4:np.array([0,-1,0]),
        #                      5:np.array([1,0,0]), 6:np.array([-1,0,0])}

        # self.actions_dict = {0:np.array([0,0]), 1:np.array([0,1]), 2:np.array([0,-1]), 3:np.array([1,0]), 4:np.array([-1,0])}
        
        self.done = np.zeros(self.n_agents)

        self.crash = False
        self.obs_dis = 1
        self.alpha = 5
        self.beta = 15


        self.obstacle = [[2,1,2],[2,1,3],[2,2,3],[2,3,2],[2,3,3]]
        # self.obstacle = [[2,2],[2,3],[2,4],[3,4],[4,4],[5,4],[5,3],[5,2]]

        self.reset()

        if scaling:
            x,y,z=np.arange(size),np.arange(size),np.arange(size)
            self.mean_state=np.array([np.mean(x),np.mean(y),np.mean(z)])
            self.std_state=np.array([np.std(x),np.std(y),np.std(z)])
            # x,y=np.arange(size),np.arange(size)
            # self.mean_state=np.array([np.mean(x),np.mean(y)])
            # self.std_state=np.array([np.std(x),np.std(y)])
        else:
            self.mean_state,self.std_state=0,1

    def reset(self):

        if self.randomize_state:
            self.state = np.random.randint([0,0,0],[self.lwh,self.lwh,self.lwh],size=(self.n_agents,self.n_states))
            for i in range(self.n_agents):
                obsarry = np.array(self.obstacle)
                while self.state[i] in obsarry:
                    self.state[i] = np.random.randint([0,0,0],[self.lwh,self.lwh])
            # self.state = np.random.randint([0,0],[self.lwh,self.lwh],size=(self.n_agents,self.n_states))
            # for i in range(self.n_agents):
            #     obsarry = np.array(self.obstacle)
            #     while self.state[i] in obsarry:
            #         self.state[i] = np.random.randint([0,0],[self.lwh,self.lwh])
        else:
            self.state = np.array(self.initial_state)
        self.reward = np.zeros(self.n_agents)
        self.done = np.zeros(self.n_agents)
        self.crash = False
        return self.state

    def isCrash(self):
        return self.crash

    def step(self, action):

        for node in range(self.n_agents):

            move = self.actions_dict[action[node]]
            dist_to_goal = np.sum(abs(self.state[node]-self.desired_state[node]))

            new_state = np.clip(self.state[node] + move,0,self.lwh - 1)
    
            dist_to_goal_next = np.sum(abs(new_state-self.desired_state[node]))
            
            dist_to_agents = np.min(np.sum(abs(self.state-self.state[node]),axis=1))
            dist_to_obstacle = np.min(np.sum(abs(self.obstacle-new_state),axis=1))


            if dist_to_goal == 0:
                self.done[node] = 1
            elif dist_to_goal_next==0:
                self.reward[node] = 10
                self.state[node] = new_state
            elif dist_to_obstacle == 0:
                self.reward[node] = -10
                self.crash = True
            else: 
                self.reward[node] = 10/dist_to_goal_next
                self.state[node] = new_state
            
        
        return self.state


    def get_data(self):

        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward / self.n_agents
        return state_scaled, reward_scaled, self.done

    def close(self):
        pass
