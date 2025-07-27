import numpy as np
import gym
from gym import spaces
import tensorflow as tf

class Grid_World_Obstacles(gym.Env):
    """
    Multi-agent grid-world: cooperative navigation 2
    This is a grid-world environment designed for the cooperative navigation problem. Each agent seeks to navigate to the desired position. The agent chooses one of five admissible actions
    (stay,left,right,down,up) and makes a transition only if the adjacent cell is not occupied. It receives a reward equal to the L1 distance between the visited cell and the target.
    ARGUMENTS:  nrow, ncol: grid world dimensions
                n_agents: number of agents
                desired_state: desired position of each agent
                initial_state: initial position of each agent
                randomize_state: True if the agents' initial position is randomized at the beginning of each episode
                scaling: determines if the states are scaled
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, nrow = 5, ncol=5, n_agents = 1, obstacles_list=None,desired_state = None,initial_state = None,randomize_state = True,scaling = False):
        self.nrow = nrow
        self.ncol = ncol
        self.n_agents = n_agents
        self.obstacles_list = obstacles_list if obstacles_list is not None else []
        self.initial_state = initial_state
        self.desired_state = desired_state
        self.randomize_state = randomize_state
        self.n_states = 2
        self.actions_dict = {0:np.array([0,0]), 1:np.array([-1,0]), 2:np.array([1,0]), 3:np.array([0,-1]), 4:np.array([0,1])}
        self.reset()

        if scaling:
            x,y=np.arange(nrow),np.arange(ncol)
            self.mean_state=np.array([np.mean(x),np.mean(y)])
            self.std_state=np.array([np.std(x),np.std(y)])
        else:
            self.mean_state,self.std_state=0,1

    def reset(self):
        obstacle_set = {tuple(obs) for obs in self.obstacles_list}

        if self.randomize_state:
            self.state = np.zeros((self.n_agents, self.n_states), dtype=int)
            
            for i in range(self.n_agents):
                while True:
                    pos = np.random.randint([0, 0], [self.nrow, self.ncol], size=self.n_states)
                    if tuple(pos) in obstacle_set:
                        continue
                    self.state[i] = pos
                    break
        else:
            self.state = np.array(self.initial_state)
        self.reward = np.zeros(self.n_agents)
        return self.state

    def step(self, action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''

        obstacle_set = {tuple(pos) for pos in self.obstacles_list}

        for node in range(self.n_agents):
            move = self.actions_dict[action[node]]
            dist_to_goal = np.sum(abs(self.state[node]-self.desired_state[node]))

            proposed_pos = self.state[node] + move
            proposed_pos = np.clip(proposed_pos, 0, self.nrow - 1)

            is_obstacle_collision = False
            if tuple(proposed_pos) in obstacle_set:
                is_obstacle_collision = True
                proposed_pos = self.state[node]

            self.state[node] = proposed_pos

            # self.state[node] = np.clip(self.state[node] + move,0,self.nrow - 1)
            other_agents_mask = np.arange(self.n_agents) != node
            other_agents_states = self.state[other_agents_mask, :]
            dist_to_agents = np.min(np.sum(abs(other_agents_states-self.state[node]),axis=1))
            dist_to_goal_next = np.sum(abs(self.state[node]-self.desired_state[node]))
            #  if self.obstacle_map[proposed_pos[0], proposed_pos[1]]:
            if dist_to_agents > 0 and not is_obstacle_collision: #agent moves to a new cell
                self.reward[node] = - dist_to_goal_next
            elif dist_to_goal == 0 and action[node] == 0:
                self.reward[node] = 0
            else:
                self.reward[node] = - dist_to_goal - 1

    def get_data(self):
        '''
        Returns scaled reward and state, and flags if the agents have reached the target
        '''
        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward / 5
        return state_scaled, reward_scaled
    
    def get_observations(self):
        """
        observations: [n_agents, observation_dim]
        """
        observations = []
        for i in range(self.n_agents):
            own_pos = (self.state[i] - self.mean_state) / self.std_state
            rel_goals = (self.state[i]-self.desired_state[i]) / self.std_state
            rel_agents = (np.delete(self.state, i, axis=0) - self.state[i]) / self.std_state

            obs = np.concatenate([
                own_pos,
                rel_goals.flatten(),
                rel_agents.flatten()
            ])
            observations.append(obs)

        observations = np.stack(observations, axis=0)
        observations = tf.expand_dims(observations, axis=0)  # [1, n_agents, observation_dim]
        observations = tf.cast(observations, dtype=tf.float32)

        return observations
   

    def close(self):
        pass
