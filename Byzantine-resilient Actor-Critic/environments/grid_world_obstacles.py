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
    state的形状是 (n_agents,state_dim)，表示每个智能体的坐标
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
        '''重置环境，确保智能体的初始位置有效'''
        # 为了高效查找，将障碍物列表转换为集合 (Set)
        obstacle_set = {tuple(obs) for obs in self.obstacles_list}

        if self.randomize_state:
            # 初始化一个空的 state 数组来存放智能体位置
            self.state = np.zeros((self.n_agents, self.n_states), dtype=int)
            
            # 逐个为每个智能体生成不冲突的位置
            for i in range(self.n_agents):
                while True:
                    # 1. 生成一个随机的候选位置
                    # 注意 size=self.n_states 会生成一个一维数组，例如 [x, y]
                    pos = np.random.randint([0, 0], [self.nrow, self.ncol], size=self.n_states)

                    # 2. 检查该位置是否是障碍物
                    # 我们将 pos 数组转为元组，以便在集合中快速查找
                    if tuple(pos) in obstacle_set:
                        continue  # 位置无效，重新生成

                    # 3. 如果位置有效，则分配该位置并跳出循环，为下一个智能体生成位置
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

            # 障碍物碰撞改动
            proposed_pos = self.state[node] + move
            proposed_pos = np.clip(proposed_pos, 0, self.nrow - 1) # 确保不出界

            # 2. 检查提议的新位置是否是障碍物
            is_obstacle_collision = False
            if tuple(proposed_pos) in obstacle_set:
                is_obstacle_collision = True
                # 如果撞到障碍物，智能体的位置不应改变
                proposed_pos = self.state[node]

            # 3. 更新状态
            # 无论是否撞墙，我们都需要更新状态，因为可能与其他智能体的位置有关
            # 如果撞墙，proposed_pos 已经被重置为原位置
            self.state[node] = proposed_pos

            # 更新当前状态
            # self.state[node] = np.clip(self.state[node] + move,0,self.nrow - 1)
            # 创建一个掩码以选择除当前智能体外的所有其他智能体
            other_agents_mask = np.arange(self.n_agents) != node
            other_agents_states = self.state[other_agents_mask, :]
            dist_to_agents = np.min(np.sum(abs(other_agents_states-self.state[node]),axis=1))
            dist_to_goal_next = np.sum(abs(self.state[node]-self.desired_state[node]))
            #  if self.obstacle_map[proposed_pos[0], proposed_pos[1]]:
            if dist_to_agents > 0 and not is_obstacle_collision: #agent moves to a new cell
                self.reward[node] = - dist_to_goal_next
            elif dist_to_goal == 0 and action[node] == 0:
                self.reward[node] = 0
            # elif is_obstacle_collision:  # 撞到障碍物
            #     # 如果撞到障碍物，奖励为负的距离到目标加上一个惩罚
            #     self.reward[node] = - dist_to_goal - 10
            else:
                self.reward[node] = - dist_to_goal - 1

    def get_data(self):
        '''
        返回缩放后的状态和奖励
        Returns scaled reward and state, and flags if the agents have reached the target
        '''
        state_scaled = (self.state - self.mean_state) / self.std_state
        reward_scaled = self.reward / 5
        return state_scaled, reward_scaled
    
    def get_observations(self):
        """
        为每个 agent 构建 observation：
        - 自身位置 (state_dim,)
        - 每个目标位置相对自身的位置 (n_agents, state_dim)
        - 每个其他 agent 相对自身的位置 (n_agents - 1, state_dim)
        返回值:
        observations: [n_agents, observation_dim]
        """
        observations = []
        for i in range(self.n_agents):
            own_pos = (self.state[i] - self.mean_state) / self.std_state
            rel_goals = (self.state[i]-self.desired_state[i]) / self.std_state
            rel_agents = (np.delete(self.state, i, axis=0) - self.state[i]) / self.std_state

            # 拼接 observation
            obs = np.concatenate([
                own_pos, # 自身位置
                rel_goals.flatten(), # 每个目标位置相对自身的位置
                rel_agents.flatten()    # 每个其他 agent 相对自身的位置
            ])
            observations.append(obs)

        # observations的尺寸为 (n_agents, observation_dim)
        observations = np.stack(observations, axis=0)
        observations = tf.expand_dims(observations, axis=0)  # [1, n_agents, observation_dim]
        observations = tf.cast(observations, dtype=tf.float32)

        return observations
   

    def close(self):
        pass
