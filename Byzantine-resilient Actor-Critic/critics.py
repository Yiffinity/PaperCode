import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

class AttentionCritic(Model):
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Attention Critic 网络，每个代理获取自己的观察和动作，也可以关注其他代理的编码观察和动作。
        
        :param sa_sizes: 每个智能体的状态和动作空间的大小列表 [(state_dim, action_dim), ...]
        :param hidden_dim: 隐藏层的维度
        :param norm_in: 是否对输入应用BatchNorm
        :param attend_heads: 注意力头的数量
        """
        super(AttentionCritic, self).__init__()
        assert hidden_dim % attend_heads == 0  # 确保 hidden_dim 能够整除 attend_heads
        
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        # 定义 Critic 编码器、Critic 网络和 State 编码器
        self.critic_encoders = []
        self.critics = []
        self.state_encoders = []

        for sdim, adim in sa_sizes:
            # 输入是状态 + 动作，输出是 Q(s,a)
            idim = sdim + adim
            critic_out_dim = 1  # Q 值的输出维度
            odim = adim
            
            # 创建 critic_encoder
            encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(idim,)),
                tf.keras.layers.BatchNormalization(scale=False) if norm_in else tf.keras.layers.Layer(),
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.LeakyReLU()
                ])

            self.critic_encoders.append(encoder)

            # 创建 critic 网络
            critic = tf.keras.Sequential()
            # 输入：[batch, 2 * hidden_dim]，输出：[batch, hidden_dim]
            critic.add(tf.keras.layers.Dense(hidden_dim, input_shape=(2 * hidden_dim,)))
            # 输入：[batch, hidden_dim]，输出：[batch, hidden_dim]
            critic.add(tf.keras.layers.LeakyReLU())
            # 输入：[batch, hidden_dim]，输出：[batch, odim]
            critic.add(tf.keras.layers.Dense(critic_out_dim))
            self.critics.append(critic)

            # 创建 state_encoder，用于生成 query 向量
            state_encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(sdim,)),  # 显式声明输入形状
                tf.keras.layers.BatchNormalization(scale=False) if norm_in else tf.keras.layers.Layer(),  # 条件添加 BN
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.LeakyReLU()
            ])

            self.state_encoders.append(state_encoder)

        # 初始化注意力机制：key、query、value
        attend_dim = hidden_dim // attend_heads
        # 创建空的列表用于存储每个头的 key, selector, value 映射
        self.key_extractors = []
        self.selector_extractors = []
        self.value_extractors = []

        # 构造：[head1, head2, ..., headN] 每头有自己的 key、query、value 映射。
        for i in range(attend_heads):
            self.key_extractors.append(tf.keras.layers.Dense(attend_dim, use_bias=False))
            self.selector_extractors.append(tf.keras.layers.Dense(attend_dim, use_bias=False))
            self.value_extractors.append(tf.keras.Sequential([
                tf.keras.layers.Dense(attend_dim, use_bias=False),
                tf.keras.layers.LeakyReLU()
            ]))

        # Register submodules manually to ensure Keras tracks them
        # for idx, net in enumerate(self.critic_encoders + self.state_encoders + self.critics):
        #     self.__setattr__(f'submodel_{idx}', net)
        # for idx, net in enumerate(self.key_extractors + self.value_extractors + self.selector_extractors):
        #     self.__setattr__(f'attn_module_{idx}', net)

        # 将共享模块添加到列表中
        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]
    
    '''
        这里先不改 后面在改
    '''
    # def shared_parameters(self):
    #     """
    #     Parameters shared across agents and reward heads
    #     """
    #     # 获取所有共享层的权重
    #     shared_params = []
    #     for module in self.shared_modules:
    #         shared_params.extend(module.trainable_variables)
    #     return shared_params

    # def scale_shared_grads(self, grads):
    #     """
    #     Scale gradients for parameters that are shared since they accumulate
    #     gradients from the critic loss function multiple times
    #     """
    #     # 迭代所有共享的参数，并缩放其梯度
    #     shared_params = self.shared_parameters()
    #     for i, param in enumerate(shared_params):
    #         grads[i] = grads[i] / self.nagents
    #     return grads

    def call(self, inps, agents=None, return_q=True, return_all_q=False, regularize=False, return_attend=False):
        """
        前向传播计算 Q 值与注意力权重
        :param inps: 输入，包括每个代理的状态和动作
        :param agents: 需要计算 Q 值的代理索引
        :param return_q: 是否返回 Q 值
        :param return_all_q: 是否返回所有 Q 值
        :param regularize: 是否计算正则化项
        :param return_attend: 是否返回注意力权重
        :return: 每个代理的 Q 值（以及所有 Q 值、注意力权重等）
        """
        if agents is None:
            agents = range(self.nagents)
        
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [tf.concat([s, a], axis=-1) for s, a in inps]

        # 提取每个代理的状态-动作编码
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        
        # 提取每个代理的状态编码作为 query
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        # 计算注意力机制
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents] for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]

        # 计算每个代理对每个注意力头的注意力权重
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(all_head_keys, all_head_values, all_head_selectors):
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                # 获取其他代理的 key 和 value
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                
                # 计算注意力得分（点积）
                attend_logits = tf.matmul(selector[:, tf.newaxis, :], tf.stack(keys), transpose_b=True)
                scaled_attend_logits = attend_logits / tf.sqrt(tf.cast(keys[0].shape[-1], tf.float32))
                attend_weights = tf.nn.softmax(scaled_attend_logits, axis=-1)

                # 计算加权平均的 value
                # values: list -> [batch_size, dim, num_agents-1]
                values_stack = tf.stack(values, axis=0)  # shape = [num_agents-1, batch, dim]
                values_stack = tf.transpose(values_stack, [1, 2, 0])  # [batch, dim, num_agents-1]
                # attend_weights: [batch, 1, num_agents-1]
                attended = tf.matmul(attend_weights, values_stack)  # shape = [batch, 1, dim]
                other_values = tf.squeeze(attended, axis=1)  # shape = [batch, dim]

                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # 计算 Q 值
        all_rets = []
        for i, a_i in enumerate(agents):
            critic_in = tf.concat([s_encodings[i], *other_all_values[i]], axis=-1)
            all_q = self.critics[a_i](critic_in)
            q = tf.gather(all_q, actions[a_i], axis=-1, batch_dims=1)
            if return_q:
                all_rets.append(q)
            if return_all_q:
                all_rets.append(all_q)
            if regularize:
                # 正则化注意力
                attend_mag_reg = 1e-3 * tf.reduce_mean(
                    tf.stack([tf.reduce_sum(tf.square(logit), axis=-1) for logit in all_attend_logits[i]], axis=0))
                all_rets.append(attend_mag_reg)
            if return_attend:
                all_rets.append(all_attend_probs[i])

        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
