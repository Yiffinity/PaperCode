import tensorflow as tf
from tensorflow.keras import layers

class CriticAttention(layers.Layer):
    def __init__(self, state_dim, hidden_dim=128, num_heads=2):
        '''
        :param state_dim: 观测维度
        '''
        super(CriticAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.state_dim = state_dim
        self.input_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 编码器：用于将 除当前agent外的state 生成key
        self.critic_encoder = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1, center=False, scale=False),
            layers.Dense(hidden_dim),
            layers.LeakyReLU()
        ])

        # 状态编码器：仅用于生成 query 向量
        self.state_encoder = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1, center=False, scale=False),
            layers.Dense(hidden_dim),
            layers.LeakyReLU()
        ])

        # 多头注意力模块
        self.key_extractors = [layers.Dense(self.head_dim, use_bias=False) for _ in range(num_heads)]
        self.selector_extractors = [layers.Dense(self.head_dim, use_bias=False) for _ in range(num_heads)]
        self.value_extractors = [tf.keras.Sequential([
            layers.Dense(self.head_dim, use_bias=False),
            layers.LeakyReLU()
        ]) for _ in range(num_heads)]

    def call(self, state, agent_index,agents=None):
        """
        :param state: Tensor of shape [batch, n_agents, obs_dim]
        state:所有智能体的状态 / 观测 形状是[batch, n_agents, obs_dim]
        :当前agent的索引
        :包含的agents的范围
        """
        # 获取当前智能体的状态作为 query
        current_agent_state = state[:, agent_index, :]  # [B, state_dim], 当前智能体的状态
        s_encoded = self.state_encoder(current_agent_state)  # [B, hidden_dim], 当前智能体的 query
        # 获取其他智能体的状态作为 key 和 value（当前智能体外的智能体）
        other_agents_state = tf.concat([state[:, :agent_index, :], state[:, agent_index+1:, :]], axis=1)  # [B, N-1, state_dim]
        # 通过 critic_encoder 编码每个智能体的 state（生成 key 和 value）
        sa_encoded = self.critic_encoder(other_agents_state)  # [B, N-1, hidden_dim], 用于生成 key 和 value

        head_outputs = []

        for i in range(self.num_heads):
            key = self.key_extractors[i](sa_encoded)
            value = self.value_extractors[i](sa_encoded)
            query = self.selector_extractors[i](s_encoded)
            query = tf.expand_dims(query, 1)  # [B, 1, hidden_dim]

            # Attention scores: Q x K^T
            scores = tf.matmul(query, key, transpose_b=True)
            scores /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))  # scaled dot-product
            attn_weights = tf.nn.softmax(scores, axis=-1)

            # context = softmax(QK^T / sqrt(d)) * V
            attended = tf.matmul(attn_weights, value)
            head_outputs.append(attended)

        # 多头拼接输出：当前智能体的注意力加权信息，它反映了当前智能体对其他智能体状态的关注程度
        out = tf.concat(head_outputs, axis=-1)  # [B, 1, hidden_dim]
        attention_output = tf.squeeze(out, axis=1)  # [batch_size, hidden_dim]

        return attention_output  # [B, 1, hidden_dim]
    
    

    '''获得注意力机制中多头注意力的值'''
    def get_attention_heads(self, state, agent_index):
        """
        获取每一个注意力头的单独输出
        输入：
          - state: [batch_size, n_agents, obs_dim]
          - agent_index: 当前智能体的索引
        输出：
          - list，每个元素是对应 head 的输出 [batch_size, head_dim]
        """
        current_agent_state = state[:, agent_index, :]  # [B, state_dim]
        s_encoded = self.state_encoder(current_agent_state)  # [B, hidden_dim]

        other_agents_state = tf.concat([state[:, :agent_index, :], state[:, agent_index+1:, :]], axis=1)
        sa_encoded = self.critic_encoder(other_agents_state)  # [B, N-1, hidden_dim]

        head_outputs = []

        for i in range(self.num_heads):
            key = self.key_extractors[i](sa_encoded)  # [B, N-1, head_dim]
            value = self.value_extractors[i](sa_encoded)  # [B, N-1, head_dim]
            query = self.selector_extractors[i](s_encoded)  # [B, head_dim]
            query = tf.expand_dims(query, 1)  # [B, 1, head_dim]

            scores = tf.matmul(query, key, transpose_b=True)  # [B, 1, N-1]
            scores /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
            attn_weights = tf.nn.softmax(scores, axis=-1)  # [B, 1, N-1]

            attended = tf.matmul(attn_weights, value)  # [B, 1, head_dim]
            attended = tf.squeeze(attended, axis=1)  # [B, head_dim]

            head_outputs.append(attended)  # 单独存每个head

        return head_outputs
