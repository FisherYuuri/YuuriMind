class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 残差连接模式：先做LayerNorm -> Attention -> 残差相加 -> LayerNorm -> FFN -> 残差相加
        # 保存残差以供后续相加
        residual = hidden_states

        # 注意力子层：输入先归一化（RMSNorm），返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        # 注意力输出与残差相加
        hidden_states = hidden_states + residual

        # 前馈子层（post-attention layernorm）并相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

