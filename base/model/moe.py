class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 每个token分配到的专家数量
        self.top_k = config.num_experts_per_tok
        # 路由专家数量
        self.n_routed_experts = config.n_routed_experts
        # 计算分数函数
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数，用于判断是否需要负载均衡
        self.alpha = config.aux_loss_alpha
        # aux_loss 是如何统计的，按seq序列还是token批次
        self.seq_aux = config.seq_aux

        # 局部归一化
        self.norm_topk_prob = config.norm_topk_prob
        # 门控维度
        self.gating_dim = config.hidden_size
        # 参数，维度为路由专家数*门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming初始化，也叫He初始化，高效初始化权重
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [bsz, seq_len, hidden]
        bsz, seq_len, h = hidden_states.shape

        # 将序列和batch合并，方便做同一个线性投影，计算简便
        # view(-1, h) -> [bsz * seq_len, hidden]
        hidden_states = hidden_states.view(-1, h)

        # 使用线性变换计算每个token对每个专家的logits
        # F.linear(input, weight, bias) 等价于 input @ weight.T + bias
        logits = F.linear(hidden_states, self.weight, None)  # [bsz*seq_len, n_routed_experts]

        # 得分函数：softmax是常用选择
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # top-k选择: topk_weight是概率，topk_idx是对应的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果top-k>1且需要归一化（数值稳定性），对topk概率做局部归一化
        if self.top_k > 1 and self.norm_topk_prob:
            # 计算这k个概率综合，1e-20是防止除以0
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 执行归一化
            topk_weight = topk_weight / denominator

        # 计算辅助负载均衡损失（aux loss），仅在训练且alpha>0时计算
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 用于aux loss的索引变形为 [bsz, seq_len*topk]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 序列级别的负载统计：统计每个专家被选到的次数
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_用于累加计数：把1累加到对应专家的位置上
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                # 归一化计数，使得期望值可比较
                ce = ce.div(seq_len * aux_topk / self.n_routed_experts)
                # aux_loss = alpha * mean_over_batch( sum( ce * mean(scores_for_token_over_seq) ) )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # token级别的负载统计
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 实际被选中次数/总次数，代表专家 j 在所有 Top-K 选择中出现的频率
                ce = mask_ce.float().mean(0)
                # 平均好感度，Gate多偏向某个专家
                Pi = scores_for_aux.mean(0)
                # fi 是一个归一化后的“负载因子”。
                # fi[j] == 1.0：完美平衡
                # fi[j] > 1.0：过载（被选中的次数超过了平均值）
                # fi[j] < 1.0：负载不足
                fi = ce * self.n_routed_experts
                # 计算辅助损失，用于下一次调整
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss


class MoEFeedForaward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList(
            [FeedForward(config)
             for _ in range(config.n_routed_experts)]
        )
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(config)
                 for _ in range(config.n_shared_experts)]
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制旋转专家
        topk_weight, topk_idx, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=torch.float32)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = (y.view(*topk_weight.shepe, -1) * topk_weight.unsqueeze(-1).sum(dim=1))
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache