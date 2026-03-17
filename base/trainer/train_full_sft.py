"""
MokioMind全参数监督微调(Full SFT)脚本
用于在预训练模型基础上进行指令跟随训练

📚 监督微调(SFT)知识点：
- 训练目标：让模型学会遵循人类指令
- 数据格式：问答对、对话数据、任务指令
- 训练特点：在预训练模型基础上继续训练
- 关键区别：只对答案部分计算损失，问题部分用mask屏蔽

📚 Full SFT vs LoRA微调：
- Full SFT：更新所有模型参数，效果更好但需要更多资源
- LoRA：只更新少量参数，效率高但效果略差
- 选择原则：资源充足选Full SFT，资源有限选LoRA
"""

import os
import sys

# 📚 Python模块系统和路径管理
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import time  # 时间测量
import warnings  # 警告控制
import torch  # PyTorch深度学习框架
import torch.distributed as dist  # 分布式训练
from contextlib import nullcontext  # 上下文管理器
from torch import optim, nn  # 优化器和神经网络
from torch.nn.parallel import DistributedDataParallel  # 分布式并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# MokioMind相关组件
from model.MokioModel import MokioMindConfig  # 模型配置
from dataset.lm_dataset import SFTDataset  # SFT数据集
from trainer.trainer_utils import (  # 训练工具
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行单个SFT训练轮次
    与预训练的主要区别在于数据格式和损失计算方式

    📚 SFT训练关键知识点：
    1. 数据格式：[instruction + response]格式的问答对
    2. 损失掩码：只对response部分计算损失，instruction部分mask掉
    3. 损失函数：仍然是交叉熵，但通过mask控制计算范围
    4. 训练目标：让模型学会根据指令生成合适的回答

    📚 SFT vs 预训练的区别：
    - 预训练：所有token都计算损失，学习语言的统计规律
    - SFT：只对回答部分计算损失，学习指令跟随能力

    Args:
        epoch: 当前训练轮次
        loader: SFT数据加载器
        iters: 总迭代次数
        start_step: 起始步数（断点续训用）
        wandb: 实验跟踪工具
    """
    # 交叉熵损失函数，不进行自动降维
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    # 遍历SFT数据批次
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 📚 SFT数据结构说明：
        # X: 输入序列 [batch_size, seq_len] (instruction + response)
        # Y: 目标序列 [batch_size, seq_len] (X向右偏移一位)
        # loss_mask: 损失掩码 [batch_size, seq_len] (只有response部分为1)

        # 将数据移动到GPU
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 📚 学习率调度（与预训练相同）
        # 使用余弦退火策略动态调整学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 📚 混合精度前向传播
        with autocast_ctx:
            # 模型前向传播
            res = model(X)

            # 📚 SFT损失计算详解：
            # 1. 将3D logits转为2D: [batch*seq, vocab_size]
            # 2. 将2D targets转为1D: [batch*seq]
            # 3. 计算每个位置的交叉熵损失
            # 4. 恢复为原始形状: [batch_size, seq_len]
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),  # 展平logits
                Y.view(-1)  # 展平targets
            ).view(Y.size())  # 恢复形状

            # 📚 SFT核心：掩码损失计算
            # 只对response部分（loss_mask=1）计算损失
            # instruction部分（loss_mask=0）被忽略
            # 这是SFT与预训练最重要的区别！
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            # 添加MoE辅助损失（如果使用MoE架构）
            loss += res.aux_loss

            # 梯度累积：模拟更大的batch size
            loss = loss / args.accumulation_steps

        # 📚 混合精度反向传播
        scaler.scale(loss).backward()

        # 📚 梯度累积和参数更新
        if (step + 1) % args.accumulation_steps == 0:
            # 恢复梯度真实值
            scaler.unscale_(optimizer)
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行参数更新
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，准备下一轮累积
            optimizer.zero_grad(set_to_none=True)

        # 📚 训练日志记录
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:')

            # 记录到实验跟踪系统
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 📚 SFT模型检查点保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式

            # 构建SFT模型保存路径
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 处理分布式模型的状态字典
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 半精度保存节省空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            # 保存完整训练状态
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()  # 恢复训练模式


if __name__ == "__main__":
    """
    SFT主函数：监督微调脚本的入口点

    📚 SFT与预训练的参数差异：
    - 学习率更小：5e-7 vs 5e-4（预训练）
    - 训练轮数较少：2轮 vs 多轮
    - batch_size可以更小：已有基础能力，不需要大batch
    - 累积步数通常为1：SFT数据质量高，不需要太多累积
    """

    parser = argparse.ArgumentParser(description="MokioMind Full SFT")

    # ========== 基础训练参数 ==========
    parser.add_argument("--save_dir", type=str, default="../out",
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str,
                        help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2,
                        help="训练轮数（SFT通常2-5轮即可）")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size（SFT可以使用较小的batch）")

    # 📚 SFT学习率设置知识点
    # SFT学习率通常比预训练小1-2个数量级
    # 因为模型已经有了基础能力，只需要微调
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="初始学习率（比预训练小很多）")

    # ========== 硬件配置 ==========
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="数据加载线程数")

    # ========== 训练策略 ==========
    # 📚 SFT梯度累积知识点
    # SFT数据质量高，通常不需要大量梯度累积
    # accumulation_steps=1 意味着每个batch都更新参数
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="梯度累积步数（SFT通常设为1）")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="模型保存间隔")

    # ========== 模型架构参数 ==========
    parser.add_argument('--hidden_size', default=512, type=int,
                        help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int,
                        help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int,
                        help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1],
                        help="是否使用MoE架构（0=否，1=是）")

    # ========== SFT数据和恢复参数 ==========
    # 📚 SFT数据路径知识点
    # SFT数据通常是结构化的问答对或对话数据
    # 包含instruction和response两部分
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl",
                        help="SFT训练数据路径")

    # 📚 SFT权重继承知识点
    # SFT通常从预训练模型开始，而不是从零开始
    # 'pretrain'表示从预训练权重开始微调
    parser.add_argument('--from_weight', default='pretrain', type=str,
                        help="基于哪个权重训练（通常从预训练权重开始）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1],
                        help="是否自动检测&续训（0=否，1=是）")

    # ========== 实验跟踪 ==========
    parser.add_argument("--use_wandb", action="store_true",
                        help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MokioMind-Full-SFT",
                        help="wandb项目名")

    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MokioMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight,
                             save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MokioMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:  # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)