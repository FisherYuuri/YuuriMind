from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话前处理：以一定概率随机插入 system 消息。

    特点：
    - 只有当首条消息不是 system 角色时才可能插入。
    - add_system_ratio 控制插入概率（默认 20%），引入随机性可提升模型
      对有/无 system prompt 两种情况的泛化能力。
    - system 内容从预定义的中英文 prompt 池中随机抽取，覆盖不同表达风格。
    """
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    """
    对话后处理：清理模板渲染后多余的空 <think> 块。

    特点：
    - 针对带 CoT（chain-of-thought）格式的模型，apply_chat_template 有时会
      渲染出 "<think>\n\n</think>\n\n" 这样的空思考块占位符。
    - 大部分情况下（概率 1 - empty_think_ratio = 95%）直接删除该空块，
      防止模型学到"无意义思考"的坏习惯。
    - 保留少量空思考块（empty_think_ratio = 5%），让模型也能处理该边界情况。
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

# ──────────────────────────────────────────────────────────────────────────────
# 2. SFTDataset —— 有监督微调（Supervised Fine-Tuning）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
# 数据格式：{"conversations": [{"role": "user"/"assistant"/"system", "content": "..."}]}
# 训练特点：
#   - 通过 generate_labels 扫描 bos_id（assistant 回复起始标记）定位每段回复，
#     仅将 assistant 回复的 token 位置设为有效 label，其余全部为 -100。
#   - 这样做的意义：让 loss 只反映模型对"正确回答"的拟合，不浪费梯度在
#     用户输入的复现上（用户输入只作为 context，不是预测目标）。
#   - 支持 function calling：若 system 消息携带 "functions" 字段，
#     会透传给 apply_chat_template，生成带工具描述的提示词。
#   - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
# ──────────────────────────────────────────────────────────────────────────────
class SFT_Dataset(Dataset):
    def __init__(self,jsonl_path,tokenizer,max_len=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = load_dataset("json", data_files = jsonl_path, split="train")
        self.bos_id = tokenizer(f"{tokenizer.bos_token_id}assistant\n",add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token_id}\n",add_special_tokens=False).input_ids

    def create_chat_prompt(self,conversations):
        messages = conversations.copy()
        tools = (
            conversations[0]["function"]
            if (
                conversations
                and conversations[0].get("role") != "system"
                and conversations[0].get("function")
            ) else None
        )

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt = False, tools = tools)

    def generate_labels(self, inputs_ids):
        # 让所有inputs_ids都为-100
        # -100是交叉熵损失函数默认忽视的值
        labels = [-100] * len(inputs_ids)
        i = 0
        # 滑动窗口，寻找回答部分，并恢复原本label
        while i < len(inputs_ids):
            if inputs_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i+len(self.bos_id)
                end = start
                while end < len(inputs_ids):
                    if inputs_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start, min(end+len(self.eos_id), len(inputs_ids))):
                    labels[j] = inputs_ids[j]

                i = end+len(self.eos_id) if end < len(inputs_ids) else len(inputs_ids)
        return labels


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 是否随机插入system_prompt
        conversations = pre_processing_chat(sample["conversations"])
        # 用chat_template把对话转换成文本
        prompt = self.create_chat_prompt(conversations)

        # 清理空think
        prompt = post_processing_chat(prompt)
        # toknize截断，并补足pad
        inputs_ids = self.tokenizer(prompt).input_ids[:self.max_len]
        inputs_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(inputs_ids))
        # 生成标签，只让assistant参与loss计算
        labels = self.generate_labels(inputs_ids)
        return torch.tensor(inputs_ids,dtype=torch.long), torch.tensor(labels, dtype=torch.long)

