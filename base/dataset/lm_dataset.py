import json
import torch.utils.data as Dataset
import torch
import os
# 禁用tokenizers并行加速
os.environ["TOKRNIZERS_PARALLELISM"] = "false"
# 先写dataset类
# ──────────────────────────────────────────────────────────────────────────────
# 1. PretrainDataset —— 自回归预训练数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：Next-Token Prediction（下一个 token 预测）
# 数据格式：{"text": "一段原始文本"}
# 训练特点：
#   - 模型对整段文本的每个位置都进行预测，没有"只学回复"的区分。
#   - 使用 BOS/EOS 标记文本边界，让模型学会文本的起止。
#   - PAD token 对应的 label 置 -100，不参与 loss 计算，节省无效梯度。
#   - labels 直接 clone 自 input_ids（即 X 和 Y 错位一格：Y[t] = X[t+1]）。
# ──────────────────────────────────────────────────────────────────────────────
class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_path = data_path
        self.samples = self.load_data(data_path)

    # 实现dataset内定的方法
    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r',encoding="utf-8") as f:
            for line_num, line in enumerate(f,1):
                data = json.loads(line.strip())
                samples.append(data)

        return samples
# len
    def __len__(self):
        return len(self.samples)
# getitem
    def __getitem__(self, index):
        sample = self.samples[index]
        encoding = self.tokenizer(str(sample['text']), max_length=self.max_len, padding="max_length", truncation=True, return_tensors='pt')
        # (max_length,)
        input_ids = encoding['input_ids'].squeeze()

        loss_mask = input_ids!=self.tokenizer.pad_token_id
        #自回归
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(input_ids[1:], dtype=torch.long)

        loss_mask = torch.tensor(loss_mask[:-1], dtype=torch.long)

        return x,y,loss_mask

