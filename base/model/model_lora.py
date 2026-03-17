import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LoRA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02) # 高斯初始化
        self.B.weight.data.zero_() # 在不训练的时候和原始W保持一致

    def forward(self, x):
        return self.B(self.A(x))

def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank).to(model.device)
            # 告诉模型这部分是lora
            setattr(module, "lora", lora)
            original_forward = module.forward
            def forward_with_lora(x, layer1 = original_forward, layer2 = lora):
                return layer1(x) + layer2(x)

def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {
        (k[7:] if k.startwith("module.") else k): v for k, v in state_dict.items()
    }
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            lora_state = {
                k.replace(f"{name}.lora", "") : v
                for k, v in state_dict.items()
                if f"{name}.lora." in k
            }
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)

    torch.save(state_dict, path)