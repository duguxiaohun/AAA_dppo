import torch

# 替换为你实际的模型文件路径
ckpt_path = "/home/codon/github/第二篇/log/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt"

# 加载 checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")

# 打印 checkpoint 顶层的 key
print("Checkpoint keys:", ckpt.keys())

# 如果包含 'model' 或 'ema'，进一步查看每个参数名和形状
if 'model' in ckpt:
    print("\n--- model 参数 ---")
    for k, v in ckpt['model'].items():
        print(f"{k}: {v.shape}")
elif 'ema' in ckpt:
    print("\n--- ema 参数 ---")
    for k, v in ckpt['ema'].items():
        print(f"{k}: {v.shape}")
else:
    print("\n[!] 未发现 'model' 或 'ema' 键，可能不是标准模型 checkpoint")
    print("所有键如下：")
    for k in ckpt.keys():
        print(f"{k}: type = {type(ckpt[k])}")
