import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from utils.tools import get_data,compute_result,config_dataset
from network import *
import torch
import torch.optim as optim
def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}}, 不同的优化器，不同的迭代方式
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        #"optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 10 ** -5}},
        
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet, # 选择网络
        "dataset": "cifar10", # 选择数据集
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 250,
        "test_map": 25,
        "save_path": "save/DSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [12],
        "pretrain":True,  # 从头训练还是微调
    }
    config = config_dataset(config) # in the utils/tools.py
    return config

def get_test_data():
    config = get_config()
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    
    # 加载保存的模型权重
    model_path = '/disks/sata4/huangyue/hy_DSH/DeepHash-hw/save/DSH/pretrain_mode/cifar10_RMSprop_12bits_0.1_0.7629493901523681/model.pt'
    num_bits = 12  # 根据你的模型定义
    net = AlexNet(num_bits,config["pretrain"]).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    #criterion = DSHLoss(config, num_bits)
    accum_loss=0
    outputs = []
    
    with torch.no_grad():
        for image, label, ind in test_loader:
            image = image.to(device)
            label = label.to(device)
            u = net(image)
            # loss = criterion(u, label.float(), ind, config)
            # accum_loss += loss.item()
            outputs.append(u.cpu().numpy())
            
    # accum_loss /= len(test_loader)
    # print(f'val loss: {accum_loss:.4f}')
            
    return np.concatenate(outputs)
        
output=get_test_data()
# 画统计图
combined_array = np.concatenate(output, axis=None)

# 绘制直方图
plt.figure()
plt.hist(combined_array, bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title('Distribution of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
save_path = f"/disks/sata4/huangyue/hy_DSH/DeepHash-hw/utils/0.1_12bits.png"
plt.savefig(save_path)
print(f"Loss curve saved at: {save_path}")

   