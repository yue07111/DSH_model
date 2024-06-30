from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')


# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)
# code [CV_Project](https://github.com/aarathimuppalla/CV_Project)
# code [DSH_tensorflow](https://github.com/yg33717/DSH_tensorflow)

# choose the dataset and optimizer,set parameters
def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}}, 不同的优化器，不同的迭代方式
        #"optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 10 ** -5}},
        
        "info": "[DSH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        #"net": AlexNet,
        "net":ResNet, # 选择网络
        #"dataset": "cifar10", # 选择数据集
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        "dataset": "coco",
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
        "device": torch.device("cuda:3"),
        "bit_list": [48],
        "pretrain":True,  # 从头训练还是微调
    }
    config = config_dataset(config) # in the utils/tools.py
    return config


# loss ds
class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"]) # the pic feature (train_num,bits)
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"]) # the one-hot label (train_num,n_class)
        self.loss_record = {"loss1": [], "loss2": []}  # 初始化记录字典

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data 
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2) # get euclidean distance (batchsize,train_num)
        y = (y @ self.Y.t() == 0).float() # (batchsize,train_num)

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0) # batch size loss
        loss1 = loss.mean() # 
        loss2 = config["alpha"] * (1 - u.abs()).abs().mean() # L1 Regularization
        
        # 记录损失值
        self.loss_record["loss1"].append(loss1.item())
        self.loss_record["loss2"].append(loss2.item())

        return loss1 + loss2


def train_val(config, bit):
    device = config["device"]
    
    # Load data and set the number of training and testing samples
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    
    # Initialize the network and move it to the specified device
    net = config["net"](bit).to(device)
    

    # Initialize the optimizer with the specified parameters
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    
    # set learning
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    # Define the loss function (DSHLoss)
    criterion = DSHLoss(config, bit)

    Best_mAP = 0  # Variable to keep track of the best mean Average Precision (mAP)
    
    test_loss=[]
    train_losses=[]
    
   
    # Training loop
    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()  # Set the network to training mode

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()  # Clear the gradients of all optimized tensors
            u = net(image)

            # Compute the loss
            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights

        # Compute the average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

        print("\b\b\b\b\b\b\b train_loss:%.3f" % (train_loss))
        
        val_loss=0
        with torch.no_grad():
            for image, label, ind in test_loader:
                image = image.to(device)
                label = label.to(device)
                u = net(image)
                loss = criterion(u, label.float(), ind, config)
                val_loss += loss.item()
    
        val_loss = val_loss / len(test_loader)
        print("\b\b\b\b\b\b\b test_loss:%.3f" % (val_loss))
        test_loss.append(val_loss)

        # Validate the model every 'test_map' epochs
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP, val_loss = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)

    
    # 记录训练的两个loss变化
    loss_save_dir = "log/pre_renew/{}_{}_batch_size_{}_{}_trail_2".format(
            config["dataset"], 
            config["optimizer"]["type"].__name__,
            config["batch_size"],
            bit
        )
    os.makedirs(loss_save_dir, exist_ok=True)
    
    loss_save_path = f"{loss_save_dir}/train_loss_record.json"
    with open(loss_save_path, 'w') as f:
        json.dump(train_losses, f)
    
    test_loss_path=f"{loss_save_dir}/test_loss_record.json"
    with open(test_loss_path, 'w') as f:
        json.dump(test_loss, f)

    print(f"Loss record saved to {loss_save_path}")
    # 绘制训练和测试损失曲线   
    plt.plot(range(1, len(train_losses) + 1),train_losses, label='train Loss')
    plt.plot(range(1, len(test_loss) + 1), test_loss, label='test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(loss_save_dir, f"loss_curve_{bit}.png"))
    print(f"Loss curve saved at: {os.path.join(loss_save_dir, f'loss_curve_{bit}.png')}")
    

if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/ResNet/pre_renewed/DSH_{config['dataset']}_{bit}_pre.json"
        train_val(config, bit)