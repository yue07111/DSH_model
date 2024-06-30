## 项目名称
论文《Deep Supervised Hashing for Fast Image Retrieval》复现和改进


## 目录

- [背景](#背景)
- [安装](#安装)
- [功能](#功能)
- [数据下载](#数据下载)
- [使用](#使用)
- [示例](#示例)
- [贡献](#贡献)
- [许可](#许可)
- [联系我们](#联系我们)

## 背景

  论文《Deep Supervised Hashing for Fast Image Retrieval》中提出了一种新的哈希方法来学习紧凑的二进制码，以便在大规模数据集上高效地检索图像。DSH设计了一种CNN架构，它以图像对(相似/不相似)作为训练输入，并鼓励每个图像的输出近似离散值(例如+1/-1)。

  本项目在复现DSH算法的基础上，提供了相关代码，记录训练损失和测试损失随迭代次数的变化过程，统计二值化程度并画出分布图，有助于进一步优化该模型，并且设置了不同的训练优化器和网络模型，使模型的性能有了进一步的提升。

## 安装
```bash
conda env create -f your_env
conda activate your_env

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

## 功能

- 功能1 
直接运行,记录模型和每轮迭代的损失。

 ```bash
python DSH.py
```

- 功能2
展现模型输出的分布
```bash
utils/output_distrubition/alaph_pic.py
```

<img src="https://github.com/yue07111/DSH_model/blob/master/utils/output_distrubition/pic_alpha.png"  alt="Matplotlib Demo"/><br/>
- 功能3
展示PR曲线，需要在config中定义"pr_curve_path"，修改utils/precision_recall/precision_recall_curve.py的pr数据地址
```bash
config["pr_curve_path"] = f"log/alexnet/DSH_{config['dataset']}_{bit}.json"
```

``` bash
python utils/precision_recall/precision_recall_curve.py
```

- 功能4
```bash
python demo.py
```



## 数据下载
使用DPCHash_Baselines库中data文件夹中为各个数据集已经划分好的data list，对应的图像集需要自行下载，并在utils/tools.py中修改你的数据集所存放位置的地址。



### 克隆项目

```bash
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名
