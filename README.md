## 项目名称
论文《Deep Supervised Hashing for Fast Image Retrieval》复现和改进


## 目录

- [背景](#背景)
- [功能](#功能)
- [安装](#安装)
- [使用](#使用)
- [示例](#示例)
- [贡献](#贡献)
- [许可](#许可)
- [联系我们](#联系我们)

## 背景

  论文《Deep Supervised Hashing for Fast Image Retrieval》中提出了一种新的哈希方法来学习紧凑的二进制码，以便在大规模数据集上高效地检索图像。DSH设计了一种CNN架构，它以图像对(相似/不相似)作为训练输入，并鼓励每个图像的输出近似离散值(例如+1/-1)。

  本项目在复现DSH算法的基础上，提供了相关代码，记录训练损失和测试损失随迭代次数的变化过程，统计二值化程度并画出分布图，有助于进一步优化该模型，并且设置了不同的训练优化器和网络模型，使模型的性能有了进一步的提升。
## 功能

主要功能和特点：

- 功能1
- 功能2
- 功能3

## 安装和执行
```bash
conda env create -f your_env
conda activate your_env

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```





### 克隆项目

```bash
git clone https://github.com/你的用户名/你的项目名.git
cd 你的项目名
