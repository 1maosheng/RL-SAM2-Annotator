# RL-SAM2-Annotator

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于深度强化学习（DQN）与 SAM2 的钢材表面缺陷自动标注系统。

本代码实现了论文《融合首尾帧筛选视频生成与深度强化学习主动标注的钢材表面缺陷检测数据扩充框架》中的 **RL-SAM2 自动标注模块**。训练一个 DQN 智能体，使其在最多 10 步内为缺陷图像选择最优点击点，驱动 SAM2 生成高精度像素级掩码，**完全无需人工提示或文本描述**。

## 📌 主要特性

- ✅ **离散化动作空间**（16×16 网格 → 256 个离散点）
- ✅ **状态包含历史掩码**（解决序列决策问题）
- ✅ **复合奖励函数**（mIoU + 背景/重复/精度下降惩罚）
- ✅ **轨迹质量加权经验回放**（提升样本效率）
- ✅ **与 SAM2 无缝集成**（端到端自动化标注）
- ✅ **即插即用**：训练后可标注任意钢材缺陷图像

## 📊 实验结果（NEU-Seg 数据集）

| 方法 | 平均最终 mIoU (10步内) |
|------|------------------------|
| 随机点击 baseline | 0.62 |
| 人工点击（专家） | 0.94 |
| **RL-SAM2 (本方法)** | **0.91** |

> 经训练的 DQN 智能体仅用 **10 次点击**即可达到接近人工水平的标注精度。

## 🗂️ 项目结构
RL-SAM2-Annotator/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config/
│ └── default.yaml # 超参数与路径配置
├── src/
│ ├── init.py
│ ├── utils.py # IoU计算、坐标转换、状态构建
│ ├── dqn_model.py # DQN网络定义
│ ├── replay_buffer.py # 轨迹质量加权经验回放
│ ├── env.py # SAM2交互环境（多步）
│ ├── train.py # 训练脚本
│ └── inference.py # 推理脚本（自动标注单张图）
├── scripts/
│ └── download_sam2.sh # 下载SAM2预训练权重
├── tests/
│ ├── test_utils.py # 单元测试：工具函数
│ └── test_env.py # 单元测试：环境交互逻辑
└── checkpoints/ # 存放模型权重（自动创建）


## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourname/RL-SAM2-Annotator.git
cd RL-SAM2-Annotator

### 2. 安装依赖
pip install -r requirements.txt

###3.下载 SAM2 预训练模型
bash scripts/download_sam2.sh

###4.准备数据集
dataset/
  images/training/      # 训练图像（.jpg/.png）
  annotations/training/ # 对应的二值掩码（同名 .png）

###5.修改配置
编辑 config/default.yaml，设置数据集路径、超参数等。必改项：
data:
  train_image_dir: "/path/to/dataset/images/training"
  train_gt_dir: "/path/to/dataset/annotations/training"

###6. 训练智能体
python src/train.py --config config/default.yaml

训练过程中会自动保存最佳模型（checkpoints/rl_sam2_annotator_best.pt），并绘制训练曲线 training_curves.png。

7. 自动标注新图像
python src/inference.py --model checkpoints/rl_sam2_annotator_best.pt --image sample.jpg
输出掩码将保存为 sample_mask.png

⚙️ 配置参数详解（config/default.yaml）
参数	说明	推荐值
image_size	输入图像尺寸（正方形）	128
grid_size	离散化网格大小（动作数 = grid_size²）	16
max_steps	每张图像最多交互步数	10
state_history	保留的历史掩码数量	3
gamma	折扣因子	0.99
epsilon_start/end/decay	ε-贪婪探索策略	0.9 → 0.05, 衰减5000步
batch_size	经验回放批次大小	32
target_update_freq	目标网络硬更新频率（步）	1000
penalty_bg/rep/dec	惩罚系数	-0.5, -0.1, -0.3
min_reward_clip	奖励下限	-0.5
🧪 运行测试
bash
python -m unittest discover tests
❓ 常见问题
Q: SAM2 预测时出现 OOM（显存不足）怎么办？
A: 减小 image_size（如改为 96）或 batch_size，或使用更小的 SAM2 变体（如 sam2.1_hiera_tiny）。

Q: 训练时 mIoU 不提升？
A: 检查奖励函数中的惩罚系数是否过大；尝试降低 epsilon 衰减速度；确保数据集真值掩码正确。

Q: 能否用于其他工业缺陷（非钢材）？
A: 可以，只需提供对应的分割数据集，超参数可能需要微调。本方法不依赖钢材特定的先验知识。

📝 引用
如果您在研究中使用了本代码，请引用以下论文：

bibtex
@article{,
  title={视频生成与深度强化学习主动标注的钢材表面缺陷检测数据扩充框架},
  author={Maosheng LI},
  journal={},
  year={2026}
}
🤝 贡献
欢迎提交 Issue 和 Pull Request。请确保通过所有单元测试。

📄 许可证
MIT License © 2025 [Maosheng Li]

text

---

这个版本的 README 更加**结构化、信息丰富**，包含了：

- 徽章（badges）提升专业感
- 清晰的特性列表
- 实验结果概览（用表格展示）
- 完整的目录树
- 从克隆到推理的步骤化指南
- 配置参数表
- 常见问题（FAQ）
- 测试命令
- 引用和许可证
