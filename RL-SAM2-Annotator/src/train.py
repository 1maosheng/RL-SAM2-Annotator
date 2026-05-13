import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .dqn_model import DQN
from .replay_buffer import TrajectoryAwareReplayBuffer
from .env import SAM2InteractionEnv
from .utils import set_grid_size, calculate_miou


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train(config):
    # 设置全局网格大小，供 utils.action_to_coord 使用
    set_grid_size(config['grid_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 SAM2
    sam2_model = build_sam2(config['sam2_config'], config['sam2_checkpoint']).to(device)
    predictor = SAM2ImagePredictor(sam2_model)

    # 读取数据
    img_dir = config['data']['train_image_dir']
    gt_dir = config['data']['train_gt_dir']
    image_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    valid_pairs = []
    for f in image_files:
        gt_name = os.path.splitext(f)[0] + '.png'
        if os.path.exists(os.path.join(gt_dir, gt_name)):
            valid_pairs.append((f, gt_name))
    print(f"找到 {len(valid_pairs)} 对图像-真值")

    # 网络参数
    in_channels = 3 + config['state_history']
    policy_net = DQN(in_channels, config['image_size'], config['grid_size'] ** 2).to(device)
    target_net = DQN(in_channels, config['image_size'], config['grid_size'] ** 2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    memory = TrajectoryAwareReplayBuffer(config['memory_size'])

    steps_done = 0
    best_avg_miou = 0.0
    episode_rewards = []
    episode_mious = []

    # 创建模型保存目录
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)

    for epoch in range(config['epochs']):
        random.shuffle(valid_pairs)
        total_reward = 0.0
        total_miou = 0.0
        epsilon = config['epsilon_end'] + (config['epsilon_start'] - config['epsilon_end']) * np.exp(
            -steps_done / config['epsilon_decay'])

        for img_file, gt_file in tqdm(valid_pairs, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            img_path = os.path.join(img_dir, img_file)
            gt_path = os.path.join(gt_dir, gt_file)
            image = np.array(Image.open(img_path).convert('RGB'))
            gt_mask = np.array(Image.open(gt_path).convert('L')) > 127

            env = SAM2InteractionEnv(
                image, gt_mask, predictor,
                img_size=config['image_size'],
                max_steps=config['max_steps'],
                state_history=config['state_history'],
                penalty_bg=config['penalty_bg'],
                penalty_rep=config['penalty_rep'],
                penalty_dec=config['penalty_dec'],
                min_reward_clip=config['min_reward_clip']
            )
            state = env.reset()
            done = False
            traj_id = memory.next_traj_id
            memory.next_traj_id += 1

            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, config['grid_size'] ** 2 - 1)
                else:
                    with torch.no_grad():
                        q_vals = policy_net(state.unsqueeze(0).to(device))
                        action = q_vals.argmax().item()

                next_state, reward, done, info = env.step(action)
                memory.push(state.cpu(), action, reward, next_state.cpu(), done, traj_id)
                state = next_state
                total_reward += reward
                steps_done += 1

                # 经验回放训练
                if len(memory) >= config['batch_size']:
                    batch = memory.sample(config['batch_size'])
                    if batch:
                        states_b, actions_b, rewards_b, next_states_b, dones_b, _ = zip(*batch)
                        states_b = torch.stack(states_b).to(device)
                        actions_b = torch.tensor(actions_b, dtype=torch.long).to(device)
                        rewards_b = torch.tensor(rewards_b, dtype=torch.float32).to(device)
                        next_states_b = torch.stack(next_states_b).to(device)
                        dones_b = torch.tensor(dones_b, dtype=torch.bool).to(device)

                        current_q = policy_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_q = target_net(next_states_b).max(1)[0]
                            target_q = rewards_b + config['gamma'] * next_q * (~dones_b).float()
                        loss = F.mse_loss(current_q, target_q)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                        optimizer.step()

                if steps_done % config['target_update_freq'] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # 回合结束，标记轨迹质量
            final_miou = info['miou']
            if final_miou >= 0.9:
                label = 'valid'
            elif final_miou >= 0.6:
                label = 'suboptimal'
            else:
                label = 'invalid'
            memory.set_traj_label(traj_id, label)
            total_miou += final_miou
            episode_mious.append(final_miou)

        avg_reward = total_reward / len(valid_pairs)
        avg_miou = total_miou / len(valid_pairs)
        episode_rewards.append(avg_reward)
        print(f"Epoch {epoch + 1}: Reward = {avg_reward:.4f}, mIoU = {avg_miou:.4f}, epsilon = {epsilon:.4f}")
        scheduler.step(avg_miou)

        if avg_miou > best_avg_miou:
            best_avg_miou = avg_miou
            best_path = config['model_save_path'].replace('.pt', '_best.pt')
            torch.save(policy_net.state_dict(), best_path)
            print(f"保存最佳模型，mIoU = {best_avg_miou:.4f}")

        torch.save(policy_net.state_dict(), config['model_save_path'])

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Avg Reward')
    plt.title('Training Rewards')
    plt.subplot(1, 2, 2)
    plt.plot(episode_mious)
    plt.xlabel('Image')
    plt.ylabel('Final mIoU')
    plt.title('Final mIoU per Image')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    print(f"训练完成，最佳平均 mIoU = {best_avg_miou:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config yaml path')
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)