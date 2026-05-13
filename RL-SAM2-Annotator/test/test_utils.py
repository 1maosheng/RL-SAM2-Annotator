import unittest
import numpy as np
from PIL import Image
import torch
import sys
import os

# 添加 src 路径以便导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import action_to_coord, calculate_iou, calculate_miou, build_state_tensor, set_grid_size


class TestUtils(unittest.TestCase):

    def setUp(self):
        set_grid_size(16)

    def test_action_to_coord(self):
        # 16x16 网格，图像 128x128
        x, y = action_to_coord(0, 128)  # 第一行第一列
        self.assertAlmostEqual(x, 4.0)  # (0.5)*(128/16)=4
        self.assertAlmostEqual(y, 4.0)

        x, y = action_to_coord(255, 128)  # 最后一行最后一列
        self.assertAlmostEqual(x, 124.0)
        self.assertAlmostEqual(y, 124.0)

        # 边界裁剪
        x, y = action_to_coord(0, 64)  # 64尺寸，边界0-63
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 63)

    def test_calculate_iou(self):
        pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        gt = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        iou = calculate_iou(pred, gt)
        # 交集：位置 (0,0),(1,0) -> 2个；并集：(0,0),(0,1? 0),(1,0),(1,1) -> 3个
        self.assertAlmostEqual(iou, 2 / 3, places=5)

        # 全空
        self.assertEqual(calculate_iou(np.zeros((2, 2)), np.zeros((2, 2))), 1.0)

    def test_calculate_miou(self):
        pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        gt = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        miou = calculate_miou(pred, gt)
        # 缺陷类 IoU = 2/3 ≈0.6667，背景类 IoU：背景预测为 0 的位置：(0,1)和(1,1) =>
        # 预测背景 [0,1]? 实际背景 (0,1)=0 正确，(1,1)=1 错误，背景交=1，并=2 -> 0.5
        # miou = (0.6667+0.5)/2 = 0.58333
        self.assertAlmostEqual(miou, (2 / 3 + 0.5) / 2, places=5)

    def test_build_state_tensor(self):
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        prev_masks = [np.random.rand(200, 200) > 0.5 for _ in range(2)]
        state = build_state_tensor(image, prev_masks, img_size=128, max_hist=3)
        self.assertIsInstance(state, torch.Tensor)
        self.assertEqual(state.shape, (3 + 3, 128, 128))  # RGB+3个历史掩码
        # 检查值域 [0,1]
        self.assertLessEqual(state.max(), 1.0)
        self.assertGreaterEqual(state.min(), 0.0)


if __name__ == '__main__':
    unittest.main()