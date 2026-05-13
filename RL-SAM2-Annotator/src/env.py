import numpy as np
from PIL import Image
from .utils import action_to_coord, calculate_miou, build_state_tensor


class SAM2InteractionEnv:
    """单张图像的多步交互环境"""

    def __init__(self, image_rgb, gt_mask_binary, predictor, img_size, max_steps, state_history,
                 penalty_bg, penalty_rep, penalty_dec, min_reward_clip):
        self.orig_image = image_rgb
        self.gt_orig = gt_mask_binary
        self.predictor = predictor
        self.img_size = img_size
        self.max_steps = max_steps
        self.state_history = state_history
        self.penalty_bg = penalty_bg
        self.penalty_rep = penalty_rep
        self.penalty_dec = penalty_dec
        self.min_reward_clip = min_reward_clip

        # 缩放图像和真值
        self.image_resized = None
        self.gt_resized = None
        self._resize()

        self.step_count = 0
        self.prev_masks = []  # 存储历史掩码
        self.history_points = []  # 存储历史点击坐标
        self.last_mask = None
        self.done = False

    def _resize(self):
        h, w = self.orig_image.shape[:2]
        if h != self.img_size or w != self.img_size:
            self.image_resized = np.array(Image.fromarray(self.orig_image).resize((self.img_size, self.img_size)))
            self.gt_resized = np.array(
                Image.fromarray((self.gt_orig * 255).astype(np.uint8)).resize((self.img_size, self.img_size))) / 255.0
            self.gt_resized = (self.gt_resized > 0.5).astype(np.float32)
        else:
            self.image_resized = self.orig_image.copy()
            self.gt_resized = self.gt_orig.copy()
        self.predictor.set_image(self.image_resized)

    def reset(self):
        self.step_count = 0
        self.prev_masks = []
        self.history_points = []
        init_mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        self.prev_masks.append(init_mask)
        self.last_mask = init_mask
        self.done = False
        state = build_state_tensor(self.image_resized, self.prev_masks, self.img_size, self.state_history)
        return state

    def step(self, action_idx):
        if self.done:
            raise ValueError("环境已终止，请 reset")

        x, y = action_to_coord(action_idx, self.img_size)

        # 惩罚项
        bg_penalty = self.penalty_bg if self.gt_resized[y, x] < 0.5 else 0.0
        rep_penalty = 0.0
        for (hx, hy) in self.history_points:
            if np.hypot(x - hx, y - hy) < 5:
                rep_penalty = self.penalty_rep
                break

        # SAM2 预测
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
        new_mask = masks[0]

        cur_miou = calculate_miou(new_mask, self.gt_resized)
        prev_miou = calculate_miou(self.last_mask, self.gt_resized) if self.last_mask is not None else 0.0
        dec_penalty = self.penalty_dec if (cur_miou < 0.95 * prev_miou) else 0.0

        reward = cur_miou + bg_penalty + rep_penalty + dec_penalty
        reward = max(reward, self.min_reward_clip)

        # 更新状态
        self.prev_masks.append(new_mask)
        self.history_points.append((x, y))
        self.last_mask = new_mask
        self.step_count += 1
        done = (self.step_count >= self.max_steps) or (cur_miou >= 0.95)
        self.done = done

        next_state = build_state_tensor(self.image_resized, self.prev_masks, self.img_size, self.state_history)
        info = {'miou': cur_miou, 'point': (x, y), 'steps': self.step_count}
        return next_state, reward, done, info