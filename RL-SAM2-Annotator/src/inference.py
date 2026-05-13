import torch
import numpy as np
from PIL import Image
import argparse
import yaml

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from .dqn_model import DQN
from .utils import build_state_tensor, action_to_coord, set_grid_size


def auto_annotate(model_path, image_path, predictor, img_size, grid_size, state_history, max_steps):
    set_grid_size(grid_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3 + state_history
    n_actions = grid_size * grid_size
    model = DQN(in_channels, img_size, n_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载原始图像
    image_orig = np.array(Image.open(image_path).convert('RGB'))
    h_orig, w_orig = image_orig.shape[:2]
    image_resized = np.array(Image.fromarray(image_orig).resize((img_size, img_size)))

    predictor.set_image(image_resized)

    prev_masks = [np.zeros((img_size, img_size), dtype=np.float32)]
    step_count = 0
    done = False
    final_mask = None

    with torch.no_grad():
        while not done:
            state = build_state_tensor(image_resized, prev_masks, img_size, state_history)
            q_vals = model(state.unsqueeze(0).to(device))
            action = q_vals.argmax().item()
            x, y = action_to_coord(action, img_size)
            point_coords = np.array([[x, y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)
            masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels,
                                            multimask_output=False)
            new_mask = masks[0]
            prev_masks.append(new_mask)
            step_count += 1
            done = (step_count >= max_steps)
            final_mask = new_mask

    # 恢复原始尺寸
    final_mask_orig = np.array(Image.fromarray((final_mask * 255).astype(np.uint8)).resize((w_orig, h_orig))) / 255.0
    return final_mask_orig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/default.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 加载 SAM2
    sam2_model = build_sam2(config['sam2_config'], config['sam2_checkpoint'])
    predictor = SAM2ImagePredictor(sam2_model)

    mask = auto_annotate(
        args.model, args.image, predictor,
        img_size=config['image_size'],
        grid_size=config['grid_size'],
        state_history=config['state_history'],
        max_steps=config['max_steps']
    )

    # 保存掩码
    out_path = args.image.replace('.', '_mask.png')
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    print(f"Saved mask to {out_path}")