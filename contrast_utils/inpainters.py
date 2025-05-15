import cv2
import numpy as np
import os
import torch
import yaml
from omegaconf import OmegaConf

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'inpaint_anything', 'lama'))
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_tensor_to_modulo

from .utils import dilate_mask


_LAMA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'inpaint_anything', 'lama', 'configs', 'prediction', 'default.yaml')
_LAMA_CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'big-lama')

_DILATE_SIZE = 5


class BaseInpainter:
    def inpaint(self, image, mask, excluded_mask=None):
        image = image.copy()
        inpainted_mask = mask | excluded_mask if excluded_mask is not None else mask
        inpainted_mask = dilate_mask(inpainted_mask, _DILATE_SIZE)
        inpainted_image = self.inpaint_mask(image, inpainted_mask)
        
        fill_mask = dilate_mask(mask, _DILATE_SIZE) & ~excluded_mask \
                    if excluded_mask is not None else dilate_mask(mask, _DILATE_SIZE)
        image[fill_mask] = inpainted_image[fill_mask]
        return image


class TeleaInpainter(BaseInpainter):
    def __init__(self):
        super().__init__()

    def inpaint_mask(self, image, mask):
        return cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)


class NavierStokesInpainter(BaseInpainter):
    def __init__(self):
        super().__init__()

    def inpaint_mask(self, image, mask):
        return cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_NS)


class LamaInpainter(BaseInpainter):
    def __init__(self):
        super().__init__()

        config_p = _LAMA_CONFIG_PATH
        ckpt_p = _LAMA_CKPT_PATH
        predict_config = OmegaConf.load(config_p)
        predict_config.model.path = ckpt_p
        device = torch.device('cuda')

        train_config_path = os.path.join(
            predict_config.model.path, 'config.yaml')

        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(
            predict_config.model.path, 'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config, checkpoint_path, strict=False, map_location='cpu')
        model.freeze()
        model.to(device)
        
        self.predict_config = predict_config
        self.device = device
        self.lama_model = model
    
    def inpaint_mask(self, image, mask):
        assert len(mask.shape) == 2
        if np.max(mask) == 1:
            mask = mask * 255
        img = torch.from_numpy(image).float().div(255.)
        mask = torch.from_numpy(mask).float()

        batch = {}
        mod = 8
        batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
        batch['mask'] = mask[None, None]
        unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
        batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
        batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
        batch = move_to_device(batch, self.device)
        batch['mask'] = (batch['mask'] > 0) * 1

        batch = self.lama_model(batch)
        cur_res = batch[self.predict_config.out_key][0].permute(1, 2, 0)
        cur_res = cur_res.detach().cpu().numpy()

        if unpad_to_size is not None:
            orig_height, orig_width = unpad_to_size
            cur_res = cur_res[:orig_height, :orig_width]

        return np.clip(cur_res * 255, 0, 255).astype('uint8')


def build_inpainter(mode):
    if mode == 'telea':
        inpainter = TeleaInpainter()
    elif mode == 'ns':
        inpainter = NavierStokesInpainter()
    elif mode == 'lama':
        inpainter = LamaInpainter()
    else:
        raise ValueError('Invalid inpainter mode')
    return inpainter
