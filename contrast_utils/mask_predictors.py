import cv2
import json
import numpy as np
import os
import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from .utils import *


_CLASSNAMES = ['robot', 'coke can', 'pepsi can', 'redbull can', '7up can', 'blue plastic bottle', 
               'apple', 'orange', 'sponge', 'bottom drawer', 'middle drawer', 'top drawer',
               'eggplant', 'spoon', 'carrot', 'plate larger', 'table cloth shorter', 
               'yellow basket', 'green cube', 'yellow cube', 'goal_site']

_BACKGROUND_CLASSNAMES = ['floor', 'wall', 'ceiling']

_NAME_TO_ALIAS_GDINO = {
    '7up can': 'white 7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    # 'blue plastic bottle': 'blue plastic bottle with label',
    'sponge': 'green sponge',
    'plate larger': 'plate',
    'table cloth shorter': 'blue towel cloth',
    'top drawer': 'dresser',
    'middle drawer': 'dresser',
    'bottom drawer': 'dresser',
    'robot': 'robot manipulator',
}

_NAME_TO_ALIAS_YOLO_WORLD = {
    '7up can': '7up pop can',
    'redbull can': 'redbull pop can',
    'coke can': 'coke pop can',
    'pepsi can': 'pepsi pop can',
    'blue plastic bottle': 'mineral water bottle with label',
    'top drawer': 'cabinet with drawers',
    'middle drawer': 'cabinet with drawers',
    'bottom drawer': 'cabinet with drawers',
    'plate larger': 'plate',
    'table cloth shorter': 'blue towel cloth',
}

_NAME_TO_ALIAS_SED = {
    'blue plastic bottle': 'mineral water bottle with label',
    'top drawer': 'cabinet with drawers',
    'middle drawer': 'cabinet with drawers',
    'bottom drawer': 'cabinet with drawers',
    'plate larger': 'plate',
    'robot': 'robot manipulator',
    'table cloth shorter': 'blue towel cloth',
}

_SAM2_MODEL_CFG = os.path.join(os.path.dirname(__file__), 'grounded_sam_2', 'sam2', 'configs', 'sam2.1', 'sam2.1_hiera_l.yaml')
_SAM2_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'sam2.1_hiera_large.pth')

_YOLO_WORLD_CFG = os.path.join(os.path.dirname(__file__), 'yolo_world', 'configs', 'pretrain', 'yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py')
_YOLO_WORLD_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'l_stage1-7d280586.pth')

_GROUNDING_DINO_CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'pretrained', 'grounding-dino-base')


def my_print(*args):
    # get gpu id
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    if gpu_id == '0':
        print(*args)


def postprocess_mask(mask):
    if mask is None or not mask.any():
        return None
    
    # only keep the largest connected component
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    if num_labels > 1:
        # for each component, if area is smaller than 0.1 * largest_area, set it to 0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 0.1 * largest_area:
                mask[labels == i] = 0

    # get counter of mask and fill poly
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, contours, 1)
    return (mask > 0)


class GroundedSAMPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        
        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.processor = AutoProcessor.from_pretrained(_GROUNDING_DINO_CHECKPOINT)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GROUNDING_DINO_CHECKPOINT).to('cuda:0')
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image_pil = Image.fromarray(image.copy())
        
        # all_prompts = prompts + [name for name in _BACKGROUND_CLASSNAMES if name not in prompts]
        all_prompts = prompts
        text = '. '.join([_NAME_TO_ALIAS_GDINO.get(p, p) for p in all_prompts]) + '.'
        
        inputs = self.processor(image_pil, text=text, return_tensors="pt").to('cuda:0')
        outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        input_boxes = result['boxes'].cpu().numpy()
        scores = result['scores'].cpu().numpy()
        labels = []

        # avoid one alias to multi name mapping
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS_GDINO.items() if k in prompts}

        for l in result['labels']:
            # label will remove word "can"
            if l.endswith('pop'):
                l += ' can'
            labels.append(alias_to_name.get(l, l))

        self.image_predictor.set_image(image.copy())
        masks, mask_scores, mask_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        output_masks = []
        for name in prompts:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)

        return [postprocess_mask(mask) for mask in output_masks]


class SEDPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor
        from .SED.demo.predictor import VisualizationDemo as SEDDemo
        from .SED.sed import add_sed_config

        with open('contrast_utils/SED/datasets/simpler.json', 'w') as f:
            self.classnames = [_NAME_TO_ALIAS_SED.get(name, name) for name in _CLASSNAMES] + _BACKGROUND_CLASSNAMES
            self.classnames = list(set(self.classnames))
            json.dump(self.classnames, f)

        def setup_cfg():
            # load config from file and command-line arguments
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_sed_config(cfg)
            cfg.merge_from_file('contrast_utils/SED/configs/convnextL_768.yaml')
            cfg.merge_from_list(['MODEL.WEIGHTS', 'pretrained/sed.pth'])
            cfg.freeze()
            return cfg
        cfg = setup_cfg()
        self.model = SEDDemo(cfg)

        sam2_checkpoint = "pretrained/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sem_seg = self.model.predictor(image)['sem_seg'].argmax(dim=0)
        
        masks = []
        for prompt in prompts:
            idx = self.classnames.index(_NAME_TO_ALIAS_SED.get(prompt, prompt))
            mask = (sem_seg == idx).detach().cpu().numpy()
            if mask.any():
                masks.append(postprocess_mask(mask))
            else:
                masks.append(None)
        
        self.image_predictor.set_image(image)
        # if mask is None, set box [0, 0, 1, 1]
        input_boxes = np.array([mask_to_bbox(mask) if mask is not None 
                                else np.array([0.0, 0.0, 1.0, 1.0]) for mask in masks])
        refined_masks, mask_scores, mask_logits = self.image_predictor.predict(point_coords=None, point_labels=None,
                                                                               box=input_boxes, multimask_output=False)
        if refined_masks.ndim == 4:
            refined_masks = refined_masks.squeeze(1)
        
        # add None to refined_masks
        refined_masks = [(refined_masks[i] > 0) if mask is not None else None for i, mask in enumerate(masks)]
        return [postprocess_mask(mask) for mask in refined_masks]


class YoloWorldPredictor:
    def __init__(self):
        # load config
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg

        cfg = Config.fromfile(_YOLO_WORLD_CFG)
        cfg.work_dir = os.path.join('./work_dirs')
        cfg.load_from = _YOLO_WORLD_CHECKPOINT
        self.model = init_detector(cfg, checkpoint=_YOLO_WORLD_CHECKPOINT, device='cpu')
        test_pipeline_cfg = get_test_pipeline_cfg(cfg=cfg)
        test_pipeline_cfg[0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(test_pipeline_cfg)
        
        from contrast_utils.grounded_sam_2.sam2.build_sam import build_sam2_video_predictor, build_sam2
        from contrast_utils.grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_checkpoint = _SAM2_CHECKPOINT
        model_cfg = _SAM2_MODEL_CFG
        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device='cpu')
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
    
    @torch.no_grad()
    def predict(self, image, prompts):
        prompts = [_NAME_TO_ALIAS_YOLO_WORLD.get(p, p) for p in prompts]
        result = self.yolo_world_inference(self.model, image, prompts, self.test_pipeline)

        input_boxes = result['boxes']
        scores = result['scores']
        labels = []
        
        if len(input_boxes) == 0:
            return [None] * len(prompts)

        # avoid one alias to multi name mapping
        alias_to_name = {v: k for k, v in _NAME_TO_ALIAS_YOLO_WORLD.items() if k in prompts}

        for l in result['label_texts']:
            labels.append(alias_to_name.get(l, l))

        self.image_predictor.set_image(image.copy())
        masks, mask_scores, mask_logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        output_masks = []
        for name in prompts:
            if name in labels:
                indexs = index_all(labels, name)
                selected_scores = scores[indexs]
                argmax_scores = selected_scores.argmax()
                output_masks.append(masks[indexs[argmax_scores]] > 0)
            else:
                output_masks.append(None)

        return [postprocess_mask(mask) for mask in output_masks]
    
    def yolo_world_inference(self, model, image, texts, test_pipeline, score_thr=0.3, max_dets=100):
        texts.append(' ')
        texts = [[t] for t in texts]

        data_info = dict(img=image, img_id=0, texts=texts)
        data_info = test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
        with torch.no_grad():
            output = model.test_step(data_batch)[0]
        pred_instances = output.pred_instances
        # score thresholding
        pred_instances = pred_instances[pred_instances.scores.float() > score_thr]
        # max detections
        if len(pred_instances.scores) > max_dets:
            indices = pred_instances.scores.float().topk(max_dets)[1]
            pred_instances = pred_instances[indices]

        pred_instances = pred_instances.cpu().numpy()
        boxes = pred_instances['bboxes']
        labels = pred_instances['labels']
        scores = pred_instances['scores']
        label_texts = [texts[x][0] for x in labels]
        return {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'label_texts': label_texts,
        }


class VisualPromptPredictor:
    def __init__(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2
        from .grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_image_model = build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)
        self.points = None
        self.boxes = None
    
    def set_points(self, points):
        # points is a list contains None
        self.points = np.array([point for point in points if point is not None])
        self.mask_index = [point is not None for point in points]
        
    def set_boxes(self, boxes):
        self.boxes = np.array([box for box in boxes if box is not None])
        self.mask_index = [box is not None for box in boxes]
        # my_print('set_box:', self.boxes, self.mask_index)
    
    def predict(self, image, prompts):
        # only support one of points or boxes
        assert self.points is not None or self.boxes is not None, 'points or boxes must be provided!'
        assert self.points is None or self.boxes is None, 'only one of points or boxes can be provided!'
        
        self.image_predictor.set_image(image.copy())
        if self.points is not None:
            point_labels = np.ones(self.points.shape[:-1], dtype=int)
            masks, _, _ = self.image_predictor.predict(point_coords=self.points, point_labels=point_labels, 
                                                       box=None, multimask_output=False)
        elif self.boxes is not None:
            masks, _, _ = self.image_predictor.predict(point_coords=None, point_labels=None, 
                                                       box=self.boxes, multimask_output=False)
            # my_print('predict:', len(masks))
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        out_masks = []
        count = 0
        for mask_index in self.mask_index:
            if mask_index:
                out_masks.append(masks[count])
                count += 1
            else:
                out_masks.append(None)
        # my_print('predict out:', len(out_masks))
        return [postprocess_mask(mask) for mask in out_masks]


class TrackingPredictor:
    def __init__(self, predictor):
        from .grounded_sam_2.sam2.build_sam import build_sam2_video_predictor

        self.predictor = predictor
        self.video_predictor = build_sam2_video_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)

        self.images = []
        self.masks_record = []
        self.has_masks_record = []
        self.inference_state = None
        self.start_tracking = False

        self.init_masks = None
    
    def predict(self, image, prompts):
        if not self.start_tracking:
            # if not start tracking, init or detect masks
            masks = self.predictor.predict(image, prompts)
            
            if all(mask is None for mask in masks):
                return masks
            
            # if there is at least one mask, start tracking
            self.reset_tracking_state_and_masks(image, masks, prompts)
            self.start_tracking = True
            return masks
        
        masks = self.tracking_next(image, prompts)
        # update tracking state
        self.reset_tracking_state_and_masks(image, masks, prompts)
        return masks
    
    def reset_tracking_state_and_masks(self, image, masks, prompts):
        self.images.append(image)
        self.masks_record.append(masks)
        self.has_masks_record.append([mask is not None for mask in masks])
        
        has_masks_init = self.has_masks_record[0]
        selected_indexs = [i for i, has_masks in enumerate(self.has_masks_record) if equal_all(has_masks, has_masks_init)]
        selected_indexs = self.uniform_select(selected_indexs, 3, 4)
        selected_images = [self.images[i] for i in selected_indexs]
        selected_masks_list = [self.masks_record[i] for i in selected_indexs]
        self.inference_state = self.video_predictor.init_state_from_images(selected_images)

        for frame_idx, masks in enumerate(selected_masks_list):
            for mask, prompt in zip(masks, prompts):
                obj_id = _CLASSNAMES.index(prompt) + 1
                if mask is not None:
                    self.video_predictor.add_new_mask(self.inference_state, frame_idx, obj_id, mask)
    
    def uniform_select(self, indexs, max_num, step):
        num = len(indexs)
        if num <= max_num:
            return indexs
        if num <= max_num * step:
            step = num // max_num
        return indexs[::step][:max_num]
    
    def tracking_next(self, image, prompts):
        images, _, _ = load_images_numpy([image], self.video_predictor.image_size, False, self.video_predictor.device)
        self.inference_state['images'] = torch.cat([self.inference_state['images'], images], dim=0)
        self.inference_state['num_frames'] = len(self.inference_state['images'])

        for frame_idx, obj_ids, mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
            if frame_idx != self.inference_state['num_frames'] - 1:
                continue
            name2mask = dict()
            for idx, obj_id in enumerate(obj_ids):
                classname = _CLASSNAMES[obj_id - 1]
                mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
                if not mask.any():
                    mask = None
                name2mask[classname] = mask
        
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        self.images = []
        self.masks_record = []
        self.has_masks_record = []
        self.inference_state = None
        self.start_tracking = False


class TrackingPredictorV2:
    def __init__(self, predictor):
        from .grounded_sam_2.sam2.build_sam import build_sam2_camera_predictor
        self.predictor = predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.start_tracking = False
        
    def predict(self, image, prompts):
        if not self.start_tracking:
            masks = self.predictor.predict(image, prompts)
            if all(mask is None for mask in masks):
                return masks
            
            self.video_predictor.load_first_frame(image)
            for mask, prompt in zip(masks, prompts):
                obj_id = _CLASSNAMES.index(prompt) + 1
                if mask is not None:
                    self.video_predictor.add_new_mask(0, obj_id, mask)
                    
            self.start_tracking = True
            return masks
        
        obj_ids, mask_logits = self.video_predictor.track(image)
        name2mask = dict()
        for idx, obj_id in enumerate(obj_ids):
            classname = _CLASSNAMES[obj_id - 1]
            mask = (mask_logits[idx].cpu().numpy() > 0).squeeze(0)
            if not mask.any():
                mask = None
            name2mask[classname] = mask
        
        masks = [name2mask.get(p, None) for p in prompts]
        return masks
    
    def reset(self):
        from .grounded_sam_2.sam2.build_sam import build_sam2_camera_predictor
        self.video_predictor.to('cpu')
        del self.video_predictor
        self.video_predictor = build_sam2_camera_predictor(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT)
        self.start_tracking = False


def build_predictor(predictor_name):
    if predictor_name == 'grounded_sam':
        return GroundedSAMPredictor()
    if predictor_name == 'grounded_sam_tracking':
        return TrackingPredictorV2(GroundedSAMPredictor())
    if predictor_name == 'sed':
        return SEDPredictor()
    if predictor_name == 'yolo_world':
        return YoloWorldPredictor()
    if predictor_name == 'sed_tracking':
        return TrackingPredictorV2(SEDPredictor())
    if predictor_name == 'yolo_world_tracking':
        return TrackingPredictorV2(YoloWorldPredictor())
    if predictor_name == 'point_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    if predictor_name == 'box_tracking':
        return TrackingPredictorV2(VisualPromptPredictor())
    raise ValueError(f'predictor_name {predictor_name} is not supported')


def predict_masks_with_predictor(image, prompts, predictor):
    masks = predictor.predict(image, prompts)
    # visualize_multi_objects(image, masks, prompts, 'test.jpg')
    return masks
