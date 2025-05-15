import cv2
import numpy as np
import random
import torch


_IMAGE_MEAN = (0.485, 0.456, 0.406)
_IMAGE_STD = (0.229, 0.224, 0.225)

def _load_image_numpy_as_tensor(image, image_size):
    # img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    # cv2 resize
    video_height, video_width = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))
    if image.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        image = image / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {image.dtype}")
    image = torch.from_numpy(image).permute(2, 0, 1)
    return image, video_height, video_width


def load_images_numpy(images, image_size, offload_video_to_cpu, compute_device):
    img_mean = torch.tensor(_IMAGE_MEAN, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(_IMAGE_STD, dtype=torch.float32)[:, None, None]
    num_frames = len(images)

    image_tensors = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, image in enumerate(images):
        image_tensors[n], video_height, video_width = _load_image_numpy_as_tensor(image, image_size)
    if not offload_video_to_cpu:
        image_tensors = image_tensors.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    image_tensors -= img_mean
    image_tensors /= img_std
    return image_tensors, video_height, video_width


def dilate_mask(mask, kernel_size):
    if mask is None:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def erode_mask(mask, kernel_size):
    if mask is None:
        return None
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def index_all(l, e):
    # find all indexs of e in l
    return [i for i, x in enumerate(l) if x == e]


def equal_all(l1, l2):
    if len(l1) != len(l2):
        return False
    return all(x == y for x, y in zip(l1, l2))


def mask_to_bbox(mask):
    y, x = np.where(mask)
    return np.array([x.min(), y.min(), x.max(), y.max()])


def get_random_color(name, min_value=100, max_value=255):
    state = random.getstate()
    random.seed(name)
    color = (random.randint(min_value, max_value), 
             random.randint(min_value, max_value), 
             random.randint(min_value, max_value))
    random.setstate(state)
    return color


def visualize_multi_objects(image, masks, names, filename, points=None, boxes=None):
    image = image.copy()
    
    for idx, (mask, name) in enumerate(zip(masks, names)):
        if mask is None or not mask.any():
            continue
        color = get_random_color(name)
        mask_rgb = cv2.merge([mask.astype(np.uint8) * color[i] for i in range(3)])
        # draw mask on image
        image_with_mask = cv2.addWeighted(image, 0.5, mask_rgb, 0.5, 0)
        image[mask] = image_with_mask[mask]
        image = image.copy()
        
        y_center, x_center = np.where(mask)
        y_center = int(y_center.mean())
        x_center = int(x_center.mean())
        cv2.putText(image, name, (x_center, y_center), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        deeper_color = [c * 0.5 for c in color]
        if points is not None:
            # (2, 1) -> (2,)
            for point in points[idx]:
                cv2.circle(image, (int(point[0]), int(point[1])), 5, deeper_color, -1)
            
        if boxes is not None:
            xmin, ymin, xmax, ymax = boxes[idx]
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), deeper_color, 2)
        
    # save image to self.save_dir/visualize/name/filename
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)

