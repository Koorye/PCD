import cv2
import datetime
import logging
import mediapy
import numpy as np
import re
import uuid


def parse_value(value):
    # parse value from string to list/int/float/bool
    if isinstance(value, str) and ',' in value:
        return [parse_value(v) for v in value.split(',')]
    if re.match(r'^-?\d+$', value):
        return int(value)
    if re.match(r'^-?\d+\.\d+$', value):
        return float(value)
    if value == 'True' or value == 'False':
        return value == 'True'
    return value
    

def parse_opts(opts):
    if len(opts) == 0:
        return dict()
    # [a, b, c, d] -> {a: b, c: d}
    return {opts[i]: parse_value(opts[i+1]) for i in range(0, len(opts), 2)}
    

def convert_numpy_or_torch_to_python(info):
    info_ = dict()
    keys = info.keys()
    for key in keys:
        value = info[key]
        try:
            info_[key] = value.item()
        except:
            info_[key] = value
    return info_


def summarize(infos):
    # delete key: elapsed_steps, episode_stats
    # for each key:
    # 1. if values are bool, if any is True, set to True, else set to False
    # 2. if values are int/float, set to smallest value
    info = dict()
    keys = infos[0].keys()
    
    for key in keys:
        if key in ["elapsed_steps", "episode_stats"]:
            continue
        
        values = [info[key] for info in infos]
        
        if isinstance(values[0], bool):
            info[key] = any(values)
        elif isinstance(values[0], int) or isinstance(values[0], float):
            info[key] = min(values)
        else:
            print(f"Key {key} has unknown type {type(values[0])}.")
            raise NotImplementedError()
    
    return info
    

def stat_first(infos):
    stat_first = dict()
    keys = infos[0].keys()
    for key in keys:
        if isinstance(infos[0][key], bool):
            values = [info[key] for info in infos]
            if any(values):
                stat_first[key] = values.index(True)
            else:
                stat_first[key] = -1
    return {f'first_{k}': v for k, v in stat_first.items()}


def stat_final(infos):
    stat_final = dict()
    keys = infos[0].keys()
    for key in keys:
        if isinstance(infos[0][key], bool):
            value = [info[key] for info in infos][-1]
            stat_final[key] = value
    return {f'final_{k}': v for k, v in stat_final.items()}


def stat_info(info_arr):
    stats = dict()
    keys = info_arr[0].keys()

    for key in keys:
        values = [info[key] for info in info_arr]
        if isinstance(values[0], bool):
            values = [int(value) for value in values]
            stats[key] = np.mean(values)
        else:
            # ignore -1
            values = [value for value in values if value != -1]
            if len(values) == 0:
                stats[key] = -1
            else:
                stats[key] = np.mean(values)
                
    return stats


def tile_images(images):
    # images: list of images, each image is a numpy array
    # return: a numpy array
    num_images = len(images)
    if num_images == 0:
        return None

    num_rows = int(np.ceil(np.sqrt(num_images)))
    num_cols = int(np.ceil(num_images / num_rows))
    image_shape = images[0].shape

    tiled_image = np.zeros((num_rows * image_shape[0], num_cols * image_shape[1], image_shape[2]), dtype=np.uint8)
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        tiled_image[row*image_shape[0]:(row+1)*image_shape[0], col*image_shape[1]:(col+1)*image_shape[1], :] = image

    return tiled_image


def write_images(images, path):
    image = tile_images(images) if len(images) > 1 else images[0]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


def write_video(images, path, fps=10):
    mediapy.write_video(path, images, fps=fps, codec='gif')


def reset_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


class Logger:
    def __init__(self, log_file, mode='w', name=None, log_level='info'):
        if name is None:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + str(uuid.uuid4())
        
        self.logger = logging.getLogger(name)
        # console handler and file handler
        ch = logging.StreamHandler()
        fh = logging.FileHandler(log_file, mode=mode)
        # formatter
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add handlers
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        if log_level == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif log_level == 'info':
            self.logger.setLevel(logging.INFO)
        elif log_level == 'warning':
            self.logger.setLevel(logging.WARNING)
        elif log_level == 'error':
            self.logger.setLevel(logging.ERROR)
        elif log_level == 'critical':
            self.logger.setLevel(logging.CRITICAL)
        else:
            raise ValueError(f"Invalid log level: {log_level}")

    def debug(self, msg):
        self.logger.debug(msg)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def critical(self, msg):
        self.logger.critical(msg)

    def infos(self, title, msgs, sort=True):
        self.info('-' * 40)
        self.info(title + ':')

        if isinstance(msgs, dict):
            if sort:
                msgs = {k: v for k, v in sorted(msgs.items(), key=lambda x: x[0])}
            for k, v in msgs.items():
                self.info(f"{k}: {v}")
        else:
            if sort:
                msgs = sorted(msgs)
            for msg in msgs:
                self.info(msg)
        
        self.info('-' * 40)
