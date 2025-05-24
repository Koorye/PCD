import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_gif(gif_path):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = np.array(gif.convert('RGB'))
            frames.append(frame)
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    gif.close()
    return frames


def sample_gif(frames, num_frames):
    if num_frames > len(frames):
        return frames
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled_frames = [frames[i] for i in indices]
    return sampled_frames


def resize_gif(frames, width, height=None):
    org_width, org_height = frames[0].shape[1], frames[0].shape[0]
    if height is None:
        height = int(org_height * (width / org_width))
    if org_width == width and org_height == height:
        return frames
    
    resized_frames = []
    for frame in frames:
        if height is None:
            height = int(frame.shape[0] * (width / frame.shape[1]))
        resized_frame = np.array(Image.fromarray(frame).resize((width, height)))
        resized_frames.append(resized_frame)
    return resized_frames


def concat_gifs(gifs, ncols, margin=0, background_color=(255, 255, 255)):
    min_num_frames = min(len(gif) for gif in gifs)
    gifs = [sample_gif(gif, min_num_frames) for gif in gifs]

    nrows = int(np.floor(len(gifs) / ncols))
    gifs = gifs[:nrows * ncols]
    
    min_nframes = min([len(gif) for gif in gifs])
    gifs = [sample_gif(gif, min_nframes) for gif in gifs]
    
    gifs = [resize_gif(gif, 640, 512) for gif in gifs]
    height, width = gifs[0][0].shape[:2]
    
    # concat_frames = [np.zeros((height * nrows, width * ncols, 3), dtype=np.uint8) for _ in range(min_nframes)]
    # add margin
    # concat_frames = [np.zeros((height * nrows + margin * (nrows - 1), width * ncols + margin * (ncols - 1), 3), dtype=np.uint8) for _ in range(min_nframes)]
    concat_frames = [np.full((height * nrows + margin * (nrows - 1), width * ncols + margin * (ncols - 1), 3), background_color, dtype=np.uint8) for _ in range(min_nframes)]
    
    for i, gif in tqdm(enumerate(gifs), desc='Concatenating GIFs', total=len(gifs)):
        for j, frame in enumerate(gif):
            row = i // ncols
            col = i % ncols
            # concat_frames[j][row * height:(row + 1) * height, col * width:(col + 1) * width] = frame
            # add margin
            concat_frames[j][row * (height + margin): (row + 1) * (height + margin) - margin, col * (width + margin): (col + 1) * (width + margin) - margin] = frame
    
    return concat_frames


def save_video(frames, save_path):
    frames = [Image.fromarray(frame) for frame in frames]
    imageio.mimsave(save_path, frames, fps=10)


if __name__ == '__main__':
    # tasks = ['pick_ball', 'pick_plug', 'move_near', 'cookie_towel', 'banana_plate', 'stack_cube']
    # save_path = 'main_real_baseline.mp4'
    
    # gif_paths = [f'main/real/baseline/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=6, margin=10)
    # save_video(frames, save_path)
    
    # save_path = 'main_real_pcd.mp4'
    
    # gif_paths = [f'main/real/pcd/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=6, margin=10)
    # save_video(frames, save_path)
    
    # tasks = ['distractor', 'spatial', 'brightness', 'lego', 'texture']
    # save_path = 'factor_real_baseline.mp4'
    
    # gif_paths = [f'factor/real/baseline/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=5, margin=10)
    # save_video(frames, save_path)
    
    # save_path = 'factor_real_pcd.mp4'
    
    # gif_paths = [f'factor/real/pcd/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=5, margin=10)
    # save_video(frames, save_path)
    
    tasks = ['pick_coke_can', 'move_near', 'carrot_plate', 'eggplant_basket']
    save_path = 'main_simpler_baseline.mp4'
    
    # gif_paths = [f'main/simpler/baseline/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=4, margin=10)
    # save_video(frames, save_path)
    
    save_path = 'main_simpler_pcd.mp4'
    
    gif_paths = [f'main/simpler/pcd/cut_{task}.gif' for task in tasks]
    gifs = [load_gif(gif_path) for gif_path in gif_paths]
    frames = concat_gifs(gifs, ncols=4, margin=10)
    save_video(frames, save_path)
    
    tasks = ['spatial', 'brightness', 'texture', 'texture2']
    save_path = 'factor_simpler_baseline.mp4'
    
    # gif_paths = [f'factor/simpler/baseline/{task}.gif' for task in tasks]
    # gifs = [load_gif(gif_path) for gif_path in gif_paths]
    # frames = concat_gifs(gifs, ncols=4, margin=10)
    # save_video(frames, save_path)
    
    save_path = 'factor_simpler_pcd.mp4'
    
    gif_paths = [f'factor/simpler/pcd/cut_{task}.gif' for task in tasks]
    gifs = [load_gif(gif_path) for gif_path in gif_paths]
    frames = concat_gifs(gifs, ncols=4, margin=10)
    save_video(frames, save_path)
    