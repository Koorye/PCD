import multiprocessing
import os
from PIL import Image, ImageSequence
from tqdm import tqdm


_FACTOR = 0.25
_NUM_PROCESSES = 50


def find_gif_files(root):
    gif_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.gif'):
                gif_files.append(os.path.join(dirpath, filename))
    return gif_files


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def get_gif_size(gif_file):
    with Image.open(gif_file) as im:
        return im.size


def resize_gif(gif_file, factor=0.5):
    with Image.open(gif_file) as im:
        size = (int(im.size[0] * factor), int(im.size[1] * factor))
        # im_resized = im.resize(size)
        # im_resized.save(gif_file, 'GIF', optimize=True, save_all=True, append_images=im_resized)
        resize_frames = [frame.resize(size) for frame in ImageSequence.Iterator(im)]
        resize_frames[0].save(gif_file, 'GIF', optimize=True, save_all=True, append_images=resize_frames[1:])


def resize_gif_files(gif_files, factor=0.5):
    for gif_file in tqdm(gif_files):
        resize_gif(gif_file, factor)


if __name__ == '__main__':
    gif_files = find_gif_files('results/default_error') + find_gif_files('results/light_variant') + find_gif_files('results/motivation') + find_gif_files('results/variant')
    gif_files_splits = split_list(gif_files, _NUM_PROCESSES)
    
    processes = []
    for gif_files in gif_files_splits:
        p = multiprocessing.Process(target=resize_gif_files, args=(gif_files, _FACTOR))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print('Done!')
