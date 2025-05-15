import argparse
import multiprocessing
import numpy as np
import os
import os.path as osp
import shutil
import traceback

from utils import *


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.unwrapped.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uid:
            camera_name = "3rd_view_camera"
        elif "panda" in env.unwrapped.robot_uid:
            camera_name = "overhead_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


class ParallelRunner:
    """
    A parallel runner to run inferences in parallel, it will indenpendently 
    build environment and policy on each GPU, and run episodes in parallel.
        - num_gpus: number of GPUs
        - policy: policy name
        - checkpoint: checkpoint path
        - task: task name
        - result_root: root directory to save results
        - n_trajs: number of episodes
        - contrast: whether to use contrast
        - opts: other options
    """
    
    def __init__(self,
                 num_gpus=0,
                 policy='',
                 checkpoint='',
                 task='',
                 result_root='./results',
                 n_trajs=100,
                 contrast=False,
                 opts=[]):
        self.num_gpus = num_gpus
        self.policy = policy
        self.checkpoint = checkpoint
        self.task = task
        self.result_root = result_root
        self.n_trajs = n_trajs
        self.contrast = contrast
        self.opts = parse_opts(opts)
        
    def run(self):
        """
        Run episodes:
            1. Check if single GPU or multiple GPUs, then run episodes in serial or parallel
            2. Collect, summarize and save results
        """
        if self.num_gpus == 1:
            self.run_in_serial()
        else:
            self.run_in_parallel()
    
    def run_in_serial(self):
        """ Run episodes in serial, used for single GPU. """
        if not self._set_result_dir():
            return
        self._build_logger()
        gpu_id = self._check_free_gpus()[0]
        episodes = range(self.n_trajs)
        infos = self.run_episodes(gpu_id, episodes, show_detail=True)
        info = stat_info(infos)
        self.logger.infos("Results", info)
        os.rename(osp.join(self.result_dir, '000.log'),
                  osp.join(self.result_dir, f"000_success_{round(info['success'], 2)}.log"))
    
    def run_in_parallel(self):
        """ 
        Run episodes in parallel, used for multiple GPUs:
            1. Check free GPUs and set GPU for each process
            2. Allocate episodes for each process, e.g., if n_trajs=100, and 3 GPUs, then each GPU will run 
            [0, 3, 6, ..., 99], [1, 4, 7, ..., 100], [2, 5, 8, ..., 101], respectively
            3. Start each processes
            4. Check if any process failed
            5. Collect informations
        Return:
            - infos: list of information of episodes
        """
        if not self._set_result_dir():
            return
        self._build_logger()
        infos = multiprocessing.Queue()
        gpu_ids = self._check_free_gpus()
        if self.num_gpus < len(gpu_ids):
            gpu_ids = gpu_ids[:self.num_gpus]
        
        # start number of processes equal to number of GPUs
        processes = []
        episodes_per_gpu = [list(range(i, self.n_trajs, len(gpu_ids))) for i, _ in enumerate(gpu_ids)]
        for gpu_id, episodes in zip(gpu_ids, episodes_per_gpu):
            self.logger.info(f"Allocating episodes for GPU {gpu_id}: {episodes}.")
            process = multiprocessing.Process(target=self.run_episodes, 
                                              args=(gpu_id, episodes, infos, gpu_id == gpu_ids[0]))
            process.start()
            processes.append(process)
            
        for process in processes:
            process.join()
            
        for process in processes:
            if process.exitcode != 0:
                print(f"Process {process.pid} failed.")
                raise RuntimeError()
            
        infos = [infos.get() for _ in range(self.n_trajs)]
        info = stat_info(infos)
        
        # run_episode with multi processing will fork the logger,
        # the logger in the main process will not be updated,
        # so we need to reinitialize the logger
        self._build_logger(mode='a')
        self.logger.infos("Results", info)
        os.rename(osp.join(self.result_dir, '000.log'),
                  osp.join(self.result_dir, f"000_success_{info['success']}.log"))
    
    def run_episodes(self, gpu_id, episodes, info_queue=None, show_detail=False):
        """
        Run episodes
        Return:
            - infos: list of information of episodes
        """
        env, policy, others = self.build_episode(gpu_id, show_detail)
        infos = []
        for episode in episodes:
            self.logger.info(f"Running episode {episode} on GPU {gpu_id}.")
            try:
                info = self.run_episode(env, policy, others, episode, show_detail=show_detail)
                if self.policy == 'pizero':
                    self.logger.info('clear cache for pi-0')
                    policy.model.to('cpu')
                    del policy.model
                    del policy
                    policy = self._build_policy(show_detail)
            
            except Exception as e:
                self.logger.error(f"Episode {episode} failed with error: {e}.")
                self.logger.error(traceback.format_exc())
                self._write_error(episode, e)
                raise e
            infos.append(info)
            if info_queue is not None:
                info_queue.put(info)
        return infos

    def run_episode(self, env, policy, others, episode, show_detail=False):
        """
        Run episode:
            1. Reset environment and policy
            2. Get initial image and instruction input
            3. While not success or terminated:
                - Get action from policy
                - Apply action to environment
                - Get new image and instruction input
            4. Summarize episode
        Return:
            - info: information of episode
        """
        # reset environment
        obs, _ = env.reset(seed=episode)

        # get initial instruction
        instruction = env.unwrapped.get_language_instruction()
        is_final_subtask = env.unwrapped.is_final_subtask() 

        # reset policy
        policy.reset(instruction, seed=episode)
        if self.contrast:
            self.contrast_image_generator.reset()
        if show_detail:
            self.logger.info(f"Initial instruction: {instruction}")

        predicted_terminated, success, truncated = False, False, False
        timestep = 0
        frames = []
        step_infos = []

        # get initial image
        image = get_image_from_maniskill2_obs_dict(env, obs)  # np.ndarray of shape (H, W, 3), uint8
        
        if not self.contrast:
            frames.append(image)
        else:
            contrast_image = self.contrast_image_generator.generate(obs, instruction)
            frames.append(tile_images([image, contrast_image]))
        
        # run episode
        while not (predicted_terminated or truncated):
            # get action from policy
            # only pi-0 use proprio
            if not self.contrast:
                raw_action, actions = policy.step(image, instruction, proprio=obs['agent']['eef_pos'])
            else:
                raw_action, actions, aux_info = policy.step(image, contrast_image, instruction, proprio=obs['agent']['eef_pos'])

            if not isinstance(actions, list):
                actions = [actions]
            
            for action in actions:
                # apply action to environment
                obs, reward, success, truncated, info = env.step(np.concatenate([action["world_vector"], 
                                                                                 action["rot_axangle"], 
                                                                                 action["gripper"]]))
                image = get_image_from_maniskill2_obs_dict(env, obs)
                if not self.contrast:
                    frames.append(image)
                    # write_images([image], f"visualize/test.jpg")
                else:
                    contrast_image = self.contrast_image_generator.generate(obs, instruction)
                    frames.append(tile_images([image, contrast_image]))
                    # write_images([image, contrast_image], f"visualize/test.jpg")

                is_final_subtask = env.unwrapped.is_final_subtask() 
                timestep += 1
                info = convert_numpy_or_torch_to_python(info)
                step_infos.append(info)
                predicted_terminated = bool(action["terminate_episode"][0] > 0)

                if show_detail:
                    self.logger.info(f"Step {timestep}: {info}")

                if predicted_terminated:
                    if not is_final_subtask:
                        # advance the environment to the next subtask
                        predicted_terminated = False
                        env.advance_to_next_subtask()
                
                new_instruction = env.unwrapped.get_language_instruction()
                if new_instruction != instruction:
                    instruction = new_instruction
                    if show_detail:
                        self.logger.info(f"New instruction: {instruction}")
        
        # summarize episode
        info = summarize(step_infos)
        info.update(stat_first(step_infos))
        info.update(stat_final(step_infos))
        success = info['success']
        self.logger.info(f"Episode {episode} finished with success {success}.")
        write_video(frames, f"{self.result_dir}/episode_{episode}_success_{success}.gif")
        return info
    
    def build_episode(self, gpu_id, show_detail):
        """ 
        Build episode:
            1. Set GPU
            2. Build environment
            3. Build contrast image generator if contrast is used
            4. Build policy
            5. Build others
        Return:
            - env: environment
            - policy: policy model
            - others: others
        """
        self._set_gpu(gpu_id)
        env = self._build_environment(show_detail)

        if self.contrast:
            self._build_contrast_image_generator(env, show_detail)
        
        policy = self._build_policy(show_detail)
        others = self._build_others(show_detail)
        return env, policy, others

    def _set_gpu(self, gpu_id):
        """  Set GPU, it must be called before building policy. """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # list_physical devices can avoid cuda error, don't know why
        import tensorflow as tf
        tf.config.list_physical_devices("GPU")

    def _build_environment(self, show_detail=False):
        """ Build environment. """
        import simpler_env
        return simpler_env.make(self.task)
    
    def _build_policy(self, show_detail=False):
        """ Build policy model. """
        from properties import get_policy_config
        config = get_policy_config(self.policy, self.checkpoint, self.task, self.opts, self.contrast)
        
        if show_detail:
            self.logger.infos("Policy Config", config)

        from contrast_policies import get_policy
        policy = get_policy(self.policy, self.contrast, config)
        
        reset_logging()
        self._build_logger(mode='a')
        self.policy_model = policy
        return policy

    def _build_others(self, show_detail=False):
        return {}
    
    def _build_contrast_image_generator(self, env, show_detail=False):
        """  Build contrast image generator. """
        from properties import get_contrast_image_generator_config
        config = get_contrast_image_generator_config(self.opts)
        
        if show_detail:
            self.logger.infos("Contrast Image Config", config)
        
        from contrast_utils import get_contrast_image_generator
        config['env'] = env
        self.contrast_image_generator = get_contrast_image_generator(config)
        reset_logging()
        self._build_logger(mode='a')
    
    def _build_logger(self, mode='w'):
        """ Build logger. """
        self.logger = Logger(osp.join(self.result_dir, '000.log'), mode)
    
    def _check_free_gpus(self):
        """ Check free GPUs. """
        used_memorys = os.popen(f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader").readlines()
        used_memorys = [int(memory.strip()) for memory in used_memorys]
        return [i for i, memory in enumerate(used_memorys) if memory < 1000]
    
    def _set_result_dir(self):
        # model_name = '/'.join(self.checkpoint.replace('\\', '/').split('/')[1:])
        model_name = self.checkpoint.replace('\\', '/').split('/')[-1]
        self.result_dir = osp.join(self.result_root, model_name)
        if len(self.opts) > 0:
            self.result_dir = osp.join(self.result_dir, '--'.join(f'{k}={v}' for k, v in self.opts.items()))
        self.result_dir = osp.join(self.result_dir, self.task)
        
        # create result directory if not exists
        if not os.path.exists(self.result_dir):
            print(f"Directory {self.result_dir} does not exist, create it.")
            os.makedirs(self.result_dir)
            return True
        
        log_filename = [filename for filename in os.listdir(self.result_dir) if filename.startswith('000') and filename.endswith('.log')]
        # abnormal state: have directory but not log has created, remove it and create directory
        if len(log_filename) == 0:
            print(f"Directory {self.result_dir} exists, but no log file, remove it and create directory.")
            shutil.rmtree(self.result_dir)
            os.makedirs(self.result_dir)
            return True
        
        # normal state: have directory and finished task
        log_filename = log_filename[0]
        if log_filename.startswith('000_success_'):
            print(f"Directory {self.result_dir} exists, and finished task, skip it.")
            return False

        # other abnormal states: e.g. have directory and log, but not finished task
        print(f"Directory {self.result_dir} exists, but not finished task, remove it and create directory.")
        shutil.rmtree(self.result_dir)
        os.makedirs(self.result_dir)
        return True
    
    def _write_error(self, episode, error):
        with open(osp.join(self.result_dir, f"000_episode_{episode}_error.log"), 'w') as f:
            f.write(str(error))
            traceback.print_exc(file=f)


class SearchParallelRunner(ParallelRunner):
    def __init__(self,
                 search_opts,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.search_opts = parse_opts(search_opts)
    
    def run(self):
        # search opts is like:
        # {
        #    'a': [1, 2, 3],
        #    'b': [4, 5, 6],
        #    'c': [7, 8, 9],
        # }
        # run episodes for each combination of a, b, c, 3 * 3 * 3 = 27
        # use meshgrid
        keys = list(self.search_opts.keys())
        values = list(self.search_opts.values())
        meshgrid = np.meshgrid(*values, indexing='ij')
        meshgrid = [m.flatten() for m in meshgrid]
        
        for i in range(len(meshgrid[0])):
            self.opts.update({keys[j]: meshgrid[j][i] for j in range(len(keys))})
            super().run()

def main(args):
    if len(args.search_opts) > 0:
        runner = SearchParallelRunner(search_opts=args.search_opts,
                                      num_gpus=args.num_gpus,
                                      policy=args.policy,
                                      checkpoint=args.checkpoint,
                                      task=args.task,
                                      result_root=args.result_root,
                                      n_trajs=args.n_trajs,
                                      contrast=args.contrast,
                                      opts=args.opts)
    else:
        runner = ParallelRunner(num_gpus=args.num_gpus,
                                policy=args.policy,
                                checkpoint=args.checkpoint,
                                task=args.task,
                                result_root=args.result_root,
                                n_trajs=args.n_trajs,
                                contrast=args.contrast,
                                opts=args.opts)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--policy", default="cogact")
    parser.add_argument("--checkpoint", type=str, default="pretrained/cogact-base")
    parser.add_argument("--task", default="google_robot_pick_coke_can")
    parser.add_argument("--result-root", type=str, default="./results")
    parser.add_argument("--n-trajs", type=int, default=100)
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--opts", nargs="+", default=[])
    parser.add_argument("--search-opts", nargs="+", default=[])
    args = parser.parse_args()
    main(args)
