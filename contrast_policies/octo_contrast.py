import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from transforms3d.euler import euler2axangle

from octo.octo.model.components.base import TokenGroup
from simpler_env.policies.octo.octo_model import OctoInference

from .kde_contrast_decoding import ContrastDecoding


class OctoContrastInference(OctoInference):
    def __init__(self, 
                 alpha=0.1,
                 num_repeats=64,
                 bandwidth_factor=1.0,
                 keep_threshold=0.5,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast_image_history = deque(maxlen=self.horizon)
        self.num_repeats = num_repeats
        self.contrast_decoding = ContrastDecoding(alpha=alpha, bandwidth_factor=bandwidth_factor, 
                                                  keep_threshold=keep_threshold, mode='jax')
        
    def reset(self, task_description, seed=None):
        super().reset(task_description, seed)
        self.task = self.model.create_tasks(texts=[task_description, task_description])
        self.contrast_image_history.clear()
        
    def step(self, image, contrast_image, task_description=None, *args, **kwargs):
        image, contrast_image, task_description = self._process_inputs(image, contrast_image, task_description)
        self._add_image_to_history(image, contrast_image)
        images, contrast_images, pad_mask = self._obtain_image_history_and_mask()
        images, contrast_images, pad_mask = images[None], contrast_images[None], pad_mask[None]
        input_observation = {"image_primary": jnp.concatenate([images, contrast_images], axis=0), 
                             "pad_mask": jnp.concatenate([pad_mask, pad_mask], axis=0)}
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        transformer_outputs = self.encode_features(input_observation, self.task, train=False, rng=key)

        action_head = self.model.module.bind({"params": self.model.params}).heads["action"]
        readout_key = action_head.readout_key
        transformer_outputs = self.split_repeat_concat_features(transformer_outputs, readout_key)
        
        distribution, contrast_distribution = self.sample_actions(action_head, transformer_outputs, key)
        norm_raw_actions = self.contrast_decoding(distribution, contrast_distribution)
        
        aux_info = {'actions': distribution, 'contrast_actions': contrast_distribution}
        raw_actions, actions = self._post_process_action(norm_raw_actions)
        return raw_actions, actions, aux_info
    
    def sample_actions(self, action_head, transformer_outputs, key):
        actions = action_head.predict_action(transformer_outputs, train=False, rng=key)
        actions, contrast_actions = jnp.split(actions, 2, axis=0)
        return actions, contrast_actions
    
    def encode_features(self, observations, tasks, train=False, rng=None):
        pad_mask = observations["pad_mask"]
        return self.model.run_transformer(observations, tasks, pad_mask, train=train)
    
    def split_repeat_concat_features(self, transformer_outputs, readout_key):
        tokens, contrast_tokens = jnp.split(transformer_outputs[readout_key].tokens, 2, axis=0)
        mask, contrast_mask = jnp.split(transformer_outputs[readout_key].mask, 2, axis=0)
        tokens = jnp.repeat(tokens, self.num_repeats, axis=0)
        mask = jnp.repeat(mask, self.num_repeats, axis=0)
        contrast_tokens = jnp.repeat(contrast_tokens, self.num_repeats, axis=0)
        contrast_mask = jnp.repeat(contrast_mask, self.num_repeats, axis=0)
        tokens = jnp.concatenate([tokens, contrast_tokens], axis=0)
        mask = jnp.concatenate([mask, contrast_mask], axis=0)
        return {readout_key: TokenGroup(tokens, mask)}

    def _process_inputs(self, image, contrast_image, task_description=None):
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
        assert image.dtype == np.uint8
        return self._resize_image(image), self._resize_image(contrast_image), task_description
    
    def _post_process_action(self, norm_raw_actions):
        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_actions = raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)

        assert raw_actions.shape == (self.pred_action_horizon, 7)
        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
            raw_actions = raw_actions[None]  # [1, 7]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
    
    def _add_image_to_history(self, image, contrast_image):
        self.image_history.append(image)
        self.contrast_image_history.append(contrast_image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self):
        images = np.stack(self.image_history, axis=0)
        contrast_images = np.stack(self.contrast_image_history, axis=0)
        horizon = len(self.image_history)
        pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        return images, contrast_images, pad_mask
