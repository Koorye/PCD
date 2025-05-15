import numpy as np
import torch
from transforms3d.euler import euler2axangle
from PIL import Image

from simpler_env.policies.openvla.openvla_model import OpenVLAInference


_TOKEN_DIM = 32064


class OpenVLAContrastInference(OpenVLAInference):
    def __init__(self, 
                 alpha=0.2, 
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    def step(self, image, contrast_image, task_description=None, *args, **kwargs):
        inputs = self.process_inputs(image, task_description=task_description)
        contrast_inputs = self.process_inputs(contrast_image, task_description=task_description)
        raw_actions, aux_info = self.predict_action(inputs, contrast_inputs, unnorm_key=self.unnorm_key, do_sample=False)
        raw_actions = raw_actions[None]
        raw_actions, actions = self.postprocess_actions(raw_actions)
        return raw_actions, actions, aux_info
    
    def process_inputs(self, image, task_description=None):
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        image = Image.fromarray(image)
        prompt = task_description
        return self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

    def predict_action(self, inputs, contrast_inputs, unnorm_key=None, **kwargs):
        scores = self._forward_scores(inputs, unnorm_key, **kwargs)
        
        if scores.size(0) < self.vla.get_action_dim(unnorm_key):
            print(f"*** Warning: scores size {scores.size(0)} < action dim {self.vla.get_action_dim(unnorm_key)} ***")
            scores = torch.zeros((self.vla.get_action_dim(unnorm_key), _TOKEN_DIM), device=scores.device)

        contrast_scores = self._forward_scores(contrast_inputs, unnorm_key, **kwargs)

        # sometimes decoding fails
        if contrast_scores.size(0) < self.vla.get_action_dim(unnorm_key) - 1:
            print(f"*** Warning: contrast scores size {contrast_scores.size(0)} < action dim {self.vla.get_action_dim(unnorm_key)} - 1***")
            contrast_scores = torch.zeros((self.vla.get_action_dim(unnorm_key) - 1, _TOKEN_DIM), device=contrast_scores.device)

        final_scores = scores.clone()
        if contrast_scores.size(0) == self.vla.get_action_dim(unnorm_key):
            final_scores[:-1] = (1 + self.alpha) * final_scores[:-1] - self.alpha * contrast_scores[:-1]
            # final_scores[:-1] = final_scores[:-1] - self.alpha * contrast_scores[:-1]

        predicted_action_token_ids = final_scores.argmax(dim=-1)
        return self._decode_actions(predicted_action_token_ids, unnorm_key), {'logits': scores, 'contrast_logits': contrast_scores, 'final_logits': final_scores}

    def postprocess_actions(self, raw_actions):
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
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
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
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action
    
    def _forward_scores(self, inputs, unnorm_key, **kwargs):
        input_ids, pixel_values, attention_mask = inputs["input_ids"], inputs["pixel_values"], inputs["attention_mask"]
        
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat((input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
        
        # proprio is not used in openvla
        if 'proprio' in kwargs:
            del kwargs['proprio']

        scores = self.vla.generate(
            input_ids=input_ids,                            # Shape: [1, seq]
            pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
            max_new_tokens=self.vla.get_action_dim(unnorm_key),
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )['scores']
        return torch.cat(scores, dim=0)
    
    # def visualize_scores(self, scores, contrast_scores, unnorm_key):
    #     scores = scores.clone()
    #     contrast_scores = contrast_scores.clone()
    #     final_scores = scores - self.alpha * contrast_scores

    #     # [0, 1, 2, ..., 32063] -> [32000, 31999, ..., 0, ..., -63]
    #     action_token_ids = torch.arange(scores.size(1))
    #     discretized_actions = self.vla.vocab_size - action_token_ids.cpu()

    #     # (bins,)
    #     # [31999, ..., -64]
    #     discretized_actions -= 1
    #     # [F, F, ..., T, ..., F]
    #     keep_mask = np.logical_and(discretized_actions >= 0, discretized_actions <= self.vla.bin_centers.shape[0] - 1)
    #     discretized_actions = discretized_actions[keep_mask]
    #     num_bins = discretized_actions.shape[0]
    #     num_actions = scores.shape[0]

    #     # (bins,) -> (actions, bins)
    #     normalized_actions = self.vla.bin_centers[discretized_actions]
    #     normalized_actions = np.repeat(normalized_actions[None], num_actions, axis=0)

    #     # Unnormalize actions
    #     action_norm_stats = self.vla.get_action_stats(unnorm_key)
    #     mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool)))
    #     action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    #     # (actions,) -> (actions, bins)
    #     mask = np.stack([mask for _ in range(num_bins)], axis=1)
    #     action_high = np.stack([action_high for _ in range(num_bins)], axis=1)
    #     action_low = np.stack([action_low for _ in range(num_bins)], axis=1)
    #     normalized_actions = np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)

    #     scores = scores[:, keep_mask].softmax(dim=-1).detach().cpu().numpy()
    #     contrast_scores = contrast_scores[:, keep_mask].softmax(dim=-1).detach().cpu().numpy()
    #     final_scores = final_scores[:, keep_mask].softmax(dim=-1).detach().cpu().numpy()
        
    #     import matplotlib.pyplot as plt
    #     for i in range(3):
    #         plt.subplot(3, 1, i + 1)
    #         plt.plot(normalized_actions[i], scores[i], color='blue')
    #         plt.plot(normalized_actions[i], contrast_scores[i], color='red')
    #         plt.plot(normalized_actions[i], final_scores[i], color='green')
    #     plt.tight_layout()
    #     plt.savefig('openvla.jpg')
    #     plt.close()
