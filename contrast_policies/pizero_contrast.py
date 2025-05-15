import torch

from simpler_env.policies.pizero.pizero_model import PiZeroInference
from .kde_contrast_decoding import ContrastDecoding


class PiZeroContrastInference(PiZeroInference):
    def __init__(self, 
                 alpha=0.1,
                 num_repeats=64,
                 bandwidth_factor=1.0,
                 keep_threshold=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_repeats = num_repeats
        self.contrast_decoding = ContrastDecoding(alpha, bandwidth_factor, keep_threshold, 'torch')

        # set to None to disable clipping in infer_action function
        self.clip_value = self.model.final_action_clip_value
        self.model.final_action_clip_value = None
    
    @torch.no_grad()
    def step(self, image, contrast_image, instruction, proprio):
        inputs = self.preprocess_inputs(image, instruction, proprio)
        contrast_inputs = self.preprocess_inputs(contrast_image, instruction, proprio)
       
        # actions = self.forward_actions(inputs)
        # contrast_actions = self.forward_actions(contrast_inputs)
 
        all_inputs = {}
        for k in inputs:
            all_inputs[k] = torch.cat([inputs[k], contrast_inputs[k]], dim=0)
        all_actions = self.forward_actions(all_inputs)
        actions, contrast_actions = torch.chunk(all_actions, 2, dim=0)

        raw_actions = self.contrast_decoding(actions, contrast_actions)
        if self.clip_value is not None:
            raw_actions = torch.clamp(raw_actions, -self.clip_value, self.clip_value)
        
        actions = self.env_adapter.postprocess(raw_actions[0].float().cpu().numpy())
        return raw_actions, actions, {}

    def forward_actions(self, inputs):
        inputs.update({'num_repeats': self.num_repeats})
        with torch.inference_mode():
            if self.use_naive:
                actions = self.model.infer_actions_naive(**inputs)
            else:
                actions = self.model.infer_actions(**inputs)
        return actions
