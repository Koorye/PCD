import torch
import yaml


# rdt_config_path = 'simpler_env/policies/rdt/configs/base.yaml'
# with open(rdt_config_path, "r") as fp:
#     rdt_config = yaml.safe_load(fp)


RT1_CONFIG = dict(
    saved_model_path="pretrained/rt_1_x",
    lang_embed_model_path="https://tfhub.dev/google/universal-sentence-encoder-large/5",
    image_width=320,
    image_height=256,
    action_scale=1.0,
    policy_setup="google_robot",
)
        
OCTO_CONFIG = dict(
    model=None,
    dataset_id=None,
    model_type='octo-base',
    model_step=None,
    policy_setup='widowx_bridge',
    horizon=2,
    pred_action_horizon=4,
    exec_horizon=1,
    image_size=256,
    action_scale=1.0,
    init_rng=0,
)

OPENVLA_CONFIG = dict(
    saved_model_path='openvla-7b',
    unnorm_key=None,
    policy_setup='google_robot',
    horizon=1,
    pred_action_horizon=1,
    exec_horizon=1,
    image_size=[224, 224],
    action_scale=1.0,
)

# RDT_POLICY_CONFIG = dict(
#     args=rdt_config, 
#     dtype=torch.bfloat16,
#     # pretrained_text_encoder_name_or_path=None,
#     pretrained_text_encoder_name_or_path='pretrained/t5-v1_1-xxl',
#     pretrained_vision_encoder_name_or_path='pretrained/siglip-so400m-patch14-384',
# )

# COGACT_CONFIG = dict(
#     saved_model_path='CogACT/CogACT-Base',
#     unnorm_key=None,
#     policy_setup="widowx_bridge",
#     horizon=0,
#     action_ensemble_horizon=None,
#     image_size=[224, 224],
#     future_action_window_size=15,
#     action_dim=7,
#     action_model_type="DiT-B",
#     action_scale=1.0,
#     cfg_scale=1.5,
#     use_ddim=True,
#     num_ddim_steps=10,
#     use_bf16=True,
#     action_ensemble=True,
#     adaptive_ensemble_alpha=0.1,
# )

OPEN_PIZERO_CONFIG = dict(
    cfg_dir='open_pi_zero/config/eval',
    use_ddp=False,
    use_naive=False,
    use_torch_compile=True,
)

CONTRAST_IMAGE_CONFIG = dict(
    camera_name=None,
    by="gt",
    inpaint_mode="lama",
    color="auto",
    sigma=5,
    version=2,
    get_all_parts=False,
)

CONTRAST_OCTO_CONFIG = dict(
    alpha=0.2,
    num_repeats=24,
    bandwidth_factor=2.0,
    keep_threshold=0.5,
)

CONTRAST_OPENVLA_CONFIG = dict(
    alpha=0.2,
)

# CONTRAST_COGACT_CONFIG = dict(
#     alpha=0.1,
#     num_repeats=64,
#     bandwidth_factor=1.0,
# )

# CONTRAST_OPEN_PIZERO_CONFIG = dict(
#     alpha=0.3,
#     bandwidth_factor=1.0,
#     num_repeats=16,
#     keep_threshold=0.5,
# )

CONTRAST_OPEN_PIZERO_CONFIG = dict(
    alpha=0.2,
    num_repeats=20,
    bandwidth_factor=1.0,
    keep_threshold=0.5,
)

def get_policy_config(policy, checkpoint, task, opts, contrast):
    if policy == 'rt1':
        config = RT1_CONFIG
        config['saved_model_path'] = checkpoint
    elif policy == 'octo':
        config = OCTO_CONFIG
        config['model_type'] = checkpoint
    elif policy == 'openvla':
        config = OPENVLA_CONFIG
        config['saved_model_path'] = checkpoint
    elif policy == 'rdt':
        config = RDT_POLICY_CONFIG
        pretrained = checkpoint
    elif policy == 'cogact':
        config = COGACT_CONFIG
        config['saved_model_path'] = checkpoint
    elif policy == 'pizero':
        config = OPEN_PIZERO_CONFIG
        config['checkpoint_path'] = checkpoint
    else:
        raise NotImplementedError()
    
    # select policy setup based on task
    if task.startswith('google_robot'):
        config['policy_setup'] = 'google_robot'
    elif task.startswith('widowx'):
        config['policy_setup'] = 'widowx_bridge'
    else:
        raise NotImplementedError

    # update config if contrast policy is used
    if contrast:
        from properties import CONTRAST_OCTO_CONFIG, CONTRAST_OPENVLA_CONFIG
        if policy == 'octo':
            config.update(CONTRAST_OCTO_CONFIG)
        elif policy == 'openvla':
            config.update(CONTRAST_OPENVLA_CONFIG)
        elif policy == 'pizero':
            config.update(CONTRAST_OPEN_PIZERO_CONFIG)
        else:
            raise NotImplementedError()
    
    # update opts
    for k, v in opts.items():
        if k in config:
            config[k] = v
    
    return config


def get_contrast_image_generator_config(opts):
    config = CONTRAST_IMAGE_CONFIG
    for k, v in opts.items():
        if k in config:
            config[k] = v
    return config
