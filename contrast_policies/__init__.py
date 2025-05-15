def get_policy(policy, contrast, config):
    if policy == 'octo':
        if not contrast:
            from simpler_env.policies.octo.octo_model import OctoInference
            policy = OctoInference(**config)
        else:
            from .octo_contrast import OctoContrastInference
            policy = OctoContrastInference(**config)
    elif policy == 'openvla':
        if not contrast:
            from simpler_env.policies.openvla.openvla_model import OpenVLAInference
            policy = OpenVLAInference(**config)
        else:
            from .openvla_contrast import OpenVLAContrastInference
            policy = OpenVLAContrastInference(**config)
    elif policy == 'pizero':
        if not contrast:
            from simpler_env.policies.pizero.pizero_model import PiZeroInference
            policy = PiZeroInference(**config)
        else:
            from .pizero_contrast import PiZeroContrastInference
            policy = PiZeroContrastInference(**config)
    else:
        raise NotImplementedError()
    
    return policy
