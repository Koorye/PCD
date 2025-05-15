import jax
import jax.numpy as jnp
import math
import torch


def gaussian_kernel_torch(x):
    return (1 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x ** 2)

def gaussian_kernel_jax(x):
    return (1 / math.sqrt(2 * math.pi)) * jnp.exp(-0.5 * x ** 2)

def scott_rule_torch(data):
    return 1.06 * torch.std(data, dim=-1) * data.shape[-1] ** (-1 / 5)

def scott_rule_jax(data):
    return 1.06 * jnp.std(data, axis=-1) * data.shape[-1] ** (-1 / 5)

def kde_jax(x, data, bandwidth, kernel_func):
    # x -> (N, B) -> (N, B, 1)
    x = jnp.expand_dims(x, axis=-1)
    # data -> (N, B) -> (N, 1, B)
    data = jnp.expand_dims(data, axis=1)
    # (N, B, 1) - (N, 1, B) -> (N, B, B)
    bandwidth_ = bandwidth[:, None, None] if not isinstance(bandwidth, float) else bandwidth
    kernel_values = kernel_func((x - data) / bandwidth_)
    # (N, B, B) -> (N, B)
    bandwidth_ = bandwidth[:, None] if not isinstance(bandwidth, float) else bandwidth
    density_estimation = jnp.sum(kernel_values, axis=-1) / (data.shape[-1] * bandwidth_)
    return density_estimation

def kde_torch(x, data, bandwidth, kernel_func):
    # x -> (N, B) -> (N, B, 1)
    x = x.unsqueeze(-1)
    # data -> (N, B) -> (N, 1, B)
    data = data.unsqueeze(1)
    # (N, B, 1) - (N, 1, B) -> (N, B, B)
    bandwidth_ = bandwidth.unsqueeze(-1).unsqueeze(-1) if not isinstance(bandwidth, float) else bandwidth
    kernel_values = kernel_func((x - data) / bandwidth_)
    # (N, B, B) -> (N, B)
    bandwidth_ = bandwidth.unsqueeze(-1) if not isinstance(bandwidth, float) else bandwidth
    density_estimation = kernel_values.sum(dim=-1) / (data.shape[-1] * bandwidth_)
    return density_estimation

def take_index_torch(prob, mode='greedy'):
    prob = prob.clone()
    if mode == 'greedy':
        return prob.argmax(dim=-1)
    elif mode == 'sample':
        prob[torch.isnan(prob)] = 0.0
        prob += 1e-8
        prob = prob / prob.sum(dim=-1, keepdims=True)
        return torch.multinomial(prob, 1).squeeze(-1)
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")

def take_index_jax(rng, prob, mode='greedy'):
    if mode == 'greedy':
        return jnp.argmax(prob, axis=-1)
    elif mode == 'sample':
        prob = jnp.where(jnp.isnan(prob), 0.0, prob)
        prob += 1e-8
        prob = prob / jnp.sum(prob, axis=-1, keepdims=True)
        return jax.random.categorical(rng, prob)


class ContrastDecoding:
    def __init__(self,
                 alpha=0.1,
                 bandwidth_factor=1.0,
                 keep_threshold=0.5,
                 mode='torch'):
        self.alpha = alpha
        self.bandwidth_factor = bandwidth_factor
        self.keep_threshold = keep_threshold
        self.mode = mode
        
        if self.mode == 'jax':
            self.rng = jax.random.PRNGKey(0)
    
    def __call__(self, data, contrast_data):
        if self.mode == 'torch':
            return self.decode_torch(data, contrast_data)
        elif self.mode == 'jax':
            return self.decode_jax(data, contrast_data)
        else:
            raise NotImplementedError()
        
    def decode_torch(self, data, contrast_data):
        B, T, D = data.shape
        N = T * D
        data = data.reshape(B, N).permute(1, 0)
        contrast_data = contrast_data.reshape(B, N).permute(1, 0)
        
        bandwidth = self.bandwidth_factor * scott_rule_torch(data)
        prob = kde_torch(data, data, bandwidth, gaussian_kernel_torch)
        
        contrast_bandwidth = self.bandwidth_factor * scott_rule_torch(contrast_data)
        contrast_prob = kde_torch(data, contrast_data, contrast_bandwidth, gaussian_kernel_torch)
        
        contrast_factor = prob / contrast_prob
        final_prob = prob * contrast_factor ** self.alpha
        
        final_prob[prob < self.keep_threshold * prob.max(dim=-1, keepdims=True).values] = 0.0
        final_prob = final_prob / final_prob.max(dim=-1, keepdims=True).values * prob.max(dim=-1, keepdims=True).values
        
        # [N, B] -> [N,] -> [T, D]
        sample = data[range(N), prob.argmax(dim=-1)].reshape(T, D)
        contrast_sample = data[range(N), final_prob.argmax(dim=-1)].reshape(T, D)

        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # for i in range(3):
        #     plt.subplot(3, 1, i + 1)
        #     sns.lineplot(x=data[i].float().cpu().numpy(), y=prob[i].float().cpu().numpy(), color='blue')
        #     sns.lineplot(x=data[i].float().cpu().numpy(), y=contrast_prob[i].float().cpu().numpy(), color='red')
        #     sns.lineplot(x=data[i].float().cpu().numpy(), y=final_prob[i].float().cpu().numpy(), color='green')
        # plt.savefig('visualize/pi0.jpg')
        # plt.close()

        # update x, y, z, roll, pitch, yaw
        sample[:, :6] = contrast_sample[:, :6]
        return sample.unsqueeze(0)

    def decode_jax(self, data, contrast_data):
        B, T, D = data.shape
        N = T * D
        data = data.reshape(B, N).transpose(1, 0)
        contrast_data = contrast_data.reshape(B, N).transpose(1, 0)
        
        bandwidth = self.bandwidth_factor * scott_rule_jax(data)
        prob = kde_jax(data, data, bandwidth, gaussian_kernel_jax)

        contrast_bandwidth = self.bandwidth_factor * scott_rule_jax(contrast_data)
        contrast_prob = kde_jax(data, contrast_data, contrast_bandwidth, gaussian_kernel_jax)

        contrast_factor = prob / contrast_prob
        final_prob = prob * contrast_factor ** self.alpha
        final_prob = final_prob.at[prob < self.keep_threshold * prob.max(axis=-1, keepdims=True)].set(0.0)
        final_prob = final_prob / jnp.max(final_prob, axis=-1, keepdims=True) * prob.max(axis=-1, keepdims=True)
        
        # [N, B] -> [N,] -> [T, D]
        sample = data[jnp.arange(N), prob.argmax(axis=-1)].reshape(T, D)
        contrast_sample = data[jnp.arange(N), final_prob.argmax(axis=-1)].reshape(T, D)

        # update x, y, z, roll, pitch, yaw
        sample = jnp.concatenate([contrast_sample[:, :6], sample[:, 6:]], axis=-1)
        return jnp.expand_dims(sample, axis=0)
