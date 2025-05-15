import jax
import tensorflow as tf
import torch

# test gpu 
def test_gpu():
    print("JAX GPU avaliable: ", jax.devices())
    print("TF GPU avaliable: ", tf.config.list_physical_devices('GPU'))
    print("Torch GPU avaliable: ", torch.cuda.is_available())
    
test_gpu()
