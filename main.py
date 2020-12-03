import sys
sys.path.append('autoencoders')
from ConvAutoencoder import ConvAutoencoder
from ae_utils import sample
import torch

def load_and_sample(path='/home/ubuntu/ray_results/train_tune_2020-12-03_21-00-28/train_tune_92be1_00001_1_channels=[1, 32, 32],lr=0.001,optimiser=adam_2020-12-03_21-00-31/checkpoint_0/checkpoint'):
    state_dict, model_config, optimiser_state = torch.load(path)
    print(model_config)
    best_model = ConvAutoencoder(**model_config)
    best_model.load_state_dict(state_dict)

    sample(best_model.decoder, 32)

if __name__ == '__main__':
    load_and_sample()