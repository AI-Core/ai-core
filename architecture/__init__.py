from architecture.get_channels import get_channels

def get_ae_architecture(input_size, latent_dim=128, encoder_depth=3, decoder_depth=4, channel_promotion_strategy='exponential'):

    """ 
    Find what channel depths, kernel_sizes, strides produce a reduction of size from input_size to latent_dim, over the given layer depths
    """

    # stride = config['stride']
    # kernel_size = config['kernel_size']

    _from = input_size
    to = round(0.2 * input_size)
    channels, kernel_sizes, strides = get_channels(input_size=28, output_dim=128, strat='from_to')

    # calc_latent_size(input_size, channels, kernel_size, stride)

    channel_sizes, remainders = calc_channel_size(28, channels, kernel_size, stride)
    calc_transpose_channel_size(channel_sizes[-1], channels[::-1], kernel_size, stride, remainders)
    output_padding = remainders[::-1], # need to reverse to mirror order of layers and apply matching

    linear_layers = get_linear_layers(channels, latent_dim)


    # return every possible
    ae_architecture = {
        'output_padding': output_padding,
        'encoder_channels': channels,
        'encoder_linear_layers': linear_layers,
        'encoder_kernel_size': kernel_size,
        'encoder_stride': stride,
        'decoder_channels': channels[::-1],
        'decoder_linear_layers': linear_layers[::-1],
        'decoder_kernel_size': kernel_size,
        'decoder_stride': stride,
        'decoder_padding': output_padding,
        'verbose': False
    }
    return ae_architecture


# automatically get the width
# 
def get_conv_architecture(input_size, output_size=[2, 3, 4]):
    """
    return an architecture required to reduce an input to a given dimensionality
    """


def calc_channel_size(w, channels, kernel_size, stride):
    # print('calculating conv layer sizes')
    c = []
    remainders = []
    for c_idx in range(len(channels)-1):
        remainder = (w - kernel_size) % stride
        w = (w - kernel_size) // stride + 1 # floor disision if not actually butting up to opposite corner
        # print(f'\tchannel {c_idx+1}\t size: {w}')
        # if remainder != 0:
            # print('\t\t^not perfect')
        remainders.append(remainder)
        c.append(w)
    return c, remainders

def calc_transpose_channel_size(w, channels, kernel_size, stride, padding):
    # print('calculating conv transpose layer sizes')
    c = []
    for c_idx in range(len(channels)-1):
        w = (w + padding[c_idx] - 1) * stride + kernel_size # if the last kernel did not perfectly butt up to the edge, then you need to add padding
        # (w - kernel_size) // stride + 1
        # print(f'\tchannel {c_idx+1}\t size: {w}') 
        c.append(w)
    return c

def calc_latent_size(w, channels, kernel_size, stride):
    c, _ = calc_channel_size(w, channels, kernel_size, stride)
    final_channel_width = c[-1]
    latent_size = final_channel_width * final_channel_width * channels[-1]
    # print('latent size:', latent_size)
    return latent_size

def get_linear_layers():
    pass