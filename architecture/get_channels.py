# from architecture import calc_channel_size, calc_latent_size
import architecture

def get_channels(
        # strat='paired_exponential',
        # strat,
        
        start_idx=5,
        stop_idx=6,
        **kwargs
    ):
    strat = kwargs['strat']
    channel_options = []
    if strat == 'exponential':
        # start_idx = 6
        # stop_idx = 8
        for idx in range(start_idx, stop_idx):
            channels = [
                1, # init channels`
                *[2**idx for idx in range(start_idx, idx+2)]
            ]
            channel_options.append(channels)
    elif strat=='paired_exponential':
        for idx in range(start_idx, stop_idx):
            c = [1]
            for i in range(start_idx, idx+1):
                val = 2**i
                c.extend([val, val, val])
            channel_options.append(
                c
            )
            c = [1]
            for i in range(start_idx, idx+1):
                val = 2**i
                c.extend([val, val, 2*val])
            channel_options.append(
                c
            )
            c = [1]
            for i in range(start_idx, idx+1):
                val = 2**i
                c.extend([val, 2*val, 2*val])
            channel_options.append(
                c
            )

    elif strat == 'from_to':
        input_size = kwargs['input_size']
        # to = 
        num_latent = 128
        strides = []
        kernel_sizes = []
        test_channels_options = get_channels(
            # start_exponent,
            # depth
            start_idx=5, 
            stop_idx=9, 
            strat='paired_exponential'
        ) 
        for test_channels in test_channels_options:
            for stride in range(1, 5):
                for kernel_size in [3, 4, 5, 7]:
                    channel_sizes, _ = architecture.calc_channel_size(
                        input_size,
                        test_channels,
                        kernel_size,
                        stride
                    )
                    if channel_sizes[-1] < 1: # if resulting in a (negative) nonsensical final output dim 
                        continue
                    sfs
                    architecture.actual_latent_size = architecture.calc_latent_size(_from, channels, kernel_size, stride)
                    if  actual_latent_size > kwargs['latent_dim']: #if this arch doesnt result in a latent size of near what we want
                        continue # continue to the next trial without adding it to the options
                    else:
                        strides.append(stride)
                        kernel_sizes.append(kernel_size)
                        channel_options.append(test_channels)

        print('channel_options:', channel_options)
        print(kernel_sizes)
        print(strides)
        fewsdf
        return channel_options, kernel_sizes, strides

    else:
        raise ValueError('Strategy for getting channels not specified')

    return channel_options