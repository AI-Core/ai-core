
def get_channels(strat='paired_exponential'):
    channel_options = []
    if strat == 'exponential':
        start_idx = 6
        stop_idx = 8
        for idx in range(start_idx, stop_idx):
            channels = [
                1, # init channels`
                *[2**idx for idx in range(start_idx, idx+2)]
            ]
            channel_options.append(channels)
    elif strat=='paired_exponential':
        start_idx = 5
        stop_idx = 6
        for idx in range(start_idx, stop_idx):
            c = [1]
            for i in range(start_idx, idx+1):
                val = 2**i
                c.extend([val, val, val])
            channel_options.append(
                c
            )

    elif strat == 'from_to':
        _from = 28
        # to = 
        num_latent = 128
        stride = 4

    else:
        raise ValueError('Strategy for getting channels not specified')

    return channel_options
