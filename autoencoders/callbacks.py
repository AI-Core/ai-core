from pytorch_lightning.callbacks import Callback
import torch

class SampleReconstructionCallback(Callback):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def on_validation_epoch_end(self, trainer, pl_module): # already in eval mode at this point
        for batch in self.loader:
            break
        originals, _ = batch
        originals = originals[:8]
        reconstructions = pl_module(originals)
        # x = x.view(x.shape[0], 1, 28, 28)
        # reconstructions = reconstructions.view(reconstructions.shape[0], 1, 28, 28)
        imgs = torch.cat((originals, reconstructions), 0)
        writer = pl_module.logger.experiment
        # writer.add_images(f'originals/{pl_module.experiment_name}', originals)
        # writer.add_images(f'reconstructions/{pl_module.experiment_name}', reconstructions)
        writer.add_images(f'reconstructions/{pl_module.experiment_name}', imgs)
        # visualise_reconstruction(pl, x, reconstructions, f'epoch-{trainer.current_epoch}')