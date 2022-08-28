import pytorch_lightning as pl
import hydra
import torch
import torchvision.transforms as transforms

from omegaconf import DictConfig
from pytorch_lightning.plugins import DDPPlugin

from model import LightningModelWrapper, BaseModel
from util import ExperimentLoader
from dataset import LightningSTL10Pair


@hydra.main(config_path="../experiment", config_name="base")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load experiment parameters
    cfg = ExperimentLoader.load_data(cfg)

    model = BaseModel(**cfg.model)
    wrapped_model = LightningModelWrapper(model=model, optim_parameters=cfg.train.optimizer)
    data_module = LightningSTL10Pair(
        train_transform=transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor()
        ]),
        **cfg.data.dataset
    )

    # TODO logger and strategy
    trainer = pl.Trainer(
        gpus=cfg.train.gpus,
        strategy=DDPPlugin(find_unused_parameters=False),
        **cfg.train.trainer
    )
    trainer.fit(wrapped_model, data_module)


    #model = STL10Pair(root="../data", split="train+unlabeled")
    #b = model[4]

if __name__ == "__main__":
    main()
