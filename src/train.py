import pytorch_lightning as pl
import hydra
import torch
import torchvision.transforms as transforms
import transform as T

from omegaconf import DictConfig
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import LightningModelWrapper, BaseModel
from util import ExperimentLoader
from dataset import LightningDatasetWrapper


@hydra.main(config_path="../experiment", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load experiment parameters
    cfg = ExperimentLoader.load_data(cfg)

    # create model
    model = BaseModel(**cfg.model)
    wrapped_model = LightningModelWrapper(
        model=model,
        batch_size=cfg.data.dataset.data_loader.batch_size,
        optim_parameters=cfg.train.optimizer,
        **cfg.train.loss
    )

    # prepare dataset module
    data_module = LightningDatasetWrapper(
        train_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            # For STL10
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # For CIFAR10
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            # For STL10
            # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # For CIFAR10
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        **cfg.data.dataset
    )

    # define wandb logger
    logger = WandbLogger(project="clp", save_code=True, save_dir=cfg.log.dir, log_model="all")   # NOQA
    logger.experiment.log_code(hydra.utils.get_original_cwd())

    # saving the best model
    model_checkpoint = ModelCheckpoint(monitor="val/acc-1", mode="min")

    trainer = pl.Trainer(
        devices=cfg.train.gpus,
        logger=logger,
        accelerator="gpu",
        auto_select_gpus=True,
        check_val_every_n_epoch=None,
        strategy=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            model_checkpoint
            # TODO wandb image callback
        ],
        **cfg.train.trainer
    )

    # train model
    logger.watch(wrapped_model)
    trainer.fit(wrapped_model, data_module)


if __name__ == "__main__":
    main()
