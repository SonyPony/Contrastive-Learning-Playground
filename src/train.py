import sys
if not sys.platform == "win32":
    from termcolor import cprint
    from safe_gpu import safe_gpu

    available_gpus = safe_gpu.get_free_gpus()
    cprint(f"Available GPUs: {available_gpus}", color="yellow")
    #safe_gpu.GPUOwner()

import pytorch_lightning as pl
import hydra
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import transform as T
import wandb

from omegaconf import DictConfig
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.tinyimagenet import TinyImageNet
from model import LightningModelWrapper, BaseModel
from common.training_type import TrainingType
from util import ExperimentLoader
from dataset import LightningDatasetWrapper


@hydra.main(config_path="../experiment", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    torch.multiprocessing.set_sharing_strategy('file_system')

    # load experiment parameters
    cfg = ExperimentLoader.load_data(cfg)

    classes_count = cfg.data.dataset.num_classes
    cfg.data.dataset["num_classes"] = classes_count if classes_count else TinyImageNet.CLASSES_COUNT
    training_type = TrainingType(cfg.train.type)

    # create model
    model = BaseModel(
        classes_count=cfg.data.dataset.num_classes,
        training_type=training_type,
        feature_size=cfg.model.feature_size
    )

    wrapped_model = LightningModelWrapper(
        model=model,
        batch_size=cfg.data.dataset.data_loader.batch_size,
        optim_parameters=cfg.train.optimizer,
        experiment_cfg=cfg,
        training_type=training_type,
        **cfg.train.loss
    )
    wrapped_model.load_model()

    # prepare dataset module
    data_module = LightningDatasetWrapper(
        train_transform=transforms.Compose([
            transforms.Resize(48, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=int(0.1 * 32)),
            transforms.ToTensor(),
            # For STL10
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # For CIFAR10
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        test_transform=transforms.Compose([
            transforms.Resize(32, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # For STL10
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            # For CIFAR10
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        **cfg.data.dataset
    )

    # define wandb logger
    logger = WandbLogger(project="clp", save_code=True, save_dir=cfg.log.dir, log_model="all", notes=cfg.log.notes)   # NOQA
    logger.experiment.log_code(hydra.utils.get_original_cwd())

    # saving the best model
    model_checkpoint = ModelCheckpoint(monitor="val/acc-1", mode="max")

    trainer = pl.Trainer(
        devices=cfg.train.gpus,
        logger=logger,
        accelerator="gpu",
        check_val_every_n_epoch=None,
        replace_sampler_ddp=False,
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
