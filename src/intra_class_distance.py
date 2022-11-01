import sys
from tqdm import tqdm

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

    device = "cuda:0"
    data_module.setup()
    test = data_module.val
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()

    classes = {} # TODO parametrize num classes

    with torch.no_grad():
        for i, xi in tqdm(enumerate(test)):
            (xi, _), xi_label, _ = xi
            xi = xi.to(device)
            xi_features, _ = wrapped_model(xi[None, ...])

            for j, xj in enumerate(test):
                if i == j:
                    continue
                (xj, _), xj_label, _ = xj
                xj = xj.to(device)
                if xj_label == xi_label:
                    continue

                xj_features, _ = wrapped_model(xj[None, ...])

                if not xi_label in classes.keys():
                    classes[xi_label] = []
                classes[xi_label].append(torch.dot(
                    torch.squeeze(xj_features),
                    torch.squeeze(xi_features)
                ).to("cpu"))

    torch.save(classes, f"{hydra.utils.get_original_cwd()}/data_1024_neg.pth")




if __name__ == "__main__":
    main()
