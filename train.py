import os
import torch
import wandb
import json
import hydra
import importlib
import pandas as pd
from config import Config
from datetime import datetime
from omegaconf import OmegaConf
from datasets.data_module import DataModule

from utils.path_utils import PathUtils

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything


def get_model_name(cfg: Config):
    model_name = datetime.now().strftime("%m_%d_%H_%M_%S")

    if cfg.model.params.get("model_base_name", False):
        model_name = f"{cfg.model.params.model_base_name}_{model_name}"
    return model_name


def get_label_names(label_json_file_path):
    labels = json.load(open(label_json_file_path, "r"))
    label_names = list(labels.keys())
    return label_names


def get_model_class(model_name):
    module_path, class_name = model_name.rsplit(".", 1)
    module = importlib.import_module(f"models.{module_path}")
    model_class = getattr(module, class_name)
    return model_class


def save_params_locally(model_base, model_name):
    print("--------------------------------------------------------------------------")
    print("Creating experiment folder to save model checkpoint and metrics...")
    root_path = "./metrics"
    base_folder_path = os.path.join(root_path, model_base)
    output_folder = (
        datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        if model_name is None
        else model_name
    )
    model_folder_path = os.path.join(base_folder_path, output_folder)
    PathUtils.create_dir(model_folder_path, show_message=False)
    print("Folder path:", model_folder_path)

    return model_folder_path


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: Config):
    # ---------------------------------------------------------------------------- #
    #                                   Set seed                                   #
    # ---------------------------------------------------------------------------- #
    seed_everything(cfg.data_module.random_state, workers=True)

    # ---------------------------------------------------------------------------- #
    #         Create experiment folder to save model checkpoint and metrics        #
    # ---------------------------------------------------------------------------- #
    MODEL_NAME = get_model_name(cfg)
    print(MODEL_NAME)
    if not cfg.train.dev_run:
        model_folder_path = save_params_locally(
            model_base=cfg.model.params.model_base_name, model_name=MODEL_NAME
        )

        # Save parameter information to json file
        print(f"Saving parameters to {model_folder_path}/parameters.json")
        with open(f"{model_folder_path}/parameters.json", "w") as f:
            json.dump(OmegaConf.to_object(cfg), f, indent=4)
        print(
            "--------------------------------------------------------------------------"
        )
    else:
        model_folder_path = None

    # ---------------------------------------------------------------------------- #
    #                                  Data Module                                 #
    # ---------------------------------------------------------------------------- #
    data_module = DataModule(
        **cfg.data_module,
    )

    # ---------------------------------------------------------------------------- #
    #                                     Model                                    #
    # ---------------------------------------------------------------------------- #
    ModelClass = get_model_class(cfg.model.name)
    model = ModelClass(
        num_classes=cfg.train.num_classes,
        label_names=get_label_names(cfg.train.needed_labels_file_path),
        model_name=MODEL_NAME,
        **cfg.model.params,
    )

    # ---------------------------------------------------------------------------- #
    #                                   Callbacks                                  #
    # ---------------------------------------------------------------------------- #

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_folder_path,
        monitor="val_MulticlassAccuracy",
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
        filename="{epoch:02d}-{val_loss:.4f}-{val_MulticlassAccuracy:.4f}",
        verbose=False,
        mode="max",
    )
    # Earlystopping
    earlystopping = EarlyStopping(
        monitor=cfg.model.params.monitor_metric,
        patience=cfg.model.params.stoping_patience,
        mode=cfg.model.params.monitor_mode,
    )

    # ---------------------------------------------------------------------------- #
    #                                    Logger                                    #
    # ---------------------------------------------------------------------------- #
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.wandb_project,
        name=MODEL_NAME,
        # config=cfg,
        log_model=False,
        # mode=cfg.wandb.wandb_mode,  # online, disabled
    )

    # ---------------------------------------------------------------------------- #
    #                                 Model Trainer                                #
    # ---------------------------------------------------------------------------- #
    # Set torch float precision
    torch.set_float32_matmul_precision("medium")
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
    # Train model
    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=cfg.train.epochs,
        fast_dev_run=cfg.train.dev_run,
        logger=wandb_logger,
        callbacks=[earlystopping],
        # deterministic=True,  # To set random seed
    )
    trainer.fit(model, data_module)
    # print(model.hparams)

    # Test model
    if not cfg.train.dev_run:
        trainer.test(model, datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
