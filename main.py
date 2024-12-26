import os

# os.environ["WANDB_API_KEY"]=" "
# os.environ["WANDB_MODE"]="dryrun" # offline
# os.environ["http_proxy"] = "http://u-KgKRFF:5XvYKmDW@10.255.128.102:3128"
# os.environ["https_proxy"] = "http://u-KgKRFF:5XvYKmDW@10.255.128.102:3128"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateFinder, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import platform
import importlib
import torch
from models.utils import seed_all
from config import config
from lightning_data import LightningData

seed_all(config.seed)
# np.seterr(all='raise')


LightningModel = importlib.import_module(f"models.{config.model}").LightningModel

if config.platform != "Darwin":
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")  # optional: 'highest', 'high', 'medium'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 或者 "true"


logger = TensorBoardLogger(
    save_dir=config.logs_dir,
    name="",  # 不加的话，会保存在子目录lightning_logs中
    #  Enables a placeholder metric with key `hp_metric` when `log_hyperparams` is called without a metric (otherwise calls to log_hyperparams without a metric are ignored). prefix: A string to put at the beginning of metric keys.
    default_hp_metric=False,
    # name=f"s-{config['data_time']}-{config['mark']}",
    version="",
)

# logger = WandbLogger(
#     project="emotion_clss",
#     config=config,
#     save_dir=config.logs,
#     name=f"s{config['data_time']}",  # 网站中运行对应名字
#     id=None,  # 本地文件中最后的后缀
# )

checkpoint_callback = ModelCheckpoint(
    monitor=config.monitor,  #
    mode=config.monitor_mode,
    save_top_k=1 if config.platform != "Darwin" else 0,  # default is 1
    # save_last=True,  # default is None
    every_n_epochs=1,  # default is None
    # every_n_train_steps=None,  # default is None
    dirpath=config.checkpoints_dir,
    filename="best",
)

earlystop_callback = EarlyStopping(
    monitor=config.monitor,
    mode=config.monitor_mode,
    patience=5,
    verbose=False,
    check_finite=True,
)


class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)


trainer = L.Trainer(
    accelerator=config.accelerator,
    devices=config.devices,
    strategy=config.strategy,
    precision=config.precision,
    # fast_dev_run=True,
    # accelerator="cpu",
    accumulate_grad_batches=config.accumulate_grad_batches,
    gradient_clip_val=config.gradient_clip_val,
    check_val_every_n_epoch=1,  # default is 1
    logger=logger,
    callbacks=[
        checkpoint_callback,
        # earlystop_callback,
    ],
    deterministic="True",  # default is None, optional is True/"warn"
    log_every_n_steps=1,  # default is 50/None, mean log every 50 steps and the end of epoch, 0 means log at end of epoch
    # profiler=AdvancedProfiler(),  # default is None, optional is SimpleProfiler, AdvancedProfiler
    # default_root_dir="logs",
    # enable_checkpointing=False,
    max_epochs=config.max_epochs if config.platform != "Darwin" else 1,
    limit_train_batches=None if config.platform != "Darwin" else 1,
    limit_val_batches=None if config.platform != "Darwin" else 1,
    limit_test_batches=None if config.platform != "Darwin" else None,
)


model = LightningModel(config)
dm = LightningData()

trainer.fit(model, datamodule=dm)
trainer.test(ckpt_path="best", dataloaders=dm)  # best or last

# trainer_test = L.Trainer()
# trainer_test.test(model, dataloaders=dm)


# model = LightningModel.load_from_checkpoint("checkpoints/s12-10-23:09/best.ckpt", config=config, strict=False)
# dm = LightningData()
# trainer.test(model, dataloaders=dm)


# model = LightningModel.load_from_checkpoint("checkpoints/s10-19-12:51/best.ckpt", config=config, strict=False)
# dm = LightningData()
# trainer_predict = L.Trainer(enable_progress_bar=False)
# trainer.predict(model, dataloaders=dm)
