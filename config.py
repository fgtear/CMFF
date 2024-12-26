import datetime
import platform
import copy
from math import sqrt
import torch


class Config:
    def __init__(self):
        self.platform = platform.system()
        if self.platform == "Darwin":
            self.devices = "auto"
            self.strategy = "auto"
        else:
            self.devices = [i for i in range(torch.cuda.device_count())]
            if len(self.devices) > 1:
                # auto, ddp, ddp_find_unused_parameters_true,
                self.strategy = "ddp_find_unused_parameters_true" if self.platform != "Darwin" else "auto"
            else:
                self.strategy = "auto"
        self.accelerator = "auto" if self.platform != "Darwin" else "cpu"
        self.num_workers = 13 if self.platform != "Darwin" else 0
        self.num_GPU = len(self.devices) if self.platform != "Darwin" else 1
        self.precision = "32" if self.platform != "Darwin" else "32"  # 16-mixed, 32

        self.dataset = "mosi"  # optional: mosi, mosei
        self.model = "pretrain"  # pretrain, cmsq, concat
        self.text_extractor_path = "model_hub/bart-large"  # roberta-large, bart-large,
        self.audio_extractor_path = "model_hub/data2vec-audio-base-960h"
        self.seed = 1
        self.max_epochs = 22  # default is -1 for infinite
        self.learning_rate = 1e-5 * self.num_GPU
        self.dropout = 0.3
        self.weight_decay = 1e-3  # default is
        self.batch_size_train = 8 if self.platform != "Darwin" else 3
        self.accumulate_grad_batches = 1  # default is 1
        self.batch_size_eval = 64 if self.platform != "Darwin" else 32  # include val and test
        self.text_max_length = 88
        self.audio_max_length = 163840  # 163840, 327680,  655360
        self.gradient_clip_val = 0  # default is None
        self.data_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
        self.logs_dir = "logs/s" + self.data_time
        self.checkpoints_dir = "checkpoints/s" + self.data_time

        self.monitor = "val/MAE_val"
        self.monitor_mode = "min"
        if "pretrain" in self.model:
            self.monitor = "train/loss_val"
            self.monitor_mode = "min"

    def get_hparams(self):
        dic = copy.deepcopy(self.__dict__)
        dic["description"] = "  "
        ##########################################################
        dic["learning_rate"] = str(dic["learning_rate"])  # tensorboard记录的lr低于1e-4都当成1e-4
        if "text" in dic["model"]:
            dic["model"] = dic["model"] + "-" + self.text_extractor_path.split("/")[-1]
        elif "data2vec" in dic["model"]:
            dic["model"] = dic["model"] + "-" + self.audio_extractor_path.split("/")[-1]
        elif "cmsq" in dic["model"] or "ts" in dic["model"]:
            dic["model"] = dic["model"] + "-" + self.text_extractor_path.split("/")[-1] + "-" + self.audio_extractor_path.split("/")[-1]
        elif "pretrain" in dic["model"]:
            dic["model"] = dic["model"] + "-" + self.text_extractor_path.split("/")[-1] + "-" + self.audio_extractor_path.split("/")[-1]
        else:
            raise ValueError("--------------------------")
        ##########################################################
        dic.pop("strategy")
        # dic.pop("precision")
        dic.pop("accelerator")
        dic.pop("batch_size_eval")
        dic.pop("num_GPU")
        dic.pop("num_workers")
        dic.pop("data_time")
        dic.pop("logs_dir")
        dic.pop("checkpoints_dir")
        dic.pop("text_extractor_path")
        dic.pop("audio_extractor_path")
        dic.pop("monitor")
        dic.pop("monitor_mode")
        # print("data_time:", self.data_time)
        return dic


config = Config()
