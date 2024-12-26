import lightning as L
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer

# from speechtokenizer import SpeechTokenizer
import typing
from config import config


class DatasetMosi(torch.utils.data.Dataset):
    def __init__(self, mode: typing.Literal["train", "test", "valid"]):
        self.data_root = "data/MOSI"
        self.num_data = [1284, 229, 686]  # train, valid, test
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_extractor_path, clean_up_tokenization_spaces=True)
        self.prepare_data()
        if mode == "train":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[0]
        elif mode == "valid":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[1]
        elif mode == "test":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[2]

    def prepare_data(self):
        df = pd.read_csv(self.data_root + "/label.csv")
        df = df[df["mode"] == self.mode].sort_values(by=["video_id", "clip_id"]).reset_index()
        # df["text"] = df["text"].str[0] + df["text"].str[1::].apply(lambda x: x.lower())  # Capitalize the first letter
        df["text"] = df["text"].apply(lambda x: x.lower())  # lowercase all
        self.texts = df["text"]
        self.labels = df["label"]
        # self.video_id = df["video_id"]

        self.audio_paths = []
        for i in range(0, len(df)):
            file_name = str(df["video_id"][i]) + "/" + str(df["clip_id"][i]) + ".wav"
            file_path = self.data_root + "/wav/" + file_name
            self.audio_paths.append(file_path)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text_token = self.tokenizer(
            text,
            return_attention_mask=True,
            max_length=config.text_max_length,
            truncation=True,
            add_special_tokens=True,  # [CLS], [SEP], <s>, etc.  TODO:
            return_tensors="pt",
        )
        audio_path = self.audio_paths[index]
        audio_wave, sample_rate = torchaudio.load(audio_path)  # [channel=2, T]
        # audio_wave = audio_wave[:, -config.audio_max_length :]
        audio_wave = audio_wave[:, : config.audio_max_length]  # TODO:
        audio_wave = torch.mean(audio_wave, dim=0, keepdim=False)  # [T]
        audio_wave = (audio_wave - audio_wave.mean()) / (torch.sqrt(audio_wave.var()) + 1e-7)  # # TODO: 0均值，1方差
        return {
            "text": text,
            "text_input_ids": text_token["input_ids"].squeeze(),
            "text_attention_mask": text_token["attention_mask"].squeeze(),
            "audio_path": audio_path,
            "audio_wave": audio_wave,
            "sample_rate": sample_rate,
            "audio_length": audio_wave.size(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.float),
        }

    def __len__(self):
        return len(self.labels)


class DatasetMosei(DatasetMosi):
    def __init__(self, mode: typing.Literal["train", "valid", "test"]):
        self.data_root = "data/MOSEI"
        self.num_data = [16326, 1871, 4659]  # train, valid, test
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_extractor_path, clean_up_tokenization_spaces=True)
        self.prepare_data()
        if mode == "train":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[0]
        elif mode == "valid":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[1]
        elif mode == "test":
            assert len(self.labels) == len(self.audio_paths) == self.num_data[2]


class LightningData(L.LightningDataModule):
    """
    Download / tokenize / process.

    Clean and (maybe) save to disk.

    Load inside Dataset.

    Apply transforms (rotate, tokenize, etc…).

    Wrap inside a DataLoader.
    """

    def __init__(self) -> None:
        super().__init__()
        self.text_padding_value = AutoTokenizer.from_pretrained(
            config.text_extractor_path, clean_up_tokenization_spaces=True
        ).pad_token_id
        # self.audio_padding_value = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000).padding_value
        self.audio_padding_value = 0

        # self.audio_extractor = SpeechTokenizer.load_from_checkpoint(
        #     "model_hub/speech-tokenizer/speechtokenizer_hubert_avg_config.json",
        #     "model_hub/speech-tokenizer/SpeechTokenizer.pt",
        # )

    def prepare_data(self):
        # download
        """
        在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        最最开始的时候，进行一些无论GPU有多少只要执行一次的操作，如写入磁盘的下载操作、分词操作(tokenize)等。
        这里是一劳永逸式准备数据的函数。
        由于只在单线程中调用，不要在这个函数中进行self.x=y似的赋值操作。
        但如果是自己用而不是给大众分发的话，这个函数可能并不需要调用，因为数据提前处理好就好了
        """
        pass

    def setup(self, stage: str) -> None:
        # Assign train/val datasets for use in dataloaders
        """
        实现数据集的定义，每张GPU都会执行该函数, stage 用于标记是用于什么阶段
        实例化数据集（Dataset），并进行相关操作，如：清点类数，划分train/val/test集合等。
        参数stage用于指示是处于训练周期(fit)还是测试周期(test)，其中，fit周期需要构建train和val两者的数据集。
        setup函数不需要返回值。初始化好的train/val/test set直接赋值给self即可。
        """
        if config.dataset == "mosi":
            Dataset = DatasetMosi
        elif config.dataset == "mosei":
            Dataset = DatasetMosei
        else:
            raise ValueError("Dataset must be mosi or mosei")

        if stage == "fit":
            self.train_data = Dataset("train")
            self.val_data = Dataset("valid")
            self.test_data = Dataset("test")
        elif stage == "test":
            self.test_data = Dataset("test")
        elif stage == "predict":
            self.test_data = Dataset("test")
            pass
        else:
            raise ValueError("Stage must be fit, test or predict")

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=config.batch_size_train,
            shuffle=True,
            num_workers=config.num_workers,
            persistent_workers=True if config.num_workers > 0 else False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_data,
                batch_size=config.batch_size_eval,
                shuffle=False,
                num_workers=config.num_workers,
                persistent_workers=True if config.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            ),
            DataLoader(
                self.test_data,
                batch_size=config.batch_size_eval,
                shuffle=False,
                num_workers=config.num_workers,
                persistent_workers=True if config.num_workers > 0 else False,
                pin_memory=True,
                collate_fn=self.collate_fn,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=config.batch_size_eval,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=True if config.num_workers > 0 else False,
            # pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=config.batch_size_eval,
            shuffle=False,
            num_workers=config.num_workers,
            persistent_workers=True if config.num_workers > 0 else False,
            # pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        batch = [i.values() for i in batch]
        text, text_input_ids, text_attention_mask, audio_path, audio_wave, sample_rate, audio_length, labels = zip(*batch)

        text_input_ids = torch.nn.utils.rnn.pad_sequence(text_input_ids, batch_first=True, padding_value=self.text_padding_value)
        text_attention_mask = torch.nn.utils.rnn.pad_sequence(text_attention_mask, batch_first=True, padding_value=0)

        audio_attention_mask = [torch.ones(i.size(0), dtype=torch.float) for i in audio_wave]
        audio_wave = torch.nn.utils.rnn.pad_sequence(audio_wave, batch_first=True, padding_value=self.audio_padding_value)
        audio_attention_mask = torch.nn.utils.rnn.pad_sequence(audio_attention_mask, batch_first=True, padding_value=0)
        audio_length = torch.tensor(audio_length)

        labels = torch.tensor(labels)
        return {
            "text": text,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask,
            "audio_path": audio_path,
            "audio_wave": audio_wave,
            "sample_rate": sample_rate,
            "audio_length": audio_length,
            "audio_attention_mask": audio_attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    from config import config

    dm = LightningData()  # audio, text, multimodal
    dm.prepare_data()
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        print(batch)
        break
