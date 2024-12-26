import torch
import lightning as L
from torch import nn
import torch.nn.functional as F
from transformers import Data2VecAudioModel, AutoModel
from .utils import masked_mean
from utils.metrics import MetricsTop


class LightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.get_hparams())
        ####################################################################################################
        self.text_extractor = AutoModel.from_pretrained(
            config.text_extractor_path,
            output_hidden_states=True,
        ).encoder
        text_hidden_size = self.text_extractor.config.hidden_size
        self.audio_extractor = Data2VecAudioModel.from_pretrained(self.config.audio_extractor_path)
        ####################################################################################################
        audio_hidden_size = self.audio_extractor.config.hidden_size
        self.audio_extractor_config = self.audio_extractor.config
        self.audio_feature_extractor = self.audio_extractor.feature_extractor
        self.audio_feature_projection = self.audio_extractor.feature_projection
        self.audio_encoder_pos_conv_embed = self.audio_extractor.encoder.pos_conv_embed
        self.audio_encoder_layer_norm = self.audio_extractor.encoder.layer_norm
        self.audio_encoder_dropout = self.audio_extractor.encoder.dropout
        self.audio_encoder_layers = self.audio_extractor.encoder.layers
        ####################################################################################################
        self.predict = nn.Linear(text_hidden_size + audio_hidden_size, 1)
        self.metrics = MetricsTop("regression").getMetics(config.dataset)
        self.num_error = 0
        del self.audio_extractor

    def forward(self, batch):
        text = batch["text"]
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"]
        audio_path = batch["audio_path"]
        audio_wave = batch["audio_wave"]  # [bs, L]
        audio_attention_mask = batch["audio_attention_mask"]
        audio_length = batch["audio_length"]

        ######################################################
        output_text = self.text_extractor(input_ids=text_input_ids, attention_mask=text_attention_mask)
        output_text = output_text.hidden_states[1:]

        ###################################################### audio feature_extractor
        audio_extract_features = self.audio_feature_extractor(audio_wave)
        audio_extract_features = audio_extract_features.transpose(1, 2)
        audio_hidden_states, audio_extract_features = self.audio_feature_projection(audio_extract_features)
        audio_attention_mask_new = self.get_feature_vector_attention_mask(audio_extract_features.shape[1], audio_attention_mask)
        ###################################################### audio embedding
        audio_attention_mask_cmsq = audio_attention_mask_new
        expand_audio_attention_mask = audio_attention_mask_new.unsqueeze(-1).repeat(1, 1, audio_hidden_states.shape[2])
        audio_hidden_states[~expand_audio_attention_mask] = 0
        audio_attention_mask_new = 1.0 - audio_attention_mask_new[:, None, None, :].to(dtype=audio_hidden_states.dtype)
        audio_attention_mask_new = audio_attention_mask_new * torch.finfo(audio_hidden_states.dtype).min
        audio_attention_mask_new = audio_attention_mask_new.expand(
            audio_attention_mask_new.shape[0], 1, audio_attention_mask_new.shape[-1], audio_attention_mask_new.shape[-1]
        )
        audio_position_embeddings = self.audio_encoder_pos_conv_embed(audio_hidden_states)
        audio_hidden_states = audio_hidden_states + audio_position_embeddings
        audio_hidden_states = self.audio_encoder_layer_norm(audio_hidden_states)
        audio_hidden_states = self.audio_encoder_dropout(audio_hidden_states)
        ###################################################### init query_embedding
        for n in range(12):
            audio_hidden_states = self.audio_encoder_layers[n](audio_hidden_states, audio_attention_mask_new)[0]

        attentions_length = torch.floor((audio_length - 1) / 320).int()
        output_audio = masked_mean(audio_hidden_states, attentions_length)

        output_text = output_text[-1][:, 0, :]
        output = torch.cat([output_audio, output_text], dim=1)

        output_cmsq = self.predict(output)

        return output_cmsq

    def compute_losses_metrics(self, batch):
        labels = batch["labels"].view(-1, 1)
        output_cmsq = self(batch)

        loss_cmsq = F.l1_loss(output_cmsq, labels, reduction="none")
        loss_cmsq = loss_cmsq.mean()
        losses = {"loss": loss_cmsq}

        metrics = self.metrics(output_cmsq, labels)

        return losses, metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
        )
        return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/ZZZ": 0})

    def training_step(self, batch, batch_idx):
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"epoch/lr_{i}",
                param_group["lr"],
                on_step=False,
                on_epoch=True,
                batch_size=self.config.batch_size_train,
                sync_dist=True if self.config.num_GPU > 1 else False,
            )
        losses, metrics = self.compute_losses_metrics(batch)
        self.log_dict(
            {"train/train_" + k: v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.config.batch_size_train,
            sync_dist=True if self.config.num_GPU > 1 else False,
        )
        return losses

    def on_train_epoch_end(self):
        pass

    def on_validation_start(self):
        pass

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        losses, metrics = self.compute_losses_metrics(batch)
        if dataloader_idx == 0:
            self.log_dict(
                {f"train/{k}_val": v for k, v in losses.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=self.config.batch_size_eval,
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )
            self.log_dict(
                {f"val/{k}_val": v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=self.config.batch_size_eval,
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )
        else:
            self.log_dict(
                {f"train/{k}_test": v for k, v in losses.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=self.config.batch_size_eval,
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )
            self.log_dict(
                {f"val/{k}_test": v for k, v in metrics.items()},
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=self.config.batch_size_eval,
                sync_dist=True if self.config.num_GPU > 1 else False,
                add_dataloader_idx=False,
            )

    def on_validation_epoch_end(self):
        pass

    def on_test_start(self):
        pass

    def test_step(self, batch):
        losses, metrics = self.compute_losses_metrics(batch)
        self.log_dict(
            {"test/" + k: v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=self.config.batch_size_eval,
            sync_dist=True if self.config.num_GPU > 1 else False,
        )

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch):
        text = batch["text"]
        audio_path = batch["audio_path"]
        labels = batch["labels"]
        output_reg, output_contrast = self(batch)
        output_reg = output_reg.squeeze()
        is_nan = torch.isnan(output_reg)
        is_diff_sign = output_reg * labels <= 0
        is_large_distance = torch.abs(output_reg - labels) > 0.5
        is_error = is_nan | is_diff_sign | is_large_distance
        for i, error in enumerate(is_error):
            if error:
                self.num_error += 1
                print(
                    f'{self.num_error}: {int(batch["audio_length"][i]/320)} ===> {text[i]} ===> {audio_path[i]} ===> pred:{output_reg[i].item()} label:{labels[i].item()}\n'
                )

    def backward(self, loss):
        loss.backward()

    def get_feat_extract_output_lengths(self, input_lengths, add_adapter=None):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = None

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.audio_extractor_config.conv_kernel, self.audio_extractor_config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.audio_extractor_config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.audio_extractor_config.adapter_stride)

        return input_lengths

    def get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        # non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
        non_padded_lengths = attention_mask.sum(-1)

        output_lengths = self.get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros((batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device)
        # attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        row_indices = torch.arange(feature_vector_length, device=self.device).expand(batch_size, feature_vector_length)
        attention_mask = row_indices < output_lengths.unsqueeze(1)

        return attention_mask
