import torch
import lightning as L
import transformers
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, RobertaModel, HubertModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, BartModel
from transformers import BartForConditionalGeneration, BartForCausalLM, Data2VecAudioModel, AutoTokenizer, Data2VecAudioModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from utils.metrics import MetricsTop, compute_metrics, calculate_metrics_for_regression
from .utils import RMSNorm
from .utils import masked_mean, ThreePhaseLRScheduler
import random
import numpy as np


class LightningModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # self.automatic_optimization = False
        self.config = config
        self.save_hyperparameters(config.get_hparams())
        ####################################################################################################
        pretrain_dropout = 0.05
        self.text_extractor = AutoModel.from_pretrained(
            config.text_extractor_path,
            output_hidden_states=True,
            # add_pooling_layer=False,
            attention_dropout=pretrain_dropout,
            dropout=pretrain_dropout,
            activation_dropout=pretrain_dropout,
        ).encoder
        # self.text_extractor.train()
        text_hidden_size = self.text_extractor.config.hidden_size
        text_num_layers = self.text_extractor.config.num_hidden_layers
        self.text_encoder_embed_tokens = self.text_extractor.embed_tokens
        self.text_encoder_embed_positions = self.text_extractor.embed_positions
        self.text_encoder_layernorm_embedding = self.text_extractor.layernorm_embedding
        self.get_extended_attention_mask = self.text_extractor.get_extended_attention_mask
        self.text_encoder_layers = self.text_extractor.layers
        del self.text_extractor
        ####################################################################################################
        self.audio_extractor = Data2VecAudioModel.from_pretrained(
            config.audio_extractor_path,
            activation_dropout=pretrain_dropout,  # default 0.1
            attention_dropout=pretrain_dropout,  # default 0.1
            final_dropout=pretrain_dropout,  # default 0.1
            hidden_dropout=pretrain_dropout,  # default 0.1
            # feat_proj_dropout=pretrain_dropout,  # default 0.0
            # mask_time_prob=0,
            # mask_feature_prob=0,
            # use_weighted_layer_sum = True,  # default False
        )
        # self.audio_extractor.train()
        audio_hidden_size = self.audio_extractor.config.hidden_size
        audio_num_layers = self.audio_extractor.config.num_hidden_layers
        self.audio_extractor_config = self.audio_extractor.config
        self.audio_feature_extractor = self.audio_extractor.feature_extractor
        self.audio_feature_projection = self.audio_extractor.feature_projection
        self.audio_encoder_pos_conv_embed = self.audio_extractor.encoder.pos_conv_embed
        self.audio_encoder_layer_norm = self.audio_extractor.encoder.layer_norm
        self.audio_encoder_dropout = self.audio_extractor.encoder.dropout
        self.audio_encoder_layers = self.audio_extractor.encoder.layers
        del self.audio_extractor
        ####################################################################################################
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.text_predict = nn.Linear(text_hidden_size, 1)
        self.audio_predict = nn.Linear(text_hidden_size, 1)

        self.metrics = MetricsTop("regression").getMetics(config.dataset)
        self.num_error = 0

    def forward(self, batch):
        text = batch["text"]
        text_input_ids = batch["text_input_ids"]
        text_attention_mask = batch["text_attention_mask"]
        audio_path = batch["audio_path"]
        audio_wave = batch["audio_wave"]  # [bs, L]
        audio_attention_mask = batch["audio_attention_mask"]
        audio_length = batch["audio_length"]
        ######################################################  text embedding
        # output_text = self.text_extractor(input_ids=text_input_ids, attention_mask=text_attention_mask)
        # output_text = output_text.hidden_states[1:]
        inputs_embeds = self.text_encoder_embed_tokens(text_input_ids)
        embed_pos = self.text_encoder_embed_positions(text_input_ids)
        encoder_hidden_states = inputs_embeds + embed_pos
        text_encoder_hidden_states = self.text_encoder_layernorm_embedding(encoder_hidden_states)
        # text_encoder_hidden_states = nn.functional.dropout(text_encoder_hidden_states, p=0.1, training=self.training)
        extended_text_attention_mask = self.get_extended_attention_mask(text_attention_mask, text_input_ids.size())
        # fill mask with torch.finfo(dtype).min
        extended_text_attention_mask = _prepare_4d_attention_mask(text_attention_mask, inputs_embeds.dtype)
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
        ###################################################### init feature_embedding
        output_text, output_audio = [], []
        for n in range(12):
            text_encoder_hidden_states = self.text_encoder_layers[n](text_encoder_hidden_states, extended_text_attention_mask, None)[0]
            audio_hidden_states = self.audio_encoder_layers[n](audio_hidden_states, audio_attention_mask_new)[0]
            output_text.append(text_encoder_hidden_states)
            output_audio.append(audio_hidden_states)

        output_text = torch.stack([i[:, 0, :] for i in output_text], dim=0)
        output_text = output_text.mean(dim=0)
        output_text = self.text_proj(output_text)

        attentions_length = torch.floor((audio_length - 1) / 320).int()
        output_audio = torch.stack(output_audio, dim=0)
        output_audio = output_audio.mean(dim=0)
        output_audio = masked_mean(output_audio, attentions_length)
        output_audio = self.audio_proj(output_audio)

        output_audio = output_audio / output_audio.norm(dim=1, keepdim=True)
        output_text = output_text / output_text.norm(dim=1, keepdim=True)

        return output_text, output_audio

    def compute_losses_metrics(self, batch):
        output_text, output_audio = self(batch)
        labels = batch["labels"]

        loss_text_reg = F.l1_loss(self.text_predict(output_text).squeeze(), labels)
        loss_audio_reg = F.l1_loss(self.audio_predict(output_audio).squeeze(), labels)

        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * output_audio @ output_text.t()
        logits_per_text = logits_per_audio.t()
        labels_contrastive = torch.arange(output_text.shape[0], device=self.device).long()
        loss_text_con = F.cross_entropy(logits_per_text, labels_contrastive)
        loss_audio_con = F.cross_entropy(logits_per_audio, labels_contrastive)

        loss = (loss_text_con + loss_audio_con) / 2 + (loss_text_reg + loss_audio_reg) / 2
        losses = {
            "loss": loss,
            "loss_text_reg": loss_text_reg,
            "loss_audio_reg": loss_audio_reg,
            "loss_text_con": loss_text_con,
            "loss_audio_con": loss_audio_con,
        }

        metrics = {}

        return losses, metrics

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.config.learning_rate,
        #     weight_decay=self.config.weight_decay,
        #     betas=(0.9, 0.999),
        # )
        # return optimizer
        params_text = []
        params_audio = []
        prrams_others = []
        for name, param in self.named_parameters():
            if "text" in name:
                params_text.append(param)
            elif "audio" in name:
                params_audio.append(param)
            else:
                prrams_others.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": params_text, "lr": 1e-5},
                {"params": params_audio, "lr": 5e-5},
                {"params": prrams_others, "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
            ],
            betas=(0.9, 0.999),
        )
        # return optimizer
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler = {"scheduler": scheduler, "name": "learning rate", "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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
