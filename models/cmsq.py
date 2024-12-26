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
from math import sqrt


class MoEExpert(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size * 4)
        self.linear2 = nn.Linear(input_size, input_size * 4)
        self.linear3 = nn.Linear(input_size * 4, input_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = F.relu(self.linear1(x))
        x1 = self.dropout(x1)
        x2 = F.sigmoid(self.linear2(x))
        x = x1 * x2
        x = self.linear3(x)
        return x


class CMSQLayer(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.self_att = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.text_expert = MoEExpert(hidden_size)
        self.audio_expert = MoEExpert(hidden_size)
        self.gate = nn.Linear(hidden_size, 2)

    def forward(self, feature, token_type_ids):
        # token_type_ids: 0 for text, 1 for audio

        residual = feature
        feature = self.norm1(feature)
        feature, attn_output_weights = self.self_att(feature, feature, feature)
        feature = self.dropout1(feature)
        feature += residual

        residual = feature
        feature = self.norm2(feature)
        text_mask = (token_type_ids == 0).unsqueeze(-1).float()
        audio_mask = (token_type_ids == 1).unsqueeze(-1).float()
        text_output = self.text_expert(feature)
        audio_output = self.audio_expert(feature)
        expert_output = text_output * text_mask + audio_output * audio_mask
        feature = expert_output
        feature = self.dropout2(expert_output)
        feature += residual
        return feature, attn_output_weights


class CMSQ(nn.Module):
    def __init__(self, num_layers, text_hidden_size, dropout):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(1, text_hidden_size))
        nn.init.xavier_normal_(self.mask_token)

        # self.cmsq_layers = nn.ModuleList(
        #     [
        #         nn.TransformerEncoderLayer(
        #             d_model=text_hidden_size,
        #             nhead=8,
        #             dim_feedforward=text_hidden_size * 4,
        #             dropout=dropout,
        #             activation="gelu",
        #             batch_first=True,
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        self.cmsq_layers = nn.ModuleList([CMSQLayer(text_hidden_size, dropout=dropout) for _ in range(num_layers)])

        self.norm1 = nn.LayerNorm(text_hidden_size)
        self.norm2 = nn.LayerNorm(text_hidden_size)
        self.norm3 = nn.LayerNorm(text_hidden_size)
        self.gate = nn.Linear(text_hidden_size * 2, 2)
        self.cls_token = nn.Parameter(torch.randn(1, text_hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(text_hidden_size, text_hidden_size * 4)
        self.fc2 = nn.Linear(text_hidden_size * 4, text_hidden_size)

    def forward(self, feature, token_type_ids):
        attn_output_weights_list = []
        if self.training:
            mask_prob = 0.2
            mask = torch.rand(feature.shape[0], feature.shape[1], device=feature.device)
            mask = mask < mask_prob
            mask = mask.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
            mask_token = self.mask_token.unsqueeze(0).expand(feature.shape[0], feature.shape[1], -1)
            feature = torch.where(mask, mask_token, feature)
        for layer in self.cmsq_layers:
            feature, attn_output_weights = layer(feature, token_type_ids)
            attn_output_weights_list.append(attn_output_weights)

        output = feature.mean(dim=1)
        return output


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
        ####################################################################################################
        self.audio_extractor_config = self.audio_extractor.config
        self.audio_feature_extractor = self.audio_extractor.feature_extractor
        self.audio_feature_projection = self.audio_extractor.feature_projection
        self.audio_encoder_pos_conv_embed = self.audio_extractor.encoder.pos_conv_embed
        self.audio_encoder_layer_norm = self.audio_extractor.encoder.layer_norm
        self.audio_encoder_dropout = self.audio_extractor.encoder.dropout
        self.audio_encoder_layers = self.audio_extractor.encoder.layers
        del self.audio_extractor
        ####################################################################################################
        if config.dataset == "mosi":
            text_extractor_weights = torch.load(
                "checkpoints/bart-s11-18-17:22/best.ckpt",
                map_location=self.device,
                weights_only=True,
            )["state_dict"]
            audio_extractor_weights = torch.load(
                "checkpoints/data2vec-s11-19-12:02/best.ckpt",
                map_location=self.device,
                weights_only=True,
            )["state_dict"]
            weights = text_extractor_weights | audio_extractor_weights
        elif config.dataset == "mosei":
            weights = torch.load(
                "checkpoints/pre-mosei-s12-04-12:40/best.ckpt",
                map_location=self.device,
                weights_only=True,
            )["state_dict"]
        else:
            raise ValueError("dataset error")
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name]
            elif (name_temp := name.replace("text_extractor.", "text_encoder_")) in weights:
                param.data = weights[name_temp]
            else:
                print("------ not in weights", name)
        ####################################################################################################

        self.audio_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(audio_hidden_size, text_hidden_size),
                    nn.GELU(),
                    nn.Linear(text_hidden_size, text_hidden_size),
                )
                for _ in range(text_num_layers)
            ]
        )
        self.text_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(text_hidden_size, text_hidden_size),
                    nn.GELU(),
                    nn.Linear(text_hidden_size, text_hidden_size),
                )
                for _ in range(audio_num_layers)
            ]
        )
        self.cmsq = CMSQ(2, text_hidden_size, config.dropout)
        self.cmsq_predict = nn.Sequential(
            nn.Linear(text_hidden_size, 1),
        )
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
        ###################################################### init feature_embedding
        output_audio = []
        for n in range(12):
            audio_hidden_states = self.audio_encoder_layers[n](audio_hidden_states, audio_attention_mask_new)[0]
            output_audio.append(self.audio_proj[n](audio_hidden_states))
        attentions_length = torch.floor((audio_length - 1) / 320).int()
        output_audio = [masked_mean(i, attentions_length) for i in output_audio]
        output_audio = torch.stack(output_audio, dim=1)

        output_text = [self.text_proj[i](j) for i, j in enumerate(output_text)]
        output_text = torch.stack(output_text, dim=1)
        output_text = output_text[:, :, 0, :]
        feature = torch.cat(
            [
                output_text,
                output_audio,
            ],
            dim=1,
        )
        token_type_ids = torch.cat(
            [
                torch.zeros(output_text.shape[0], output_text.shape[1], dtype=torch.long, device=feature.device),
                torch.ones(output_text.shape[0], output_audio.shape[1], dtype=torch.long, device=feature.device),
            ],
            dim=1,
        )
        output_cmsq = self.cmsq(feature, token_type_ids)
        output_cmsq = self.cmsq_predict(output_cmsq)

        return output_cmsq

    def compute_losses_metrics(self, batch):
        labels = batch["labels"].view(-1, 1)
        output_cmsq = self(batch)

        loss_cmsq = F.l1_loss(output_cmsq, labels, reduction="none")
        loss_cmsq = loss_cmsq.mean()

        # loss_text = F.l1_loss(output_text, labels, reduction="mean")
        # loss_audio = F.l1_loss(output_audio, labels, reduction="mean")
        # loss = loss_cmsq + loss_text + loss_audio

        losses = {"loss": loss_cmsq}
        metrics = self.metrics(output_cmsq, labels)

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
                {"params": params_text, "lr": 1e-5 / 3 * sqrt(self.config.num_GPU)},
                {"params": params_audio, "lr": 5e-5 / 3 * sqrt(self.config.num_GPU)},
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
        # scheduler = ThreePhaseLRScheduler(optimizer, total_steps=self.trainer.estimated_stepping_batches)
        lr_scheduler = {"scheduler": scheduler, "name": "learning rate", "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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
        opt1 = torch.optim.AdamW(
            [
                {"params": params_text, "lr": 1e-5, "weight_decay": self.config.weight_decay},
                {"params": params_audio, "lr": 5e-5, "weight_decay": self.config.weight_decay},
            ],
        )
        opt2 = torch.optim.AdamW(
            prrams_others,
            lr=1e-5,
            weight_decay=self.config.weight_decay,
        )
        sch1 = transformers.get_cosine_schedule_with_warmup(
            opt1,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        sch2 = transformers.get_cosine_schedule_with_warmup(
            opt2,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler1 = {"scheduler": sch1, "name": "lr_scheduler1", "interval": "step", "frequency": 1}
        lr_scheduler2 = {"scheduler": sch2, "name": "lr_scheduler2", "interval": "step", "frequency": 1}
        return (
            {"optimizer": opt1, "lr_scheduler": lr_scheduler1},
            {"optimizer": opt2, "lr_scheduler": lr_scheduler2},
        )

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/ZZZ": 0})

    def training_step(self, batch, batch_idx):
        # opt1, opt2 = self.optimizers()
        # for i, param_group in enumerate(list(enumerate(opt1.param_groups)) + list(enumerate(opt2.param_groups))):
        #     self.log(
        #         f"epoch/lr_{i}",
        #         param_group[1]["lr"],
        #         on_step=False,
        #         on_epoch=True,
        #         batch_size=self.config.batch_size_train,
        #         sync_dist=True if self.config.num_GPU > 1 else False,
        #     )
        # opt1.zero_grad()
        # opt2.zero_grad()

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

        # self.manual_backward(losses["loss"])
        # opt1.step()
        # opt2.step()

        # sch1, sch2 = self.lr_schedulers()
        # sch1.step()
        # sch2.step()

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
