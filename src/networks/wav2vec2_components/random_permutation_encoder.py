########################################################################################
#
# Implement random permutations in the encoder for potentially beneficial
# regularization.
#
# Author(s): Anonymous
########################################################################################

from typing import Optional

import torch as t
import torch.nn as nn

import numpy as np

from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2 import configuration_wav2vec2

from src.networks.wav2vec2_components.base_components import (
    Wav2vec2Encoder,
    Wav2Vec2EncoderStableLayerNorm,
)

########################################################################################
# implementation of encoder with random permutation of time sequence


class Wav2vec2RandomPermutationEncoder(nn.Module):
    shuffle_input_sequence_options = ["before", "after", None]

    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
        shuffle_input_sequence: Optional[str] = None,
    ):
        super().__init__()

        base_encoder = Wav2vec2Encoder(
            cfg,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            pretrained_weights=pretrained_weights,
        )

        self.config = base_encoder.encoder.config
        self.pos_conv_embed = base_encoder.encoder.pos_conv_embed
        self.layer_norm = base_encoder.encoder.layer_norm
        self.dropout = base_encoder.encoder.dropout
        self.layers = base_encoder.encoder.layers
        self.gradient_checkpointing = base_encoder.encoder.gradient_checkpointing

        del base_encoder

        if shuffle_input_sequence not in self.shuffle_input_sequence_options:
            raise ValueError(
                f"{shuffle_input_sequence} should be one of"
                f" {self.shuffle_input_sequence_options}"
            )

        self.shuffle_input_sequence = shuffle_input_sequence

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

            # extend attention_mask
            attention_mask = (
                1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            ) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        if self.training and self.shuffle_input_sequence == "before":
            hidden_states = hidden_states[:, t.randperm(hidden_states.shape[1]), :]

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.training and self.shuffle_input_sequence == "after":
            hidden_states = hidden_states[:, t.randperm(hidden_states.shape[1]), :]

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.layerdrop)
                else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = t.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class Wav2vec2RandomPermutationEncoderStableLayerNorm(nn.Module):
    shuffle_input_sequence_options = ["before", "after", None]

    def __init__(
        self,
        cfg: configuration_wav2vec2.Wav2Vec2Config,
        enable_gradient_checkpointing: bool,
        pretrained_weights: Optional[str] = None,
        shuffle_input_sequence: Optional[str] = None,
    ):
        super().__init__()

        base_encoder = Wav2Vec2EncoderStableLayerNorm(
            cfg,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            pretrained_weights=pretrained_weights,
        )

        self.config = base_encoder.encoder.config
        self.pos_conv_embed = base_encoder.encoder.pos_conv_embed
        self.layer_norm = base_encoder.encoder.layer_norm
        self.dropout = base_encoder.encoder.dropout
        self.layers = base_encoder.encoder.layers
        self.gradient_checkpointing = base_encoder.encoder.gradient_checkpointing

        del base_encoder

        if shuffle_input_sequence not in self.shuffle_input_sequence_options:
            raise ValueError(
                f"{shuffle_input_sequence} should be one of"
                f" {self.shuffle_input_sequence_options}"
            )

        self.shuffle_input_sequence = shuffle_input_sequence

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            hidden_states[~attention_mask] = 0

            # extend attention_mask
            attention_mask = (
                1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            ) * -10000.0
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        if self.training and self.shuffle_input_sequence == "before":
            hidden_states = hidden_states[:, t.randperm(hidden_states.shape[1]), :]

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        if self.training and self.shuffle_input_sequence == "after":
            hidden_states = hidden_states[:, t.randperm(hidden_states.shape[1]), :]

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = (
                True
                if self.training and (dropout_probability < self.config.layerdrop)
                else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
