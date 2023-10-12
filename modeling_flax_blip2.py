from functools import partial
from typing import Any, Optional, Tuple
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxBaseModelOutputWithPooling
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from transformers.models.t5.modeling_flax_t5 import FlaxT5ForConditionalGenerationModule


class FlaxBlip2Attention(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embed_dim = self.config.hidden_size
        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = self.config.attention_dropout

        dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.qkv = dense(self.embed_dim * 3)
        self.projection = dense(self.embed_dim)


    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
       # proj q, k, v
        fused_qkv = self.qkv(hidden_states)
        fused_qkv = self._split_heads(fused_qkv)
        query, key, value = jnp.split(fused_qkv, 3, axis=-1)

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.projection(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxBlip2MLP(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.activation_fn = ACT2FN[self.config.hidden_act]
        self.fc1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.01),
        )
        self.fc2 = nn.Dense(self.config.hidden_size, dtype=self.dtype, kernel_init=jax.nn.initializers.normal(0.01))

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxBlip2VisionEmbeddings(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        image_size = self.config.image_size
        patch_size = self.config.patch_size

        self.class_embedding = self.param("class_embedding", jax.nn.initializers.normal(stddev=0.02), (embed_dim,))

        self.patch_embedding = nn.Conv(
            embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(),
        )

        self.num_patches = (image_size // patch_size) ** 2
        num_positions = self.num_patches + 1
        self.position_embedding = nn.Embed(num_positions, embed_dim, embedding_init=jax.nn.initializers.normal())
        self.position_ids = jnp.expand_dims(jnp.arange(0, num_positions, dtype="i4"), axis=0)

    def __call__(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        batch_size, height, width, channels = patch_embeds.shape
        patch_embeds = jnp.reshape(patch_embeds, (batch_size, height * width, channels))

        class_embeds = jnp.expand_dims(self.class_embedding, axis=(0, 1))
        class_embeds = jnp.tile(class_embeds, (batch_size, 1, 1))
        embeddings = jnp.concatenate([class_embeds, patch_embeds], axis=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class FlaxBlip2EncoderLayer(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self_attn = FlaxBlip2Attention(self.config, dtype=self.dtype)
        self.layer_norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.mlp = FlaxBlip2MLP(self.config, dtype=self.dtype)
        self.layer_norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        hidden_states = attn_outputs[0]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += attn_outputs[1:]

        return outputs


class FlaxBlip2LayerCollection(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBlip2EncoderLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states, attention_mask, deterministic=deterministic, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class FlaxBlip2Encoder(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = FlaxBlip2LayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layers(
            hidden_states=inputs_embeds,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

class FlaxBlip2VisionModule(nn.Module):
    config: Blip2VisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxBlip2VisionEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBlip2Encoder(self.config, dtype=self.dtype)
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class FlaxBlip2QFormerMultiHeadAttention(nn.Module):
    config: Blip2QFormerConfig
    is_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        if self.config.hidden_size % self.config.num_attention_heads != 0 and not hasattr(self.config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (self.config.hidden_size, self.config.num_attention_heads)
            )
        
        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(self.config.hidden_size / self.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        dense = partial(
            nn.Dense,
            self.all_head_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.query, self.key, self.value = dense(), dense(), dense()
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def _split_heads(self, hidden_states):
        # TODO
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_attention_heads, self.attention_head_size))

    def _merge_heads(self, hidden_states):
        # TODO
        return hidden_states.reshape(hidden_states.shape[:2] + (self.all_head_size,))

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = encoder_hidden_states is not None

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = self._split_heads(query_states)
        key_states = self._split_heads(key_states)
        value_states = self._split_heads(value_states)

        if attention_mask is not None:
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            # Convert the boolean attention mask to an attention bias.
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.dropout,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = self._merge_heads(attn_output)
        return attn_output, attn_weights

class FlaxBlip2QFormerSelfOutput(nn.Module):
    config: Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states



class FlaxBlip2QFormerAttention(nn.Module):
    config: Blip2QFormerConfig
    is_cross_attention: bool = False
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self) -> None:
        self.attention = FlaxBlip2QFormerMultiHeadAttention(self.config, self.is_cross_attention, self.dtype)
        self.output = FlaxBlip2QFormerSelfOutput(self.config, self.dtype)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            deterministic=deterministic
        )
        attention_output = self.output(self_outputs[0], hidden_states, deterministic=deterministic)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



class FlaxBlip2QFormerIntermediate(nn.Module):
    config: Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std),
        )
        self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
    
    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FlaxBlip2QFormerOutput(nn.Module):
    config: Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBlip2QFormerLayer(nn.Module):
    config: Blip2QFormerConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxBlip2QFormerAttention(self.config, dtype=self.dtype)

        if self.layer_idx % self.config.cross_attention_frequency == 0:
            self.crossattention = FlaxBlip2QFormerAttention(self.config, is_cross_attention=True, dtype=self.dtype)
            self.has_crossattention = True
        
        self.intermediate_query = FlaxBlip2QFormerIntermediate(self.config, dtype=self.dtype)
        self.output_query = FlaxBlip2QFormerOutput(self.config, dtype=self.dtype)
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        query_length: int = 0,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            deterministic=deterministic
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]

        query_attention_output = attention_output[:, :query_length, :]
        if self.has_crossattention:
            cross_attention_outputs = self.crossattention(
                query_attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                deterministic=deterministic
            )
            query_attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]

        intermediate_output = self.intermediate_query(query_attention_output)
        layer_output = self.output_query(intermediate_output, query_attention_output, deterministic=deterministic)

        outputs = (layer_output,) + outputs
        return outputs


class FlaxBlip2QFormerLayerCollection(nn.Module):
    config = Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBlip2QFormerLayer(self.config, i, dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
    
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        query_length: int = 0,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                query_length,
                deterministic=deterministic
            )
            hidden_states = layer_outputs[0]

        outputs = (hidden_states,)
        return outputs


class FlaxBlip2QFormerEncoder(nn.Module):
    config: Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer = FlaxBlip2QFormerLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        query_length: int = 0,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        return self.layers(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            query_length,
            deterministic=deterministic
        )


class FlaxBlip2QFormerModule(nn.Module):
    config: Blip2QFormerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.encoder = FlaxBlip2QFormerEncoder(self.config, dtype=self.dtype)
    
    def __call__(
        self,
        query_embeds: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        query_length: int = 0,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray]:
        hidden_states = self.layernorm(query_embeds)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            query_length,
            deterministic=deterministic
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        return (sequence_output, pooled_output) + encoder_outputs[1:]


class FlaxBlip2ForConditionalGenerationModule(nn.Module):
    config: Blip2Config
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        self.vision_model = FlaxBlip2VisionModule(self.config.vision_config, dtype=self.dtype)
        self.qformer = FlaxBlip2QFormerModule(self.config.qformer_config, dtype=self.dtype)
        self.query_tokens = self.param("query_tokens", jax.nn.initializers.normal(stddev=0.02), (1, self.config.num_query_tokens, self.config.qformer_config.hidden_size))
        self.language_projection = nn.Dense(
            self.config.text_config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.text_config.initializer_range),
        )
        self.language_model = FlaxT5ForConditionalGenerationModule(self.config.text_config, dtype=self.dtype)
    
    def _get_encoder_module(self):
        return self.language_model._get_encoder_module()

    def _get_decoder_module(self):
        return self.language_model._get_decoder_module()
    
    def __call__(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        decoder_past_key_values=None,
        return_dict=None,
        deterministic: bool = True,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(pixel_values, deterministic=deterministic)
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = jnp.ones(image_embeds.shape[:-1], dtype="i4")
        query_tokens = jnp.tile(self.query_tokens, (image_embeds.shape[0], 1, 1))
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            deterministic=deterministic
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = jnp.ones(language_model_inputs.shape[:-1], dtype="i4")
        # TODO