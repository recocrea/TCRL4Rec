# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import Encoder, LayerNorm


class ProjectedContentEmbedding(nn.Module):
    def __init__(self, item_size, cache_path, input_dim=2048, hidden_dim=512,
                 output_dim=128, initializer_range=0.02):
        super(ProjectedContentEmbedding, self).__init__()
        self.register_buffer(
            "raw_content_embeddings",
            self._load_content_embeddings(cache_path, item_size, input_dim)
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_projection(initializer_range)

    def _init_projection(self, initializer_range):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _load_torch_object(self, cache_path):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Content cache not found: {cache_path}")
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location="cpu")

    def _normalize_loaded_cache_keys(self, cache_obj):
        normalized = {}
        for raw_key, embedding in cache_obj.items():
            if isinstance(raw_key, int):
                normalized[raw_key] = embedding
            elif isinstance(raw_key, str):
                if raw_key.isdigit():
                    normalized[int(raw_key)] = embedding
                elif ":" in raw_key:
                    _, item_id = raw_key.split(":", 1)
                    if item_id.isdigit():
                        normalized[int(item_id)] = embedding
        return normalized

    def _load_content_embeddings(self, cache_path, item_size, input_dim):
        cache_obj = self._load_torch_object(cache_path)

        if isinstance(cache_obj, torch.Tensor):
            features = cache_obj.float()
            if features.dim() != 2:
                raise ValueError(
                    f"Expected content cache tensor to be 2D, got shape {tuple(features.shape)}."
                )
            if features.size(-1) != input_dim:
                raise ValueError(
                    f"Expected content dimension {input_dim}, got {features.size(-1)}."
                )
            if features.size(0) > item_size - 1:
                raise ValueError(
                    f"Content rows {features.size(0)} exceed available item slots {item_size - 1}."
                )
            full_features = torch.zeros(item_size, input_dim, dtype=features.dtype)
            full_features[1:1 + features.size(0)] = features
            return full_features

        if not isinstance(cache_obj, dict):
            raise ValueError(
                f"Unsupported cache type {type(cache_obj)}. Expected tensor or dict payload."
            )

        embedding_cache = cache_obj.get("embedding_cache", cache_obj)
        if not isinstance(embedding_cache, dict):
            raise ValueError("Cache payload must contain a dict field `embedding_cache`.")

        normalized_cache = self._normalize_loaded_cache_keys(embedding_cache)
        if not normalized_cache:
            raise ValueError("No valid item embeddings found in cache payload.")

        example = next(iter(normalized_cache.values()))
        if not isinstance(example, torch.Tensor):
            raise ValueError("Cached item embedding must be a torch.Tensor.")
        if example.dim() != 1 or example.numel() != input_dim:
            raise ValueError(
                f"Expected each cached item embedding to have shape ({input_dim},), "
                f"got {tuple(example.shape)}."
            )

        full_features = torch.zeros(item_size, input_dim, dtype=example.dtype)
        for item_id, emb in normalized_cache.items():
            if 0 < item_id < item_size:
                full_features[item_id] = emb.float()
        return full_features

    @property
    def weight(self):
        projected = self.projection(self.raw_content_embeddings)
        projected = projected.clone()
        projected[0].zero_()
        return projected

    def forward(self, item_ids):
        raw_embeddings = F.embedding(item_ids, self.raw_content_embeddings, padding_idx=0)
        projected = self.projection(raw_embeddings)
        return projected * item_ids.ne(0).unsqueeze(-1).to(projected.dtype)

    def project_items(self, item_ids):
        return self.forward(item_ids)


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)

    def build_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape, device=input_ids.device), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1).long()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def add_position_embedding(self, sequence):
        return self.add_position_embedding_with_modules(
            sequence,
            self.item_embeddings,
            self.position_embeddings,
            self.LayerNorm,
            self.dropout,
        )

    def add_position_embedding_with_modules(self, sequence, item_embeddings, position_embeddings,
                                            layer_norm, dropout):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        sequence_emb = item_embeddings(sequence) + position_embeddings(position_ids)
        sequence_emb = layer_norm(sequence_emb)
        sequence_emb = dropout(sequence_emb)
        return sequence_emb

    def encode_with_modules(self, input_ids, item_embeddings, position_embeddings,
                            layer_norm, dropout, item_encoder):
        extended_attention_mask = self.build_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding_with_modules(
            input_ids, item_embeddings, position_embeddings, layer_norm, dropout
        )
        item_encoded_layers = item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True
        )
        return item_encoded_layers[-1]

    def transformer_encoder(self, input_ids):
        return self.encode_with_modules(
            input_ids,
            self.item_embeddings,
            self.position_embeddings,
            self.LayerNorm,
            self.dropout,
            self.item_encoder,
        )

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DualBranchSASRecModel(SASRecModel):
    def __init__(self, args):
        super(DualBranchSASRecModel, self).__init__(args)
        # The dual-branch model reuses SASRec helper methods but not its base modules.
        del self.position_embeddings
        del self.LayerNorm
        del self.dropout
        del self.item_encoder

        cache_path = getattr(
            args,
            "cache_path",
            os.path.join(args.output_dir, f"{args.data_name}_raw_cache.pt"),
        )

        self.is_dual_branch = True
        self.id_item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.id_position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.id_item_encoder = Encoder(args)
        self.id_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.id_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.content_item_embeddings = ProjectedContentEmbedding(
            item_size=args.item_size,
            cache_path=cache_path,
            input_dim=2048,
            hidden_dim=512,
            output_dim=args.hidden_size,
            initializer_range=args.initializer_range,
        )
        self.content_position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.content_item_encoder = Encoder(args)
        self.content_layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.content_dropout = nn.Dropout(args.hidden_dropout_prob)

        # Keep compatibility with existing trainer attribute access.
        self.item_embeddings = self.id_item_embeddings
        self.apply(self.init_weights)
        self.args = args

    def encode_branches(self, input_ids):
        id_sequence_output = self.encode_with_modules(
            input_ids,
            self.id_item_embeddings,
            self.id_position_embeddings,
            self.id_layer_norm,
            self.id_dropout,
            self.id_item_encoder,
        )
        content_sequence_output = self.encode_with_modules(
            input_ids,
            self.content_item_embeddings,
            self.content_position_embeddings,
            self.content_layer_norm,
            self.content_dropout,
            self.content_item_encoder,
        )
        return id_sequence_output, content_sequence_output

    def encode_content_branch(self, input_ids):
        return self.encode_with_modules(
            input_ids,
            self.content_item_embeddings,
            self.content_position_embeddings,
            self.content_layer_norm,
            self.content_dropout,
            self.content_item_encoder,
        )

    def transformer_encoder(self, input_ids):
        id_sequence_output, content_sequence_output = self.encode_branches(input_ids)
        return 0.5 * (id_sequence_output + content_sequence_output)

    def get_item_embeddings(self, branch):
        if branch == 'id':
            return self.id_item_embeddings
        if branch == 'content':
            return self.content_item_embeddings
        raise ValueError(f"Unknown branch: {branch}")

    def predict_branch_full(self, seq_out, branch):
        item_embeddings = self.get_item_embeddings(branch).weight
        return torch.matmul(seq_out, item_embeddings.transpose(0, 1))

    def alignment_loss(self, id_sequence_output, content_sequence_output, input_ids):
        valid_mask = (input_ids > 0).unsqueeze(-1).float()
        valid_count = valid_mask.sum().clamp_min(1.0)
        return ((id_sequence_output - content_sequence_output) ** 2 * valid_mask).sum() / valid_count
