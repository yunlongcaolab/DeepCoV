import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from model.dms_module import DMSLayer

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm

class NormalizedResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout.
    
    Standard transformer residual block with pre-normalization
    and dropout for stable training of deep networks.
    """
    def __init__(
        self,
        layer: nn.Module,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = ESM1bLayerNorm(self.embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x

class FeedForwardNetwork(nn.Module):
    """Feed-forward network for transformer layers.
    
    Standard transformer FFN with activation function and dropout
    for non-linear feature transformation.
    """
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = max_tokens_per_msa
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x

class SpatialDepthWiseConvolution(nn.Module):
    """Depth-wise convolution for spatial feature processing.
    
    Applies depth-wise convolution across spatial dimensions
    for efficient local feature extraction.
    """
    def __init__(self, head_dim: int, kernel_size: int = 3, num_targets: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=head_dim, out_channels=head_dim, kernel_size=(kernel_size,), padding='same', groups=head_dim)
        self.num_targets = num_targets
    
    def forward(self, x: torch.Tensor):
        # Need to separate the targets from protein embeddings (convolutions only apply to protein embeddings)
        x , y = x[:,:-self.num_targets], x[:,-self.num_targets:]
        # Apply conv. Input x is of dim (num_rows, seq_len, batch_size, self.num_heads, self.head_dim)
        num_rows, seq_len, batch_size, num_heads, head_dim = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(num_rows * batch_size * num_heads, head_dim, seq_len)
        x = self.conv(x)
        x = x.view(num_rows, batch_size, num_heads, head_dim, seq_len)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Concatenate back with targets
        x = torch.cat([x,y], dim=1)
        return x

class AxialTransformerLayer(nn.Module):
    """Axial MSA Transformer block.
    
    Implements axial attention for efficient processing of
    multiple sequence alignments with reduced complexity.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2**14,
        deactivate_col_attention: bool = False,
        tranception_attention: bool = False,
        num_targets: int = 1,
        num_seqs: int = 33,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        self.deactivate_col_attention = deactivate_col_attention

        row_self_attention = RowSelfAttention(
            embedding_dim,
            num_attention_heads,
            dropout=dropout,
            max_tokens_per_msa=max_tokens_per_msa,
            tranception_attention=tranception_attention,
            num_targets=num_targets,
        )

        if not self.deactivate_col_attention:
            column_self_attention = ColumnSelfAttention(
                embedding_dim,
                num_attention_heads,
                dropout=dropout,
                max_tokens_per_msa=max_tokens_per_msa,
                num_seqs = num_seqs
            )
        else:
            print("No column attention in the underlying axial transformer module")

        feed_forward_layer = FeedForwardNetwork(
            embedding_dim,
            ffn_embedding_dim,
            activation_dropout=activation_dropout,
            max_tokens_per_msa=max_tokens_per_msa,
            
        )

        self.row_self_attention = self.build_residual(row_self_attention)

        if not self.deactivate_col_attention: 
            self.column_self_attention = self.build_residual(column_self_attention)
            
        self.feed_forward_layer = self.build_residual(feed_forward_layer)

    def build_residual(self, layer: nn.Module):
        return NormalizedResidualBlock(
            layer,
            self.embedding_dim,
            self.dropout_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_head_weights: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        x, row_attn = self.row_self_attention(
            x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )

        if not self.deactivate_col_attention:
            x, column_attn = self.column_self_attention(
                x,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
        else:
            column_attn = None
            
        x = self.feed_forward_layer(x)
        if need_head_weights:
            return x, column_attn, row_attn
        else:
            return x

class RowSelfAttention(nn.Module):
    """Row-wise self-attention for 2D inputs.
    
    Computes attention across row dimensions for efficient
    processing of sequence alignments.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
        tranception_attention: bool = False,
        num_targets: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "hnij"

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)
        
        self.tranception_attention = tranception_attention
        self.num_targets = num_targets
        if self.tranception_attention:
            assert self.num_heads%4==0, "Invalid number of heads. Tranception requires the number of heads to be a multiple of 4."
            self.num_heads_per_kernel_size = self.num_heads // 4
            self.query_depthwiseconv = nn.ModuleDict()
            self.key_depthwiseconv = nn.ModuleDict()
            self.value_depthwiseconv = nn.ModuleDict()
            for kernel_idx, kernel in enumerate([3,5,7]):
                self.query_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(self.head_dim,kernel,self.num_targets)
                self.key_depthwiseconv[str(kernel_idx)]   = SpatialDepthWiseConvolution(self.head_dim,kernel,self.num_targets)
                self.value_depthwiseconv[str(kernel_idx)] = SpatialDepthWiseConvolution(self.head_dim,kernel,self.num_targets)

    def align_scaling(self, q):
        num_rows = q.size(0)
        return self.scaling / math.sqrt(num_rows)

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attns = 0
        scaling = self.align_scaling(x)
        for start in range(0, num_rows, max_rows):
            attn_weights = self.compute_attention_weights(
                x[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, start : start + max_rows]
                if self_attn_padding_mask is not None
                else None,
            )
            attns += attn_weights
        attn_probs = attns.softmax(-1)
        attn_probs = self.dropout_module(attn_probs)

        outputs = []
        for start in range(0, num_rows, max_rows):
            output = self.compute_attention_update(x[start : start + max_rows], attn_probs)
            outputs.append(output)

        output = torch.cat(outputs, 0)
        return output, attn_probs

    def compute_attention_weights(
        self,
        x,
        scaling: float,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()#(33,1024,2,embed_dim)
        q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        q *= scaling
        if self_attn_padding_mask is not None:
            # Zero out any padded aligned positions - this is important since
            # we take a sum across the alignment axis.
            q *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(q)
        
        if self.tranception_attention:
            # We do not do anything on the first self.num_heads_per_kernel_size heads (kernel =1)
            query_list=[q[:,:,:,:self.num_heads_per_kernel_size,:]]
            key_list=[k[:,:,:,:self.num_heads_per_kernel_size,:]]
            for kernel_idx in range(3):
                query_list.append(self.query_depthwiseconv[str(kernel_idx)](q[:,:,:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:]))
                key_list.append(self.key_depthwiseconv[str(kernel_idx)](k[:,:,:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:]))
            q=torch.cat(query_list, dim=1)
            k=torch.cat(key_list, dim=1)
            
        attn_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", q, k)

        if self_attn_mask is not None:
            raise NotImplementedError
            # Mask Size: [B x R x C], Weights Size: [H x B x C x C]

        if self_attn_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )

        return attn_weights

    def compute_attention_update(
        self,
        x,
        attn_probs,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        
        if self.tranception_attention:
            value_list=[v[:,:,:,:self.num_heads_per_kernel_size,:]]
            for kernel_idx in range(3):
                value_list.append(self.value_depthwiseconv[str(kernel_idx)](v[:,:,:,(kernel_idx+1)*self.num_heads_per_kernel_size:(kernel_idx+2)*self.num_heads_per_kernel_size,:]))
            v=torch.cat(value_list, dim=1)

        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attn_probs, v)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        output = self.out_proj(context)
        return output

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()#(33,1024,2,embed_dim)
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(x, self_attn_mask, self_attn_padding_mask)
        else:
            scaling = self.align_scaling(x)
            attn_weights = self.compute_attention_weights(
                x, scaling, self_attn_mask, self_attn_padding_mask
            )
            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            output = self.compute_attention_update(x, attn_probs)
            return output, attn_probs

class ColumnSelfAttention(nn.Module):
    """Column-wise self-attention for 2D inputs.
    
    Computes attention across column dimensions for efficient
    processing of sequence position relationships.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        max_tokens_per_msa: int = 2 ** 16,
        num_seqs: int = 33

        
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_tokens_per_msa = max_tokens_per_msa

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

        # learnable
        self.mask_param = nn.Parameter(torch.zeros(num_heads,1,1,num_seqs))

    def _batched_forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attns = []
        for start in range(0, num_cols, max_cols):
            output, attn = self(
                x[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask[:, :, start : start + max_cols]
                if self_attn_padding_mask is not None
                else None,
            )
            outputs.append(output)
            attns.append(attn)
        output = torch.cat(outputs, 1)
        attns = torch.cat(attns, 1)
        return output, attns

    def compute_attention_update(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        if num_rows == 1:
            # if there is only 1 position, this is equivalent and doesn't break with padding
            attn_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=x.device,
                dtype=x.dtype,
            )
            output = self.out_proj(self.v_proj(x))
        else:
            q = self.q_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            q *= self.scaling

            
            attn_weights = torch.einsum("icnhd,jcnhd->hcnij", q, k)
            #add learnable mask
            mask = self.mask_param.unsqueeze(-2).expand(attn_weights.size())
            attn_weights += mask

            if self_attn_mask is not None:
                raise NotImplementedError
            if self_attn_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )

            attn_probs = attn_weights.softmax(-1)
            attn_probs = self.dropout_module(attn_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attn_probs, v)
            context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
            output = self.out_proj(context)
        return output, attn_probs

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        num_rows, num_cols, batch_size, embed_dim = x.size()
        # if False and num_rows * num_cols > 2 ** 14 and not torch.is_grad_enabled():
        if (num_rows * num_cols) > self.max_tokens_per_msa and not torch.is_grad_enabled():
            return self._batched_forward(
                x,
                self_attn_mask,
                self_attn_padding_mask,
            )
        else:
            return self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)
