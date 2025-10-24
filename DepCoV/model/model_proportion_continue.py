from collections import defaultdict
import torch
import torch.nn as nn
import esm
import torch.nn.functional as F
from typing import Optional
import math
from transformers import  AutoModel

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
    """Implements an Axial MSA Transformer block."""

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
    """Compute self-attention over rows of a 2D input."""

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
    """Compute self-attention over columns of a 2D input."""

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




# class LSTM_output_layer(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers,time_step):
#         super().__init__()
#         self.time_encoder = nn.Embedding(time_step,input_size)
#         self.time_step = time_step
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.1)
#         self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
#                                           nn.ReLU(),
#                                           nn.Linear(hidden_size // 2, 1),
#                                           nn.Sigmoid())

#     def forward(self, x):
#         x = x.unsqueeze(1).expand(-1,self.time_step,-1)
#         time = torch.arange(0,self.time_step).unsqueeze(0).expand(x.size(0),-1)
#         time_embed =  self.time_encoder(time)#(b,time_point,dim)

#         x = x + time_embed

#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         # Initialize cell state
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
#         # One time step
#         out, (hn, cn) = self.lstm(x, (h0, c0))
#         logits = self.output_layer(out[:,-1,:])
        
#         return logits


class encoder_msa_transformer(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder,_ = esm.pretrained.load_model_and_alphabet_local(model_path)        

    def forward(self,tokens):
        x = self.encoder.forward(tokens.squeeze(1),[12])['representations'][12][:,:,1:,:]
   
        return x#(b,1+n_bg,201,768)
    
class encoder_esm2_150m(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
    def forward(self,tokens, attention_mask): #640
        tokens_list =[]
        for i in range(0,tokens.shape[0]):
            x = self.encoder.forward(tokens[i,:,:],attention_mask[i,:,:]).last_hidden_state[:,1:-1,:]
            tokens_list.append(x)
        x = torch.stack(tokens_list, dim=0)
        return x#(b,1+n_bg,201,640)

class encoder_lstm(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,dropout=0.1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch,n_target_bg,len_bg,_ = x.size()
        x_viewed = x.view(batch*n_target_bg,len_bg,-1)
        
        h0 = torch.zeros(self.num_layers, x_viewed.size(0), self.hidden_size).to(x_viewed.device)
        c0 = torch.zeros(self.num_layers, x_viewed.size(0), self.hidden_size).to(x_viewed.device)
        
        out, (hn, cn) = self.lstm(x_viewed, (h0, c0))
        out = out[:, -1, :].unsqueeze(1)

        out_viewed = out.view(batch,n_target_bg,self.hidden_size)
       
        return out_viewed#(batch,1+n_bg,128)

class LSTM_output_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,time_step):
        super().__init__()
        self.time_encoder = nn.Embedding(time_step,input_size)
        self.time_step = time_step
        self.hidden_size = hidden_size *2
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size *2 , self.hidden_size, num_layers, batch_first=True,dropout=0.1)
        self.output_layer = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size // 2),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size // 2, 1),
                                          nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1,self.time_step,-1)
        time = torch.arange(0,self.time_step).unsqueeze(0).expand(x.size(0),-1).to(x.device)
        time_embed =  self.time_encoder(time)#(b,time_point,dim)
        
        x = torch.cat((x,time_embed), dim=-1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.output_layer(out)
        
        return logits

class proportion_continue_t1(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config

        self.dms_feature_choices = config['feature']['dms']
        self.dms_num_features = len(self.dms_feature_choices)  if  config['feature']['dms'] is not None else 0
        
        #seq encoder
        if config['model']['seq_encoder'] == 'msa_transformer':
            dim = 768
            self.seq_encoder = encoder_msa_transformer(model_path = config['dataset']['path']['msa_transformer_model_path'])
        elif config['model']['seq_encoder'] == 'esm2_150m':
            dim = 640
            self.seq_encoder =  encoder_esm2_150m(model_path = config['dataset']['path']['esm2_150m_model_path'])

        
        #bg ratios encoder
        if config['model']['bg_ratios_encoder']:
            self.bg_ratios_encoder = encoder_lstm(hidden_size =  config['model']['bg_ratios_encoder_dim'], 
                                                num_layers =  config['model']['bg_ratios_encoder_num_layers'])
            self.bg_ratios_feature_expand = nn.Sequential(nn.Linear(config['model']['bg_ratios_encoder_dim'],256),
                                                      nn.ReLU(),
                                                      nn.Linear(256,config['model']['bg_ratios_encoder_dim']))
            
        
        #msa_feature + bg_ratios_feature
        if config['model']['bg_ratios_encoder']:
            self.liner_msa_feature_and_bg_ratios_feature = nn.Sequential(nn.Linear(dim+config['model']['bg_ratios_encoder_dim'],(dim+config['model']['bg_ratios_encoder_dim']) // 2),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear((dim+config['model']['bg_ratios_encoder_dim']) // 2,config['model']['evopredict_embedding_dim']))
        else:
            self.liner_msa_feature_and_bg_ratios_feature = nn.Sequential(nn.Linear(dim,dim // 2),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear(dim // 2,config['model']['evopredict_embedding_dim']))
        
        
        
        #dms encoder
        if self.dms_num_features != 0:
            self.dms_encoder = nn.ModuleDict({
                f'{self.dms_feature_choices[i]}_encoder': DMSLayer(in_features = dim,
                                                                seq_dim = config['model']['DMS_encoder_seq_dim'], 
                                                                dms_dim = config['model']['DMS_encoder_out_dim'], 
                                                                n_cluster = 1 if self.dms_feature_choices[i] != 'ab_escape' else config['feature']['ab_cluster'],
                                                                latent_dim = 8,
                                                                dms_class_weight = config['model']['dms_ab_class_weight'],
                                                                # vae = self.config['model']['vae']
                                                                )  
                for i in range(self.dms_num_features)
            })

       
        #evopredict module
        self.evo_AxialTransformer = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim = config['model']['evopredict_embedding_dim'],
                    num_seqs = config['feature']['top_k'] + 1
                )
                for _ in range(config['model']['evoprediction_num_layers'])
            ]
        )

        #evopredict + dms
        if self.dms_num_features != 0:
            if 'ab_escape' in  self.dms_feature_choices:
                self.ab_escape = nn.Sequential(nn.Linear(config['model']['evopredict_embedding_dim'] + config['model']['DMS_encoder_out_dim'],(config['model']['evopredict_embedding_dim'] + config['model']['DMS_encoder_out_dim']) // 2),
                                            nn.ELU(),
                                            nn.Linear((config['model']['evopredict_embedding_dim'] + config['model']['DMS_encoder_out_dim']) // 2,config['model']['dms_layer_dim']))
                self.layer_norm_escape  = nn.LayerNorm(config['model']['dms_layer_dim'])
            
                self.other_dms = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim'],(config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim']) // 2),
                        nn.ELU(),
                        nn.Linear((config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim']) // 2,config['model']['dms_layer_dim']),
                        nn.LayerNorm(config['model']['dms_layer_dim'])
                    )
                    for _ in range(self.dms_num_features - 1)
                ])
            else:
                self.x_reduce = nn.Sequential(nn.Linear(config['model']['evopredict_embedding_dim'] ,config['model']['evopredict_embedding_dim']  // 2),
                                            nn.ELU(),
                                            nn.Linear(config['model']['evopredict_embedding_dim']  // 2,config['model']['dms_layer_dim']))
                self.other_dms = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim'],(config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim']) // 2),
                        nn.ELU(),
                        nn.Linear((config['model']['dms_layer_dim'] + config['model']['DMS_encoder_out_dim']) // 2,config['model']['dms_layer_dim']),
                        nn.LayerNorm(config['model']['dms_layer_dim'])
                    )
                    for _ in range(self.dms_num_features)
                ])
          
        else:
            self.substitute_dms = nn.Sequential(nn.Linear(config['model']['evopredict_embedding_dim'] ,config['model']['evopredict_embedding_dim']  // 2),
                                            nn.ReLU(),
                                            nn.Linear(config['model']['evopredict_embedding_dim'] // 2,config['model']['dms_layer_dim']))

        #transformer layer 
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['model']['dms_layer_dim'],
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=config['model']['transformer_num_layers'])


        # output layer
        # if self.config['model']['lstm']:
        #     self.lstm_output_layer = LSTM_output_layer(input_size = config['model']['dms_layer_dim'], 
        #                                                hidden_size = config['model']['dms_layer_dim'], 
        #                                                num_layers = config['model']['output_layer_nums'],
        #                                                 time_step = config['label']['target_t1_point'])
        # else:
        #     self.output_layer = nn.Sequential(nn.Linear(config['model']['dms_layer_dim'], config['model']['dms_layer_dim'] // 2),
        #                                   nn.ReLU(),
        #                                   nn.Linear(config['model']['dms_layer_dim'] // 2, 1),
        #                                   nn.Sigmoid())
        self.lstm_output_layer = LSTM_output_layer(input_size = config['model']['dms_layer_dim'], 
                                                    hidden_size = config['model']['dms_layer_dim'], 
                                                    num_layers = config['model']['lstm_output_num_layers'],
                                                    time_step = config['label']['target_t1_point'])
        # self.output_layer = nn.Sequential(nn.Linear(config['model']['dms_layer_dim'], config['model']['dms_layer_dim'] // 2),
        #                                   nn.ReLU(),
        #                                   nn.Linear(config['model']['dms_layer_dim'] // 2, 1),
        #                                   nn.Sigmoid())
        
    def forward(self,input):
        #target and bg seq encode 
        with torch.no_grad():
            if self.config['model']['seq_encoder'] == 'msa_transformer':
                target_and_bg_msa_emded = self.seq_encoder(input['tokens'])#(b,1+n_bg,201,768)
            elif self.config['model']['seq_encoder'] == 'esm2_150m':
                target_and_bg_msa_emded = self.seq_encoder(input['tokens'],input['attention_mask'])#(b,1+n_bg,201,640)
            # print('target_and_bg_msa_emded.shape: ',target_and_bg_msa_emded)   

        #bg ratios encode 
        if self.config['model']['bg_ratios_encoder']:
            bg_ratios_emded = self.bg_ratios_feature_expand(self.bg_ratios_encoder(input['background_ratios']).unsqueeze(-2).expand(-1,-1,target_and_bg_msa_emded.size(2),-1))#(b,1+n_bg,201,128)
            # print('bg_ratios_emded.shape: ',bg_ratios_emded) 

        #target dms
        if self.dms_num_features != 0:
            with torch.no_grad():
                dms_ref_embedding = {}
                for dms_type in self.dms_feature_choices:
                    dms_ref_embedding[dms_type]=torch.split(tensor = self.seq_encoder(input[f'{dms_type}_dms_ref_msa_tokens']) if self.config['model']['seq_encoder'] == 'msa_transformer' else self.seq_encoder(input[f'{dms_type}_dms_ref_msa_tokens'],input[f'{dms_type}_dms_ref_msa_attention_mask']),
                                                            split_size_or_sections = [self.config['feature']['top_k'] + 1,input[f'{dms_type}_dms_value'].shape[1]],
                                                            dim=1)
                    # print(dms_ref_embedding[dms_type][0].shape)
            dms_embed_dict = {} 
            for dms_type in self.dms_feature_choices:
                dms_embed_dict[dms_type] = self.dms_encoder[f'{dms_type}_encoder'](target_embedding = dms_ref_embedding[dms_type][0], 
                                                                ref_embedding = dms_ref_embedding[dms_type][1], 
                                                                x_dms = input[f'{dms_type}_dms_value'],
                                                        dms_delay_mask = input[f'{dms_type}_delay_mask'])#(b,201,dim)
            if 'ab_escape' in  self.dms_feature_choices:
                # ab_escape_embed =  dms_embed_dict['ab_escape']['vae_dms_reconstructed'] if self.config['model']['vae'] else dms_embed_dict['ab_escape']['dms_feature']
                ab_escape_embed =  dms_embed_dict['ab_escape']['dms_feature']
                dms_embed_dict.pop('ab_escape')#返回被删除的key的对应value

        ##msa_transformer_feature + bg_ratios_feature
        if self.config['model']['bg_ratios_encoder']:
            msa_bg_ratios=self.liner_msa_feature_and_bg_ratios_feature(torch.cat((target_and_bg_msa_emded,bg_ratios_emded),dim=-1))
        else:
            msa_bg_ratios = self.liner_msa_feature_and_bg_ratios_feature(target_and_bg_msa_emded)#(b,1+n_bg,201,d)
        # print('msa_bg_ratios.shape: ',msa_bg_ratios.shape)

        #add target indicator
        indicator = torch.zeros(msa_bg_ratios.size(0), msa_bg_ratios.size(1), 1,msa_bg_ratios.size(3)).to(msa_bg_ratios.device)
        indicator[:, 0, :, :] = 1
        x = torch.cat((indicator,msa_bg_ratios),dim=-2)

        #evopredict
        x = x.permute(1, 2, 0, 3)
        for layer in self.evo_AxialTransformer:
            x = layer(x)
        x = x.permute(2, 0, 1, 3) 
        x = x[:,0,1:,:]#(b,201,dim)

        #evopredict + dms_embed
        if self.dms_num_features != 0:
            #1.ab_escape
            if 'ab_escape' in  self.dms_feature_choices:
                x = torch.cat((x,ab_escape_embed),dim=-1)
                x = self.layer_norm_escape(self.ab_escape(x))
            else:
                x = self.x_reduce(x)
            #2.other
            for idx,dms_f in enumerate(dms_embed_dict.items()):
                # x = torch.cat((x,dms_f[1]['vae_dms_reconstructed'] if self.config['model']['vae'] else dms_f[1]['dms_feature']),dim=-1)
                x = torch.cat((x,dms_f[1]['dms_feature']),dim=-1)
                x = self.other_dms[idx](x)#(b,201,dim)
        else:
            x = self.substitute_dms(x)
    
        #transformer
        cls_token = torch.zeros(x.size(0), 1, x.size(2)).to(x.device) 
        x = torch.cat((cls_token, x), dim=1) 
        x = self.transformer_encoder(x)[:,0,:]#(b,dim)


        #output
        # if self.config['model']['lstm']:
        #     logits = self.lstm_output_layer(x)
        # else:
        #     logits = self.output_layer(x)
        logits = self.lstm_output_layer(x) # LSTM for continue prediction (only change the output layer)
        print('logits.shape: ',logits.shape)


        if self.dms_num_features != 0:
            
            return logits,dms_embed_dict
        else:
            return logits,None
        




    
    