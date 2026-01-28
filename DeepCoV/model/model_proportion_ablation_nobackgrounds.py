import torch
import torch.nn as nn
import esm
from transformers import  AutoModel

class LSTM_output_layer(nn.Module):
    """LSTM output layer for no-background ablation model.
    
    Specialized LSTM output for ablation studies
    without background ratio features.
    """
    def __init__(self, input_size, hidden_size, num_layers,time_step):
        super().__init__()
        self.time_encoder = nn.Embedding(time_step,input_size)
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.1)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size // 2, 1),
                                          nn.Sigmoid())

    def forward(self, x):
        x = x.unsqueeze(1).expand(-1,self.time_step,-1)
        time = torch.arange(0,self.time_step).unsqueeze(0).expand(x.size(0),-1).to(x.device)
        time_embed =  self.time_encoder(time)#(b,time_point,dim)

        x = x + time_embed

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.output_layer(out[:,-1,:])
        
        return logits

class encoder_msa_transformer(nn.Module):
    """MSA Transformer encoder for no-background ablation.
    
    Specialized encoder for ablation studies
    excluding background ratio features.
    """
    def __init__(self, model_path):
        super().__init__()
        self.encoder,_ = esm.pretrained.load_model_and_alphabet_local(model_path)        

    def forward(self,tokens):
        x = self.encoder.forward(tokens.squeeze(1),[12])['representations'][12][:,:,1:,:]
   
        return x#(b,1+n_bg,201,768)
    
class encoder_esm2_150m(nn.Module):
    """ESM-2 150M encoder for no-background ablation.
    
    Specialized ESM-2 encoder for ablation studies
    without background ratio features.
    """
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
    def forward(self,tokens, attention_mask): #640
        x = self.encoder.forward(tokens,attention_mask).last_hidden_state[:,1:-1,:]
        return x#(b，201,640)

class encoder_lstm(nn.Module):
    """Enhanced LSTM encoder for ablation studies.
    
    Advanced LSTM with multi-layer and fusion capabilities
    for comprehensive ablation analysis.
    """
    def __init__(self, hidden_size, num_layers,bg_len, seq_embed_dim,msa_lstm_fusion,bg_ratios_encoder_multi_lstm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.multi_lstm = bg_ratios_encoder_multi_lstm
        self.msa_lstm_fusion = msa_lstm_fusion
        
        if bg_ratios_encoder_multi_lstm:
            self.lstm_list = nn.ModuleList([
                nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.1)
                for _ in range(bg_len) 
            ])
        else:
            self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.1)
        if self.msa_lstm_fusion == 'attention':
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.seq_proj = nn.Linear(seq_embed_dim, hidden_size)  
        
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.out_viewed_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.seq_attention_proj = nn.Linear(seq_embed_dim, hidden_size)
        self.seq_attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.seq_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.seq_out_linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, seq_states):
        x = x.unsqueeze(-1)
        batch,  len_bg, _ = x.size()
        n_target_bg = 1
        x_viewed = x.view(batch*n_target_bg, len_bg, -1)#(batch*n_target_bg, len_bg, 1)
        seq_states_pooled = seq_states.unsqueeze(1).mean(dim=2)
        seq_proj = self.seq_proj(seq_states_pooled)#(batch,n_target_bg, self.hidden_size)
      
        h = torch.zeros(self.num_layers, x_viewed.size(0), self.hidden_size).to(x_viewed.device)
        c = torch.zeros(self.num_layers, x_viewed.size(0), self.hidden_size).to(x_viewed.device)
        
        # 按时间步更新
        for t in range(len_bg):
            if self.multi_lstm:
                _, (new_h, new_c) = self.lstm_list[t](x_viewed[:, t:t+1, :], (h, c))
            else:
                _, (new_h, new_c) = self.lstm(x_viewed[:, t:t+1, :], (h, c))
            
            last_h = new_h[-1].view(batch,n_target_bg,self.hidden_size)# (batch,1+b_bg,hidden_size)
            
            if self.msa_lstm_fusion == 'attention':
                attn_out, _ = self.attention(
                    query=last_h,# (batch，1+b_bg,  self.hidden_size)
                    key=seq_proj,# (batch，1+b_bg,  self.hidden_size)
                    value=last_h# (batch，1+b_bg,  self.hidden_size)
                )# (batch，1+b_bg,  self.hidden_size)
                gate = self.update_gate(torch.cat([last_h, attn_out], dim=-1))
                updated_last_h = gate * last_h + (1 - gate) * attn_out
            elif self.msa_lstm_fusion == 'liner':
                gate = self.update_gate(torch.cat([last_h, seq_proj], dim=-1))
                updated_last_h = gate * last_h + (1 - gate) * seq_proj

            new_h = h.clone()
            new_h[-1] = updated_last_h.view(batch*n_target_bg, self.hidden_size)
            h = new_h
            c = new_c.clone()
        out_viewed = h[-1].view(batch, n_target_bg, self.hidden_size)
        out_viewed_expand = self.out_viewed_linear(out_viewed.unsqueeze(2).expand(-1, -1, seq_states.size(1), -1))
        
        #用lstm输出更新序列的表达
        seq_states_proj = self.seq_attention_proj(seq_states_pooled)
        seq_attn_out, _ = self.seq_attention(
            query=seq_states_proj,
            key=out_viewed,
            value=seq_states_proj
        )
        seq_output = self.seq_linear(seq_states_proj + seq_attn_out)
        seq_output_expand = self.seq_out_linear(seq_output.unsqueeze(2).expand(-1, -1, seq_states.size(1), -1))

        return out_viewed_expand,seq_output_expand # (batch, 1+n_bg,seq_len, hidden_size)

class proportion_single_t1(nn.Module):
    """No-background ablation proportion prediction model.
    
    Specialized model for ablation studies
    excluding background ratio features.
    """
    def __init__(self,config):
        super().__init__()
        self.config = config

        dim = 640
        self.seq_encoder =  encoder_esm2_150m(model_path = config['dataset']['path']['esm2_150m_model_path'])

        
        #bg ratios encoder
        if config['model']['bg_ratios_encoder']:

            self.bg_ratios_encoder = encoder_lstm(hidden_size = config['model']['bg_ratios_encoder_dim'], #（batch,1+nbg,60）
                                                num_layers = config['model']['bg_ratios_encoder_num_layers'],
                                                bg_len = config['feature']['n_bg_days'] // config['feature']['stride'], 
                                                seq_embed_dim = dim,
                                                msa_lstm_fusion = 'attention',
                                                bg_ratios_encoder_multi_lstm = True
                                                )
            
        
        #msa_feature + bg_ratios_feature
        if config['model']['bg_ratios_encoder']:
            self.liner_msa_feature_and_bg_ratios_feature = nn.Sequential(nn.Linear(dim+config['model']['bg_ratios_encoder_dim'] *2,(dim+config['model']['bg_ratios_encoder_dim'])),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear((dim+config['model']['bg_ratios_encoder_dim']) ,128))
        else:
            self.liner_msa_feature_and_bg_ratios_feature = nn.Sequential(nn.Linear(dim,dim // 2),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear(dim // 2,128))
        


        #transformer layer 
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu',
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=config['model']['transformer_num_layers'])


        #output layer
        self.output_layer = nn.Sequential(nn.Linear(config['model']['dms_layer_dim'], config['model']['dms_layer_dim'] // 2),
                                          nn.ReLU(),
                                          nn.Linear(config['model']['dms_layer_dim'] // 2, 1),
                                          nn.Sigmoid())
        
    def forward(self,input):
        #target and bg seq encode 
        with torch.no_grad():
            target_and_bg_msa_emded = self.seq_encoder(input['tokens'][:, 0, ...],input['attention_mask'][:, 0, ...])#（b,201,640）
            

        #bg ratios encode 
        if self.config['model']['bg_ratios_encoder']:
            bg_ratios_emded,seq_bgratios= self.bg_ratios_encoder(input['background_ratios'][:, 0, ...],target_and_bg_msa_emded)#(b,1,201,128)

        ##msa_transformer_feature + bg_ratios_feature
        if self.config['model']['bg_ratios_encoder']:
            x=self.liner_msa_feature_and_bg_ratios_feature(torch.cat((target_and_bg_msa_emded,bg_ratios_emded.squeeze(dim=1),seq_bgratios.squeeze(dim=1)),dim=-1))#(b,201,640+128*2)
        else:
            x = self.liner_msa_feature_and_bg_ratios_feature(target_and_bg_msa_emded)#(b,201,640)

        #transformer
        cls_token = torch.zeros(x.size(0), 1, x.size(2)).to(x.device) 
        x = torch.cat((cls_token, x), dim=1) 
        x = self.transformer_encoder(x)[:,0,:]#(b,dim)

        logits = self.output_layer(x)


        
        return logits,None
        
       
        




    
    