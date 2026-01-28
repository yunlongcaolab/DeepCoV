
import torch
import torch.nn as nn
import esm
from transformers import  AutoModel
from model.dms_module import DMSLayer
from model.axial_transformer_modules import (
    AxialTransformerLayer,
)

class encoder_msa_transformer(nn.Module):
    """MSA Transformer encoder for continuous prediction.
    
    Enhanced MSA Transformer for continuous time series
    evolution prediction tasks.
    """
    def __init__(self, model_path):
        super().__init__()
        self.encoder,_ = esm.pretrained.load_model_and_alphabet_local(model_path)        

    def forward(self,tokens):
        x = self.encoder.forward(tokens.squeeze(1),[12])['representations'][12][:,:,1:,:]
   
        return x#(b,1+n_bg,201,768)
    
class encoder_esm2_150m(nn.Module):
    """ESM-2 150M model encoder.
    
    Uses ESM-2 transformer for sequence embedding
    generation with continuous prediction support.
    """
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
    """LSTM encoder for continuous prediction.
    
    Enhanced LSTM for processing temporal sequences
    in continuous evolution prediction tasks.
    """
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
    """LSTM output layer for continuous predictions.
    
    Final LSTM layer for generating continuous time
    series predictions in evolution modeling.
    """
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
    """Continuous time series proportion prediction model.
    
    Advanced model for predicting continuous evolution
    trajectories across multiple time points.
    """
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

        #bg ratios encode 
        if self.config['model']['bg_ratios_encoder']:
            bg_ratios_emded = self.bg_ratios_feature_expand(self.bg_ratios_encoder(input['background_ratios']).unsqueeze(-2).expand(-1,-1,target_and_bg_msa_emded.size(2),-1))#(b,1+n_bg,201,128)

        #target dms
        if self.dms_num_features != 0:
            with torch.no_grad():
                dms_ref_embedding = {}
                for dms_type in self.dms_feature_choices:
                    dms_ref_embedding[dms_type]=torch.split(tensor = self.seq_encoder(input[f'{dms_type}_dms_ref_msa_tokens']) if self.config['model']['seq_encoder'] == 'msa_transformer' else self.seq_encoder(input[f'{dms_type}_dms_ref_msa_tokens'],input[f'{dms_type}_dms_ref_msa_attention_mask']),
                                                            split_size_or_sections = [self.config['feature']['top_k'] + 1,input[f'{dms_type}_dms_value'].shape[1]],
                                                            dim=1)
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
        




    
    