import torch
import torch.nn as nn

    
class dms_class_weight_landsape(nn.Module):
    def __init__(self, n_class):
        super(dms_class_weight_landsape, self).__init__()
        self.n_class = n_class
        self.dms_class_weights = nn.Parameter(torch.ones(n_class))

    def forward(self, ref_socres,dms_landscape):
        class_dms_list = []
        for i in range(self.n_class):
            dms_landscape_class = dms_landscape[:,:,i,:,:] 
            ref_socres_new = ref_socres.unsqueeze(-1)
            dms_landscape_class = dms_landscape_class.unsqueeze(1)
            ref_weighted_dms_landscape = torch.sum(ref_socres_new * dms_landscape_class,dim=2)
            class_dms_list.append(ref_weighted_dms_landscape)
        weighted_tensors = [weight * tensor for weight, tensor in zip(self.dms_class_weights, torch.stack(class_dms_list))]
        weighted_sum = torch.sum(torch.stack(weighted_tensors), dim=0)
        return weighted_sum,self.dms_class_weights
    
class DMSLayer(nn.Module):
    def __init__(self, in_features = 768,seq_dim=64, dms_dim=64, n_cluster=13,latent_dim = 8,dms_class_weight = None):
        super().__init__()

        self.dms_class_weight =dms_class_weight
        # self.vae = vae

        self.s_emb = nn.Linear(in_features=in_features, out_features=seq_dim)
        self.s_norm = nn.LayerNorm(seq_dim)
        
        if dms_class_weight:
            self.d_emb = nn.Linear(in_features=21, out_features=dms_dim)
            trans_layer = nn.TransformerEncoderLayer(dms_dim, nhead=8, dim_feedforward=128, batch_first=True)
            self.trans_encoder = nn.TransformerEncoder(trans_layer, num_layers=2) 
            self.dms_class_weight_landsape_layer = dms_class_weight_landsape(n_cluster)
        else:
            self.d_emb = nn.Linear(in_features=n_cluster * 21, out_features=dms_dim)
            trans_layer = nn.TransformerEncoderLayer(dms_dim, nhead=8, dim_feedforward=128, batch_first=True)
            self.trans_encoder = nn.TransformerEncoder(trans_layer, num_layers=2)
        self.dms_norm = nn.LayerNorm(dms_dim)

    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, target_embedding, ref_embedding, x_dms,dms_delay_mask):
        """

        Args:
            target_embedding: (b, n1, r, d)#n1=1
            ref_embedding: (b, n2, r, d)
            x_dms: (b, n2, r, c, 21)

        Returns:

        """
        target_embedding = target_embedding[:,0,:,:].unsqueeze(1)

        n1 = 1
        n2 = ref_embedding.size()[1]

        x_emb = torch.concat([target_embedding, ref_embedding], dim=1)
        x_emb = self.s_norm(self.s_emb(x_emb))
        x_emb = torch.nn.functional.normalize(x_emb, dim=-1)
        x_bg, x_ref = torch.split(x_emb, [n1, n2], dim=1)

        x_bg_b, x_ref_b = torch.broadcast_tensors(x_bg[:, :, None, :, :], x_ref[:, None, :, :, :])
        r = torch.einsum('bnrmc,bnrmc->bnrm', x_bg_b, x_ref_b)

        dms_delay_mask_reshaped = dms_delay_mask.unsqueeze(1).unsqueeze(-1).expand(-1,n1,-1,r.size()[-1])
        scores=r.masked_fill(dms_delay_mask_reshaped == 0,float('-inf'))
        s_exp=torch.exp(scores)
        s_exp_s = torch.sum(s_exp,dim =2,keepdim=True)
        s_mask=s_exp_s.masked_fill(s_exp_s == 0, 1e-8)
        w = s_exp / s_mask #（b,n1,n2,201）
       


        # for dms
        if self.dms_class_weight:
            x_dms = x_dms.permute(0,1,3,2,4)
            n_batch, n_ref,n_cluster,n_res, n_mutant = x_dms.size()
            x_dms = x_dms.reshape(n_batch * n_ref * n_cluster, n_res, n_mutant)
            x_dms = self.d_emb(x_dms)
            x_dms = self.trans_encoder(x_dms)
            # (b, n2, c, r, dms_dim)
            x_dms = x_dms.view(n_batch, n_ref, n_cluster,n_res, -1)
            y,dms_class_weights = self.dms_class_weight_landsape_layer(w,x_dms)
            y =self.dms_norm(y)
            output = {'dms_feature':y.squeeze(1),
                      'dms_class_weights':dms_class_weights}
        else:
            n_batch, n_ref, n_res, n_cluster, n_mutant = x_dms.size()
            x_dms = x_dms.view(n_batch * n_ref, n_res, n_cluster * n_mutant)
            x_dms = self.d_emb(x_dms)
            x_dms = self.trans_encoder(x_dms)

            # (b, n2, r, dms_dim)
            x_dms = x_dms.view(n_batch, n_ref, n_res, -1)

            x_dms_w = x_dms[:, None] * w[..., None]

            # (b, n1, r, dms_dim)
            y = torch.sum(x_dms_w, dim=2)
            y = self.dms_norm(y)
            output = {'dms_feature':y.squeeze(1)}
            
        return output
