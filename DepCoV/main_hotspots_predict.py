import argparse
import torch
import torch.nn as nn
import torch.distributed
import torch.utils
import torch.utils.data
import pandas as pd 
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import functools
from scipy.stats import pearsonr, spearmanr
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau as lr_sheduler_use
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from utils.modules import (
    AxialTransformerLayer
)
import yaml
import wandb
import numpy as np
import datetime
import random


from dataset.dataset_proportion_hotspots import SARS2SpikeProteinDataset_EvoData as SARS2SpikeProteinDataset

torch.autograd.set_detect_anomaly(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    os.environ['PYTHONHASHSEED'] = str(seed)  

set_seed(721)

from scipy.interpolate import interp1d
from scipy.integrate import simps

class EarlyStopping:
    def __init__(self,config,patience=7, verbose=False, delta=0):

        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.config = config
        self.patience = config['hyperparameters']['early_stopping_patience']
       

    def __call__(self, val_loss,states, epoch,scheduler,optimizer_model,name,global_step):
    
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, states,epoch,scheduler,optimizer_model,name,global_step)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.save_checkpoint(val_loss, states,epoch,scheduler,optimizer_model,name,global_step)
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, states,epoch,scheduler,optimizer_model,name,global_step)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss,states, epoch,scheduler,optimizer_model,name,global_step):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
        if self.config["other"]["save_model_train"]:
            save_model(self.config,epoch,scheduler,states,optimizer_model, name,global_step)
       
    def reset(self):
        self.best_score = None
        self.counter = 0
        self.val_loss_min = np.Inf


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def log_metrics_eval(config,
                     all_date_eval_gather,
                     all_location_eval_gather,
                     all_label_cla_eval_gather,
                     all_target_cluster_eval_gather,
                     all_regres_predictions_eval_gather,
                     name,
                     global_step,
                     dataset_csv,
                     val_dataset,
                     status):
    status_log = status.split("_")[-1]
    
    
    csv_file_path_regres = os.path.join(config['other']['output_dir'], name,f"{status_log}_regres_outputs_labels-step-{global_step}.csv")
    target_ratio_t1_output,target_ratio_t1_label = save_to_csv_regres(config,dataset_csv,all_label_cla_eval_gather,all_date_eval_gather,all_location_eval_gather,all_target_cluster_eval_gather,all_regres_predictions_eval_gather, csv_file_path_regres,val_dataset)
    print(f"Regression outputs and labels saved to {csv_file_path_regres}")
    return None

def save_model(config,epoch,scheduler,states ,optimizer_model, name,global_step):
    save_path = os.path.join(config["other"]['output_dir'],name, f"checkpoint-epoch-{epoch+1}-step-{global_step}.pt")
    checkpoint = {
    'epoch': epoch+1,
    'model_state_dict': states,
    'model_optimizer_state_dict': optimizer_model.state_dict(),
    'scheduler':scheduler.state_dict(),
    'global_step':global_step
}
    torch.save(checkpoint, save_path) 
    print(f"Model saved at epoch {epoch+1} to {save_path}")
 
def _add_t1_infor(group ,val_dataset,config):
        group_names = group.name
        loc_ind = val_dataset.reader_a.name2index('location',group_names[1])
        seq_ind = val_dataset.reader_a.name2index('sequence_name',group_names[2])
        t0_ind = val_dataset.reader_a.name2index('date',group_names[0])
        t1_info=val_dataset.reader_a.get_t1_values(loc_index = loc_ind,
                                               seq_index = seq_ind,
                                               t0_index = t0_ind,
                                               min_total_isolates_t1=config["dataset"]['min_total_isolates_t1'])
        group['target_isolates_t1'] = t1_info['target_isolates_t1'][config['label']['target_t1_point']-1]
        group['total_isolates_t1'] = t1_info['total_isolates_t1'][config['label']['target_t1_point']-1]
        group['target_ratio_t1_label'] = t1_info['target_ratio_t1'][config['label']['target_t1_point']-1]
        group['target_ratio_t1_mask'] = t1_info['target_ratio_t1_mask'][config['label']['target_t1_point']-1]
        return group

def save_to_csv_regres(config,dataset_csv,predict_time,date,location,target,outputs, file_path,val_dataset):
    tag = config['dataset']['tag']
    df = pd.DataFrame({
        "t0":date,
        'location':location,
        f'{tag}_name':target,
        't1':predict_time,
        'target_ratio_t1_output':outputs,
    })
   
    results_pd = pd.merge(df, dataset_csv, how='left', on=["t0",'location',f'{tag}_name'], sort=True, suffixes=('_output', '_label'), copy=True)
    results_pd[['target_isolates_t1','total_isolates_t1','target_ratio_t1_label','target_ratio_t1_mask']] = None
    results_pd.to_csv(file_path, index=False)
    return results_pd['target_ratio_t1_output'],results_pd['target_ratio_t1_label']

def gather_tensors(rank, tensors):
    if rank == 0:
        gathered_tensors = [torch.empty_like(tensors) for _ in range(dist.get_world_size())]
    else:
        gathered_tensors = None
    dist.barrier()
    dist.gather(tensors, gather_list=gathered_tensors, dst=0)
    dist.barrier()
    return gathered_tensors


def validation(config,model,rank,local_rank,world_size,val_loader,val_dataset,name,criterion_regression,global_step,dataset_csv,status):
    tag = config['dataset']['tag']
    model.eval()
    
    concatenated_all_date_eval_gather_list = []
    concatenated_all_location_eval_gather_list= []
    concatenated_all_label_cla_eval_gather_list= []
    concatenated_all_target_cluster_eval_gather_list= []
    concatenated_all_regres_predictions_eval_gather_list= []

   
    if rank == 0:
        val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"{status} at step {global_step}")
    else:
        val_progress_bar = enumerate(val_loader)
    
    with torch.no_grad():
        for val_step,val_batch in val_progress_bar:
            for key in val_batch.keys():
                val_batch[key] = val_batch[key].to(local_rank)
            
            logits_regression,vae_out = model(val_batch)
            logits_regression = logits_regression.squeeze(-1)
            print('logits_regression.shape: ',logits_regression.shape)
            
            
            all_target_cluster_eval=torch.repeat_interleave(val_batch["target_name"].unsqueeze(-1),1,dim=1).flatten().tolist()
            print('all_target_cluster_eval: ',all_target_cluster_eval)
            all_date_eval=torch.repeat_interleave(val_batch["t0_index"].unsqueeze(-1),1,dim=1).flatten().tolist()
            all_location_eval=torch.repeat_interleave(val_batch["location_index"].unsqueeze(-1),1,dim=1).flatten().tolist()
            outputs_list=logits_regression.flatten().tolist()
            all_label_cla_eval=[config['label']['target_t1_point']] * logits_regression.size(0)
           
            
            all_date_eval_tensor = torch.tensor(all_date_eval).to(local_rank)
            all_location_eval_tensor = torch.tensor(all_location_eval).to(local_rank)
            all_label_cla_eval_tensor = torch.tensor(all_label_cla_eval).to(local_rank)
            all_target_cluster_eval_tensor = torch.tensor(all_target_cluster_eval).to(local_rank)
            all_regres_predictions_eval_tensor = torch.tensor(outputs_list).to(local_rank)

            gather_tensor_list = []
            for t in [all_date_eval_tensor,all_location_eval_tensor,all_label_cla_eval_tensor,all_target_cluster_eval_tensor,all_regres_predictions_eval_tensor]:
                gather_tensor_list.append(gather_tensors(rank, t))
        
        
            if rank == 0:
                concat_gather_tensor_list = []
                for g in gather_tensor_list:
                    concat_gather_tensor_list.append(torch.cat(g, dim=0))

                concatenated_all_date_eval_gather_list.append(concat_gather_tensor_list[0].detach().cpu())
                concatenated_all_location_eval_gather_list.append(concat_gather_tensor_list[1].detach().cpu())
                concatenated_all_label_cla_eval_gather_list.append(concat_gather_tensor_list[2].detach().cpu())
                concatenated_all_target_cluster_eval_gather_list.append(concat_gather_tensor_list[3].detach().cpu())
                concatenated_all_regres_predictions_eval_gather_list.append(concat_gather_tensor_list[4].detach().cpu())
        
        print(f"Rank1 {rank} reached barrier")
        dist.barrier()
        print(f"Rank1 {rank} passed barrier") 

        if rank == 0:
            concatenated_all_date_eval_gather_list_cat = torch.cat(concatenated_all_date_eval_gather_list,dim=0)
            concatenated_all_location_eval_gather_list_cat = torch.cat(concatenated_all_location_eval_gather_list,dim=0)
            concatenated_all_label_cla_eval_gather_list_cat = torch.cat(concatenated_all_label_cla_eval_gather_list,dim=0)
            concatenated_all_target_cluster_eval_gather_list_cat = torch.cat(concatenated_all_target_cluster_eval_gather_list,dim=0)
            concatenated_all_regres_predictions_eval_gather_list_cat = torch.cat(concatenated_all_regres_predictions_eval_gather_list,dim=0)

            concatenated_all_date_eval_gather_list_cat_out=val_dataset.reader_a.index2name('date', concatenated_all_date_eval_gather_list_cat.tolist())
            concatenated_all_location_eval_gather_list_cat_out = val_dataset.reader_a.index2name('location', concatenated_all_location_eval_gather_list_cat.tolist())
            concatenated_all_label_cla_eval_gather_list_cat_out = concatenated_all_label_cla_eval_gather_list_cat.tolist()
            concatenated_all_target_cluster_eval_gather_list_cat_out = [
                'n' + str(i) if i >= 1000000 else config['dataset']['tag'][0] + str(i) 
                for i in concatenated_all_target_cluster_eval_gather_list_cat.tolist()]
            concatenated_all_regres_predictions_eval_gather_list_cat_out = concatenated_all_regres_predictions_eval_gather_list_cat.tolist()
            
            mse=log_metrics_eval(config,
                                concatenated_all_date_eval_gather_list_cat_out,
                                concatenated_all_location_eval_gather_list_cat_out,
                                concatenated_all_label_cla_eval_gather_list_cat_out,
                                concatenated_all_target_cluster_eval_gather_list_cat_out,
                                concatenated_all_regres_predictions_eval_gather_list_cat_out,
                                name,
                                global_step,
                                dataset_csv,
                                val_dataset,
                                status)
        dist.barrier()
        if rank == 0:
            return mse
        else:
            return None

def setup(rank,local_rank,world_size):

    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,timeout=datetime.timedelta(hours=4))
    torch.cuda.set_device(local_rank)
    set_seed(42 + rank) 

def cleanup():
    dist.destroy_process_group()

def train(config,start_epoch,model,rank,local_rank,world_size,train_loader,optimizer_model,epoch,global_step,criterion_regression,scheduler,sampler = None):
    model.train()
    epoch_loss = torch.zeros(1).to(local_rank)
    print('epoch_loss: ',epoch_loss)
   
    if sampler:
        sampler.set_epoch(epoch)

    if rank ==0 :
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{config['hyperparameters']['num_epochs'] + start_epoch }")
    else:
        progress_bar = enumerate(train_loader) 
    for step, batch in progress_bar:
        step_loss = torch.zeros(1).to(local_rank)
        global_step += 1
        
        optimizer_model.zero_grad()

        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        
        logits_regression,vae_out = model(batch)
        logits_regression = logits_regression.squeeze(-1)
        loss = criterion_regression.loss_comput(logits_regression, 
                                                batch["target_ratio_t1"].squeeze(-1),
                                                t1_total_isolate_mask=batch["target_ratio_t1_mask"].squeeze(-1),
                                                t1_label_weight = batch["t1_label_weight"].squeeze(-1),
                                                vae_out = vae_out)
        loss.backward()
        optimizer_model.step() 
        scheduler.step()

        step_loss = loss
           
        if rank == 0:
            all_step_loss_gather =  [torch.zeros_like(step_loss) for _ in range(dist.get_world_size())]
        else:
            all_step_loss_gather =  None
            
        dist.gather(step_loss, gather_list=all_step_loss_gather, dst=0)
        
        if rank == 0:
            all_step_loss_gather_total = sum(all_step_loss_gather)
            epoch_loss += all_step_loss_gather_total
            wandb.log({"Train_step_loss": (all_step_loss_gather_total / world_size),
                       "Avg_step_loss":(epoch_loss / world_size) / (step + 1), 
                       "step": global_step, 
                       "learning_rate": scheduler.get_last_lr()[0]}) 
            if global_step % config['other']['logging_steps'] == 0:        
                    print(f"Step {global_step} - Train_step_loss: {(all_step_loss_gather_total / world_size).item():.4f} - Avg_step_loss: {((epoch_loss / world_size) / (step + 1)).item():.4f} - LR: {scheduler.get_last_lr()[0]:.8f}")
    dist.barrier()
    if rank == 0:
        wandb.log({"avg_train_epoch_loss": ((epoch_loss / world_size) / (step + 1)), "epoch": epoch + 1})
        print(f"Epoch {epoch + 1} - Average Training Loss: {((epoch_loss / world_size) / (step + 1)).item():.4f}")
    
    return global_step

def creat_dataset(config,world_size,rank,local_rank):
    train_dataset = SARS2SpikeProteinDataset(args=args,config =config,ds="train")  
    Sampler_train = DistributedSampler(train_dataset,rank=local_rank,num_replicas=world_size,shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['hyperparameters']['batch_size'],sampler=Sampler_train)
    return train_loader,train_dataset,Sampler_train

def fsdp_config(rank): 
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            AxialTransformerLayer,
        },
    )

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD #for Zero2 and FULL_SHARD for Zero3
    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    bfSixteen = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    fp32_policy = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    mp_policy = None

    return auto_wrap_policy,sharding_strategy,mp_policy

class combine_loss(object): 
    def __init__(self,config):
        self.loss_func = torch.nn.MSELoss(reduction = 'none')
        self.config = config
    def loss_comput(self,y_pred,y_true,lambda_vae=1.0, lambda_reg=1.0,t1_total_isolate_mask = None,t1_label_weight=None,vae_out = None):
        p1 = self.loss_func(torch.log(y_pred * 100.0 + 1.0),torch.log(y_true * 100.0 + 1.0))
        element_weights = t1_total_isolate_mask * t1_label_weight
        row_sums = torch.sum(element_weights)
        normalized_element_weights = element_weights / (row_sums + 1e-8)
        part1_mse_regression = torch.sum(p1 * normalized_element_weights)
        return part1_mse_regression
    
    def sparse_loss(self,h1, sparsity_target=0.05, beta=1e-4):
        sparsity_penalty = sparsity_target - torch.mean(torch.abs(h1), dim=0)
        return beta * torch.sum(sparsity_penalty**2)

    def loss_function(self,x_reconstructed, x, mu, log_var, h1, sparsity_target=0.05, beta=1e-4):

        mse_reconstruction = nn.functional.mse_loss(x_reconstructed , x , reduction='sum')
        KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
        sparsity_penalty = self.sparse_loss(h1, sparsity_target, beta)
        
        return mse_reconstruction,KL_divergence,sparsity_penalty
        
    
def optimizer_scheduler(config,train_dataset,model,world_size):
    #==============================optimizer===============================
    criterion_regression = combine_loss(config)
    
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=config['hyperparameters']['learning_rate'], weight_decay=config['hyperparameters']["weight_decay"])
 
    #==============================scheduler===============================
    total_steps = (len(train_dataset) // world_size // config['hyperparameters']['batch_size']) * config['hyperparameters']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer_model,
        num_warmup_steps=config['hyperparameters']["warmup_steps"],
        num_training_steps=total_steps
    )
    return criterion_regression,optimizer_model,scheduler

def get_pt_filenames(folder_path):
    pt_filenames = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pt'):
                pt_filenames.append(file)
    
    return pt_filenames

def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank,local_rank,world_size)
    
    name = args.run_name

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if config['model']['target_only']:
        from model.model_proportion_ablation_nobackgrounds import proportion_single_t1 as Evo_prediction_proportion
    elif config['other']['ablation']:
        from model.model_proportion_include_ablation import proportion_single_t1 as Evo_prediction_proportion
    else:
        from model.model_proportion import proportion_single_t1 as Evo_prediction_proportion
    
    
    if rank == 0:
        folder_path = os.path.join(config['other']['output_dir'],name)
        os.makedirs(folder_path, exist_ok=True)

        if args.mode == 'run_train':
            with open(folder_path+"/config.yaml", "w") as yaml_file:
                yaml.dump(config, yaml_file,default_flow_style=False)
        print('**************config***************:\n',config)
    
    config["dataset_csv_path"] = os.path.join(config['dataset']['path']["dataset_csv_base_path"],config['dataset']['path']["test_split_time"],config['dataset']['path']['dataset_csv_name'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
    
        wandb.init(project="sars-cov-2 evo prediction proportion",name=name,mode="offline",config=config,resume=config['other']['wandb_resume'], id=config['other']['wandb_resume_id'])

        print(f" ues {device} training")

        early_stopping = EarlyStopping(config=config,verbose=True)
        

    #=============================Training and validation logic===========================
    if args.mode == 'run_train': 
        auto_wrap_policy,sharding_strategy,mp_policy = fsdp_config(rank)
        model = Evo_prediction_proportion(config).to(local_rank)

        model = FSDP(model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=sharding_strategy,
            device_id=local_rank,
            use_orig_params=True)
        train_loader,train_dataset,Sampler_train = creat_dataset(config,world_size,rank,local_rank)
        print(train_dataset)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")

        criterion_regression,optimizer_model,scheduler = optimizer_scheduler(config,train_dataset,model,world_size)
        start_epoch = 0
        global_step = 0 

        
        if rank == 0:
            wandb.watch(model, log="all")

        for epoch in range(start_epoch,config['hyperparameters']['num_epochs']+start_epoch):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{config['hyperparameters']['num_epochs']+start_epoch}")
            global_step = train(config,
                                start_epoch,  
                                model,
                                rank,
                                local_rank,
                                world_size,
                                train_loader,
                                optimizer_model,
                                epoch,
                                global_step,
                                criterion_regression,
                                scheduler,
                                sampler = Sampler_train)   
            #===============================Validation in training=====================================
            dataset_csv = pd.read_csv(config["dataset_csv_path"])
            val_dataset = SARS2SpikeProteinDataset(args=args,
                                                    config =config,
                                                    ds="validate",
                                                    )
            Sampler_val = DistributedSampler(val_dataset,rank=local_rank,num_replicas=world_size,drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'],drop_last=True,sampler=Sampler_val)
            validation_mse = validation(config,model,rank,local_rank,world_size,val_loader,val_dataset,name,criterion_regression,global_step,dataset_csv,status = "validation")

            if config["other"]["save_model_train"]:
                dist.barrier()
                states = model.state_dict()
            else :
                states = None   
            dist.barrier() 

            if rank == 0:
                if early_stopping(validation_mse,states, epoch,scheduler,optimizer_model,name,global_step):
                    print("Stopping early!")
                    break
            
            dist.barrier()
            global_step = global_step
    #===============================evaluate=====================================
    elif args.mode == 'run_validation':
        if config['dataset']['path']['use_to_evaluate_checkpoint_path'] is not None:
            model = Evo_prediction_proportion(config).to(local_rank)
            model_state = torch.load(config['dataset']['path']['use_to_evaluate_checkpoint_path'], map_location=device)
            model.load_state_dict(model_state['model_state_dict'])
            model= DDP(model, device_ids=[local_rank])
            global_step = model_state['global_step']
            mseloss = torch.nn.MSELoss()

            dataset_csv = pd.read_csv(config["dataset_csv_path"])
            val_dataset = SARS2SpikeProteinDataset(args=args,
                                                        config =config
                                                        )
            Sampler_val = DistributedSampler(val_dataset,rank=local_rank,num_replicas=world_size,drop_last=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'],drop_last=True,sampler=Sampler_val,num_workers=0)
            mse = validation(config,model,rank,local_rank,world_size,val_loader,val_dataset,name,mseloss,global_step,dataset_csv,status = f"Epoch{model_state['epoch']}_validation_{config['dataset']['path']['dataset_csv_name'].split('.')[0]}")

        else:
            checkpoint_path = os.path.join(config['other']['output_dir'],args.run_name)
            pt_files = get_pt_filenames(checkpoint_path)
            if rank == 0:
                print('checkpoint to eval:',pt_files)
            for i in pt_files:
                use_to_evaluate_checkpoint_path = os.path.join(checkpoint_path,i)
                model = Evo_prediction_proportion(config).to(local_rank)
                model_state = torch.load(use_to_evaluate_checkpoint_path, map_location=device)
                model.load_state_dict(model_state['model_state_dict'])
                model= DDP(model, device_ids=[local_rank])
                global_step = model_state['global_step']
                mseloss = torch.nn.MSELoss()

                dataset_csv = pd.read_csv(config["dataset_csv_path"])
                val_dataset = SARS2SpikeProteinDataset(args=args,
                                                            config =config
                                                            )
                Sampler_val = DistributedSampler(val_dataset,rank=local_rank,num_replicas=world_size,drop_last=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['hyperparameters']['batch_size'],drop_last=True,sampler=Sampler_val)
                mse = validation(config,model,rank,local_rank,world_size,val_loader,val_dataset,name,mseloss,global_step,dataset_csv,status = f"Epoch{model_state['epoch']}_validation_{config['dataset']['path']['dataset_csv_name'].split('.')[0]}")
                
    dist.barrier()  
    if rank == 0:
        wandb.finish()
    dist.barrier()
    cleanup()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sars-cov-2 evolution prediction proportion single-point')
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--mode',  type=str,default ='run_train')
    parser.add_argument('--config',  type=str,default ='/lustre/grp/cyllab/luoxw/SARS_CoV_2_evolution_prediction/DMS_LLM/evopred/config/train_config.yaml')
    parser.add_argument('--code_testing', type=bool,default =False)
    args = parser.parse_args()
    

    main(args)
    