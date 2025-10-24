library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(future.apply)
library(stringr)

# Set file directory
file_dir = '<path/to/ablation_dir>'
args <- commandArgs(trailingOnly = TRUE)
n_truth_top <- as.numeric(args[1])
print('n_truth_top:')
print(n_truth_top)


rbd_name_mapper <- fromJSON(str_glue('{file_dir}/data/processed/to241030/rbd_name_mapper.json'))
meta <- read_csv(str_glue('{file_dir}/data/processed/to241030/meta241030.csv.gz'))

# Read model results 
ours <- read_csv(str_glue('{file_dir}/results/keep_all_rep1/TestFull_regres_outputs_labels_keep_all_rep1.csv')) %>% 
    mutate(model = 'full model') %>% 
    mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

test_2 <- read_csv(str_glue('{file_dir}/results/no_dms_all_rep3/TestFull_regres_outputs_labels_no_dms_all_rep1.csv')) %>% 
    mutate(model = 'no DMS') %>% 
    mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

test_3 <- read_csv(str_glue('{file_dir}/results/ESM2_rep1/TestFull_regres_outputs_labels_ESM2_rep1.csv')) %>% 
    mutate(model = 'ESM-2 in place of ESM-MSA-1b') %>% 
    mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

test_4 <- read_csv(str_glue('{file_dir}/results/no_backgrounds_rep2/TestFull_regres_outputs_labels_no_backgrounds_rep1.csv')) %>% 
    mutate(model = 'no backgrounds') %>% 
    mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

meta_test = meta %>% filter(rbd_name %in% unique(ours$rbd_name),submit_date > ymd('2023-10-01'),submit_date < ymd('2024-09-01')) 

options(future.globals.maxSize = 2 * 1024^3) 
plan(multisession, workers = 64)

jaccard_index <- function(set1, set2) {
  inter = length(intersect(set1, set2))
  union = length(union(set1, set2))
  if (union == 0) return(0)
  inter / union
}

unique_dates <- sort(unique(ours %>% pull(t0)))
all_rank_eval_results <- list()
print('start')
n_pred_top_max=20
for(n_pred_top in c(n_truth_top:n_pred_top_max)){
    print(n_pred_top)
    rank_eval_results <- future_lapply(seq_along(unique_dates), function(i) {
   
    window_start <- unique_dates[i]-30; window_end <- unique_dates[i]
    rbd_set = meta_test %>% filter(submit_date<=window_end,submit_date>=window_start) %>% pull(rbd_name_mut) %>% unique
    truth_set = ours %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_label) %>% slice(1:n_truth_top) %>% pull(rbd_name_mut)
    if (length(truth_set) == 0) {
        return(NULL)}
    ours_set =  ours %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)
    test_2_set =  test_2 %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)
    test_3_set =  test_3 %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)
    test_4_set =  test_4 %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)

    if(n_truth_top ==1){
        evaluation = data.frame(date=window_start,n_pred_top=n_pred_top,n_truth_top=n_truth_top,
                                `full model`=(truth_set %in% ours_set),`no DMS`=(truth_set %in% test_2_set),`esm2`=(truth_set %in% test_3_set),`no backgrounds`=(truth_set %in% test_4_set))
    }
    else{
        evaluation = data.frame(date=window_start,n_pred_top=n_pred_top,n_truth_top=n_truth_top,
                               `full model`=jaccard_index(truth_set,ours_set),`no DMS`=jaccard_index(truth_set,test_2_set),`esm2`=jaccard_index(truth_set,test_3_set),`no backgrounds`=jaccard_index(truth_set,test_4_set))
    }
    return(evaluation)
    })
    all_rank_eval_results[[as.character(n_pred_top)]] <- bind_rows(rank_eval_results)
}
rank_eval_results <- bind_rows(all_rank_eval_results)
rank_eval_results %>% write_csv(str_glue('{file_dir}/analysis/ablation_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv'))
print('save file')
summary_success = rank_eval_results %>% group_by(n_pred_top,n_truth_top) %>% summarize(`full model` = mean(`full model`),`no DMS`=mean(`no DMS`),`esm2`=mean(`esm`),`no backgrounds`=mean(`no backgrounds`)) %>% 
    pivot_longer(-c('n_pred_top','n_truth_top'),names_to='methods',values_to='success_rate')

p = summary_success %>% ggplot(aes(x= n_pred_top,y=success_rate,color=methods)) + geom_point() + geom_line() + 
    theme_classic() + scale_color_locuszoom() + labs(x='k',y = 'prediction success rate')

pdf(str_glue('{file_dir}/plots/success_rate_predtop{n_pred_top_max}_truthtop{n_truth_top}_ablation.pdf'), width=4, height=3)
p
dev.off()
