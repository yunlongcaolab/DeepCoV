library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)
library(future.apply)
args <- commandArgs(trailingOnly = TRUE)
n_truth_top <- as.numeric(args[1])
print('n_truth_top:')
print(n_truth_top)

data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'

rbd_name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"))
meta <- read_csv(str_glue("{data_dir}/data/processed/to241030/meta241030.csv.gz"))

ours = read_csv(str_glue("{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv")) %>% 
    mutate(t1_date = t0 + t1) %>% mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)
evescape = read_csv(str_glue("{data_dir}/benchmark/EVEscape/EVEscape_scores_test_JN1era.csv"))
e2vd = read_csv(str_glue("{data_dir}/benchmark/E2VD/E2VD_scores_test_JN1era.csv"))
ga = read_csv(str_glue('{data_dir}/benchmark/growth_advantage/rbd_test_JN1era_ga.csv')) %>%
    mutate(ga = as.numeric(ga_cov)) %>%
    filter(ga < 1) %>%
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

meta_test = meta %>% filter(rbd_name %in% unique(ours$rbd_name),submit_date > ymd('2023-10-01'),submit_date < ymd('2024-09-01')) 
evescape_score_mapper = evescape %>% select(rbd_name_mut,`EVEscape score_pos`) %>% deframe() 
e2vd_score_mapper = e2vd %>% select(rbd_name_mut,E2VD) %>% deframe() 

plan(multisession)

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
    if (length(truth_set) == 0) {return(NULL)}
    ours_set =  ours %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)
    EVEScape_set = names(sort(evescape_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top]
    E2VD_set = names(sort(e2vd_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top]
    ga_set =  ga %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-ga) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)

    if(n_truth_top ==1){
        evaluation = data.frame(date=window_start,n_pred_top=n_pred_top,n_truth_top=n_truth_top,
                                EVEScape=(truth_set %in% EVEScape_set),E2VD=(truth_set %in% E2VD_set),ours=(truth_set %in% ours_set),ga = (truth_set %in% ga_set))
    }
    else{
        evaluation = data.frame(date=window_start,n_pred_top=n_pred_top,n_truth_top=n_truth_top,
                                EVEScape=jaccard_index(truth_set,EVEScape_set),E2VD=jaccard_index(truth_set,E2VD_set),ours=jaccard_index(truth_set,ours_set),ga = jaccard_index(truth_set,ga_set))
    }
    return(evaluation)
    })
    all_rank_eval_results[[as.character(n_pred_top)]] <- bind_rows(rank_eval_results)
}

rank_eval_results <- bind_rows(all_rank_eval_results)

rank_eval_results %>% write_csv(str_glue('{data_dir}/benchmark/analysis/res_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv'))
print('saved file')
