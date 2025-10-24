library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)
library(ggvenn)
# 
data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'

rbd_name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"))
meta <- read_csv(str_glue("{data_dir}/data/processed/to241030/meta241030.csv.gz"))

ours = read_csv(str_glue("{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv")) %>% 
    mutate(t1_date = t0 + t1) %>% mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)
evescape = read_csv(str_glue("{data_dir}/benchmark/EVEscape/EVEscape_scores_test_JN1era.csv"))
e2vd = read_csv(str_glue("{data_dir}/benchmark/E2VD/E2VD_scores_test_JN1era.csv"))

meta_test = meta %>% filter(rbd_name %in% unique(ours$rbd_name),submit_date > ymd('2023-10-01'),submit_date < ymd('2024-09-01')) 
evescape_score_mapper = evescape %>% select(rbd_name_mut,`EVEscape score_pos`) %>% deframe() 
e2vd_score_mapper = e2vd %>% select(rbd_name_mut,E2VD) %>% deframe() 

unique_dates <- sort(unique(ours %>% pull(t0)))

# k to k venn
data = read_csv(str_glue('{data_dir}/benchmark/analysis/res_success_predtop20_truthtop20.csv'))

# i = 140; n_truth_top = 3; n_pred_top = 3
i = 160; n_truth_top = 5; n_pred_top = 5
print(unique_dates[i])
    
window_start <- unique_dates[i]-30; window_end <- unique_dates[i]
rbd_set = meta_test %>% filter(submit_date<=window_end,submit_date>=window_start) %>% pull(rbd_name_mut) %>% unique
truth_set = ours %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_label) %>% slice(1:n_truth_top) %>% pull(rbd_name_mut)
if (length(truth_set) == 0) {eturn(NULL)}
print('truth_set: ');print(truth_set)
ours_set =  ours %>% filter(t0 == unique_dates[i],rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_output) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut)
print('ours_set: ');print(ours_set)
EVEscape_set = names(sort(evescape_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top]
print('EVEscape_set: ');print(EVEscape_set)
E2VD_set = names(sort(e2vd_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top]
print('E2VD_set: ');print(E2VD_set)

p1 = ggvenn(
  sets <- list(truth = truth_set,ours = ours_set),
  # show_elements = TRUE,label_sep = "\n",fill_color = c('#88A69D','#E7BA8F'),
  show_elements = TRUE,label_sep = "\n",fill_color = c('#BA91A9','#468CBC'),
  stroke_size = 0.5,set_name_size = 6,text_size = 5.5
) +labs(title = str_glue("Top {n_pred_top} Predictions vs Top {n_truth_top} Truth")) + theme(plot.title = element_text(hjust = 0.5,size=15)) 
p2 = ggvenn(
  sets <- list(truth = truth_set,EVEscape = EVEscape_set),
  # show_elements = TRUE,label_sep = "\n",fill_color = c('#88A69D','#E7BA8F'),
  show_elements = TRUE,label_sep = "\n",fill_color = c('#BA91A9','#468CBC'),
  stroke_size = 0.5,set_name_size = 6,text_size = 5.5
) +labs(title = str_glue("Top {n_pred_top} Predictions vs Top {n_truth_top} Truth")) + theme(plot.title = element_text(hjust = 0.5,size=15)) 
p3 = ggvenn(
  sets <- list(truth = truth_set,E2VD = E2VD_set),
  # show_elements = TRUE,label_sep = "\n",fill_color = c('#88A69D','#E7BA8F'),
  show_elements = TRUE,label_sep = "\n",fill_color = c('#BA91A9','#468CBC'),
  stroke_size = 0.5,set_name_size = 6,text_size = 5.5
) +labs(title = str_glue("Top {n_pred_top} Predictions vs Top {n_truth_top} Truth")) + theme(plot.title = element_text(hjust = 0.5,size=15)) 

pdf(str_glue('{data_dir}/benchmark/plots/dynamic_topk_venn_{unique_dates[i]}_predtop{n_pred_top}_truthtop{n_truth_top}.pdf'),width=5,height=3)
p1
p2
p3
dev.off()