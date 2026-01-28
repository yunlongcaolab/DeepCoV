library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)
library(ggvenn)
# 
data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'
plot_dir=str_glue('{data_dir}/benchmark/plots')

rbd_name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"))
meta <- read_csv(str_glue("{data_dir}/data/processed/to241030/meta241030.csv.gz"))

ours <- read_csv(str_glue("{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv")) %>% 
    # mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)
mlr <- read_csv(str_glue('{data_dir}/benchmark/MLR/results/TestFull_2023-10-01_to241030_MLR_pred30days.csv')) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)
evescape = read_csv(str_glue("{data_dir}/benchmark/EVEscape/EVEscape_scores_test_JN1era.csv"))
e2vd <- read_csv(str_glue("{data_dir}/benchmark/E2VD/E2VD_scores_test_JN1era.csv"))

meta_test <- meta %>% filter(rbd_name %in% unique(ours$rbd_name),submit_date > ymd('2023-10-01'),submit_date < ymd('2024-09-01')) 
evescape_score_mapper <- evescape %>% select(rbd_name_mut,`EVEscape score_pos`) %>% deframe() 
e2vd_score_mapper <- e2vd %>% select(rbd_name_mut,E2VD) %>% deframe() 

unique_dates <- sort(unique(ours %>% pull(t0)))

plot_venn_comparison <- function(target_date, n_truth_top = 5, n_pred_top = 5) {
    target_date <- as.Date(target_date)
    message(str_glue("Processing date: {target_date}"))
    
    window_start <- target_date - days(30)
    window_end <- target_date
    
    # 获取该时间窗口内的候选变异株集
    rbd_set <- meta_test %>% 
        filter(submit_date <= window_end, submit_date >= window_start) %>% 
        pull(rbd_name_mut) %>% unique()
    
    if (length(rbd_set) == 0) return(NULL)
    
    rbd_set = meta_test %>% filter(submit_date<=window_end,submit_date>=window_start) %>% filter(rbd_name_mut %>% str_detect("^HK\\.3|BA\\.2\\.86|JN\\.1|KP\\.2|KP\\.3|XBB\\.1\\.5|EG\\.5") == TRUE)%>% pull(rbd_name_mut) %>% unique()
    truth_set <- ours %>% filter(t0 == target_date,rbd_name_mut %in% rbd_set) %>% arrange(-target_ratio_t1_label) %>% slice(1:n_truth_top) %>% pull(rbd_name_mut);print('truth_set: ');print(truth_set)
    if (length(truth_set) == 0) {return(NULL)}
    ours_set <-  ours %>% filter(t0 == target_date,rbd_name_mut %in% rbd_set) %>% arrange(desc(target_ratio_t1_output)) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut);print('ours_set: ');print(ours_set)
    mlr_set <-  mlr %>% filter(t0 == target_date,rbd_name_mut %in% rbd_set) %>% arrange(desc(median_freq_forecast)) %>% slice(1:n_pred_top) %>% pull(rbd_name_mut);print('mlr_set: ');print(mlr_set)
    EVEscape_set <- names(sort(evescape_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top];print('EVEscape_set: ');print(EVEscape_set)
    E2VD_set <- names(sort(e2vd_score_mapper[rbd_set] , decreasing = TRUE))[1:n_pred_top];print('E2VD_set: ');print(E2VD_set)

    # 通用绘图参数
    venn_colors <- c('#BA91A9', '#468CBC')
    
    # 绘图逻辑封装
    make_v_plot <- function(pred_set, name) {
      sets <- list(Truth = truth_set);sets[[name]] <- pred_set
      ggvenn(
            sets,show_elements = TRUE, label_sep = "\n",
            fill_color = venn_colors, stroke_size = 0.5, 
            set_name_size = 5, text_size = 4
        ) + labs(title = str_glue("{name} vs Truth")) +
        theme(plot.title = element_text(hjust = 0.5, size = 12))
    }

    p1 <- make_v_plot(ours_set, "DeepCoV")
    p2 <- make_v_plot(EVEscape_set, "EVEscape")
    p3 <- make_v_plot(E2VD_set, "E2VD")
    p4 <- make_v_plot(mlr_set, "MLR")

    
    date_predict = target_date + days(30)
    pdf(str_glue('{plot_dir}/dynamic_topk_venn_{date_predict}_predtop{n_pred_top}_truthtop{n_truth_top}.pdf'),width=5,height=3)
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    dev.off()

}

plot_venn_comparison('2024-02-05',3,3)
plot_venn_comparison('2024-02-17',3,3)
plot_venn_comparison('2024-05-01',3,3)

plot_venn_comparison('2024-02-26',5,5)
plot_venn_comparison('2024-03-15',5,5)
plot_venn_comparison('2024-03-28',5,5)
