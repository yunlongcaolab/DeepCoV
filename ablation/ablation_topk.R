library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)
library(stringr)

data_dir = '<path/to/dir>'

#######################
### static topk
#######################
rbd_name_mapper <- fromJSON(str_glue("{data_dir}/ablation/data/processed/to241030/rbd_name_mapper.json"))

test_1 <- read_csv(str_glue('{data_dir}/ablation/results/keep_all_rep1/TestFull_regres_outputs_labels_keep_all_rep1.csv')) %>% 
  mutate(model = 'full model')

test_2 <- read_csv(str_glue('{data_dir}/ablation/results/no_dms_all_rep1/TestFull_regres_outputs_labels_no_dms_all_rep1.csv')) %>% 
  mutate(model = 'no DMS')

test_3 <- read_csv(str_glue('{data_dir}/ablation/results/ESM2_rep1/TestFull_regres_outputs_labels_ESM2_rep1.csv')) %>% 
  mutate(model = 'ESM-2 in place of ESM-MSA-1b')

test_4 <- read_csv(str_glue('{data_dir}/ablation/results/no_backgrounds_rep1/TestFull_regres_outputs_labels_no_backgrounds_rep1.csv')) %>% 
  mutate(model = 'no backgrounds')

test_data <- bind_rows(test_1, test_2, test_3, test_4)
test_max <- test_data %>% group_by(rbd_name,model) %>%
  summarise(
    target_ratio_t1_output = max(target_ratio_t1_output, na.rm = TRUE),
  ) %>% ungroup() %>%  mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

truth_max = test_1 %>% group_by(rbd_name,model) %>%
  summarise(
    # target_ratio_t1_output = max(target_ratio_t1_output, na.rm = TRUE),
    target_ratio_t1_label = max(target_ratio_t1_label, na.rm = TRUE)
  ) %>% ungroup() %>%  mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

score_summary = test_max %>% pivot_wider(names_from=model,values_from=target_ratio_t1_output) 
# score_summary = score_summary %>% filter(rbd_name_mut %>% str_detect("^HK\\.3|BA\\.2\\.86|JN\\.1|KP\\.2|KP\\.3") == TRUE)

# n_major=3
n_major=10
positive_case = truth_max %>% arrange(-target_ratio_t1_label) %>% slice(1:n_major) %>% pull(rbd_name_mut)
methods=c('full model', 'no DMS', 'no backgrounds', 'ESM-2 in place of ESM-MSA-1b')

result_metrics <- list()

for (method in methods) {
    sorted_df <- score_summary %>% arrange(desc(!!sym(method)))
    total_positive <- sum(score_summary$rbd_name_mut %in% positive_case)
    total_negative <- nrow(score_summary) - total_positive
    
    for (k in 1:nrow(score_summary)) {
        top_k <- head(sorted_df, k)

        TP <- sum(top_k$rbd_name_mut %in% positive_case)  
        FP <- k - TP                                      
        FN <- total_positive - TP                         
        TN <- total_negative - FP                         
        
        recall <- TP / (TP + FN)          
        accuracy <- (TP + TN) / (TP + FP + TN + FN) 
        precision <- TP / ( TP + FP )
        FDR <- FP / (TP + FP) 
        TPR <- recall
        FPR <- FP / (FP + TN)  
        
        result_metrics[[length(result_metrics) + 1]] <- data.frame(method = method,k = k,
                                                                   recall = recall,accuracy = accuracy,precision = precision,
                                                                   FDR=FDR,TPR=TPR,FPR=FPR)
}}

result_metrics = bind_rows(result_metrics) %>% mutate(method = factor(method,levels = methods))
result_metrics %>% write_csv(str_glue('{data_dir}/ablation/analysis/Topk_ablation_result_metrics_{n_major}major_fullset.csv'))

theme_fontsize =  theme(axis.text=element_text(size=15),axis.title=element_text(size=17),
        legend.text = element_text(size = 7),legend.title = element_text(size = 7),legend.position = "bottom",
        plot.title = element_text(size = 18, face = "plain", hjust = 0.5)) 

p0 = ggplot(result_metrics, aes(x = k, y = precision, color = method)) +
  geom_line(size=0.7,alpha=0.7) + 
  geom_point(size = 0.8, fill = "white") +
  scale_y_continuous(limits = c(0, 1.05),expand = expansion()) + 
  # scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "Precision", x= 'k') +
  scale_color_nejm() + 
  # theme_minimal() + 
  theme_cowplot() + 
  theme_fontsize

p1 = ggplot(result_metrics, aes(x = k, y = recall, color = method)) +
  geom_line(size=0.7,alpha=0.7) + 
  geom_point(size = 0.8, fill = "white") + #shape = 21, 
  scale_y_continuous(limits = c(0, 1.05),expand = expansion())  + 
  # scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "Recall", x= 'k') +
  scale_color_nejm() + 
  # theme_minimal() + 
  theme_cowplot() + 
  theme_fontsize

p2 = ggplot(result_metrics, aes(x = k, y = FDR, color = method)) +
  geom_line(size=0.7,alpha=0.7) + 
  geom_point(size = 0.8, fill = "white") + #shape = 21, 
  scale_y_continuous(limits = c(0, 1.05),expand = expansion())  +
  # scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "False Discovery Rate", x= 'k') +
  scale_color_nejm() + 
  theme_cowplot() + 
  theme_fontsize

pdf(str_glue('{data_dir}/ablation/plot/Topk_ablation_metrics_curves_{n_major}major_fullset.pdf'),width=4.5,height = 3)
p0
p1
p2
dev.off()
