library(tidyverse)
library(ggsci)
library(cowplot)
theme_set(theme_cowplot())

data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'
analysis_data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/analysis'
plot_save_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/plots'

ours = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv'))
evescape = read_csv(str_glue('{data_dir}/benchmark/EVEscape/EVEscape_scores_test_JN1era.csv'))
dms_fitting_score = read_csv(str_glue('{data_dir}/benchmark/DMS/fitting/dms_fittings_score_test_JN1era.csv')) 
e2vd = read_csv(str_glue('{data_dir}/benchmark/E2VD/E2VD_scores_test_JN1era.csv'))
ga = read_csv(str_glue('{data_dir}/benchmark/growth_advantage/rbd_test_JN1era_ga.csv'))

ours_strain_max <- ours %>% group_by(rbd_name) %>%
  summarise(
    target_ratio_t1_output = max(target_ratio_t1_output, na.rm = TRUE),
    target_ratio_t1_label = max(target_ratio_t1_label, na.rm = TRUE)
  ) %>% ungroup()

ga_strain_max = ga %>% mutate(ga_cov = as.numeric(ga_cov))%>% filter(ga_cov < 1) %>% group_by(rbd_name) %>% summarise(ga = max(ga_cov, na.rm = TRUE)) %>% ungroup()

score_summary = ours_strain_max %>% full_join(ga_strain_max) %>% 
    full_join(evescape %>% select(rbd_name,rbd_name_mut,EVEscape=`EVEscape score_pos`)) %>% 
    full_join(dms_fitting_score %>% select(rbd_name,rbd_name_mut,dms_fitting_score = Q_e_JN1ref)) %>%
    full_join(e2vd %>% select(rbd_name,rbd_name_mut,E2VD)) %>%
    full_join(ga_strain_max %>% select(rbd_name,ga)) %>% relocate(rbd_name_mut, .after = 'rbd_name')  %>% mutate_all(~replace(., is.na(.), 0))

score_summary %>% write_csv(str_glue('{analysis_data_dir}/benchmark_scores_summary_static.csv')) 
score_summary = score_summary %>% rename(ours=target_ratio_t1_output,`growth advantage`=ga,`DMS(fit)`=dms_fitting_score# ,`DMS(sum)`=dms_sum_score
) %>% filter(rbd_name_mut %>% str_detect("^HK\\.3|BA\\.2\\.86|JN\\.1|KP\\.2|KP\\.3") == TRUE)

# positive_case = c('JN.1','KP.2','KP.3')
n_major=3
# n_major=10
positive_case = score_summary %>% arrange(-target_ratio_t1_label) %>% slice(1:n_major) %>% pull(rbd_name_mut)
methods = c("ours","EVEscape","E2VD","growth advantage","DMS(fit)") 

result_metrics <- list()

# 计算每个方法的 top k 指标
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
        
        # 计算指标
        recall <- TP / (TP + FN)          
        accuracy <- (TP + TN) / (TP + FP + TN + FN) 
        precision <- TP / ( TP + FP )
        FDR <- FP / (TP + FP) # 1 - Precision
        TPR <- recall
        FPR <- FP / (FP + TN)  
        
        result_metrics[[length(result_metrics) + 1]] <- data.frame(method = method,k = k,
                                                                   recall = recall,accuracy = accuracy,precision = precision,
                                                                   FDR=FDR,TPR=TPR,FPR=FPR)
}}

result_metrics = bind_rows(result_metrics) %>% mutate(method = factor(method,levels = methods))

theme_fontsize =  theme(axis.text=element_text(size=15),axis.title=element_text(size=17),
        legend.text = element_text(size = 15),legend.title = element_text(size = 15),legend.position = "bottom",
        plot.title = element_text(size = 18, face = "plain", hjust = 0.5)) 


p1 = ggplot(result_metrics, aes(x = k, y = recall, color = method)) +
  geom_line(size=0.7,alpha=0.7) + # geom_point() + 
  geom_point(size = 0.8, fill = "white") + #shape = 21, 
  scale_y_continuous(limits = c(0, 1.05),expand = expansion())  + 
  scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  # scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "Recall", x= 'k') +
  scale_color_nejm() + 
  # theme_minimal() + 
  theme_cowplot() + 
  theme_fontsize

p2 = ggplot(result_metrics, aes(x = k, y = FDR, color = method)) +
  geom_line(size=0.7,alpha=0.7) + # geom_point() + 
  geom_point(size = 0.8, fill = "white") + #shape = 21, 
  scale_y_continuous(limits = c(0, 1.05),expand = expansion())  +
  scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  # scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "False Discovery Rate", x= 'k') +
  scale_color_nejm() + 
  theme_cowplot() + 
  theme_fontsize

p3 = ggplot(result_metrics, aes(x = k, y = accuracy, color = method)) +
  geom_line(size=0.7,alpha=0.7) + # geom_point() + 
  geom_point(size = 0.8, fill = "white") + #shape = 21, 
  scale_y_continuous(limits = c(0, 1.05 ),expand = expansion())  + 
  scale_x_continuous(breaks=seq(0,100,20),limits=c(1,110),expand = expansion()) +
  # scale_x_continuous(breaks=seq(0,50,10),limits=c(1,60),expand = expansion()) +
  labs(title = "JN.1 era prediction", y = "Accuracy", x= 'k') +
  scale_color_nejm() + 
  theme_cowplot() +
  theme_fontsize 


pdf(str_glue('{plot_save_dir}/Topk_metrics_curves_{n_major}major.pdf'), width=6, height=3.3)
p1
p2
p3
dev.off()

