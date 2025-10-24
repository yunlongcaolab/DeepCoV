library(tidyverse)
library(ggsci)
library(cowplot)
library(ggrepel)
library(stringr)

# Base directory
base_dir <- '/lustre/grp/cyllab/share/evolution_prediction_dl'



## XBB era
file_dir <- str_glue('{base_dir}/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/XBBera_Thres')
save_file <- str_glue('{base_dir}/generalization/XBB_era/plots/XBBera_evaluation_dynamic_prop_thres.pdf')
model <- 'XBB era'
prop_max <- 0.15

## JN1 spike 
# file_dir <- str_glue('{base_dir}/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_spike_Thres')
# save_file <- str_glue('{base_dir}/generalization/spike/plots/JN1_era_spike_evaluation_dynamic_prop_thres.pdf')
# model <- 'JN.1 era spike'
# prop_max <- 0.275

## JN1 
# file_dir <- str_glue('{base_dir}/spatiotemporal_evaluation/dynamic_assessment_under_variable_prevalence_threshold/results/JN1era_Thres')
# save_file <- str_glue('{base_dir}/spatiotemporal_evaluation/plots/JN1_era_evaluation_dynamic_prop_thres.pdf')
# model <- 'JN.1 era'
# prop_max <- 0.375

thresholds <- seq(0.05, prop_max, by = 0.025) %>% round(3)


result_metrics <- list()
for(i in thresholds){
    # Construct input file path using glue
    input_file <- str_glue('{file_dir}/S3prop_{i}_S2ga_0.3.csv')
    if (!file.exists(input_file)) {
      stop(glue("Input file not found: {input_file}"))
    }
    dat <- read_csv(input_file, show_col_types = F)
    print(i)
    if(model=='XBB era'){dat = dat %>% filter(rbd_name_mut %>% str_detect("^XBB|EG\\.5|HK\\.3|BQ\\.1") == TRUE)}
    print(dat %>% filter(S3 == "TRUE") %>% pull(name))
    for(method in c('growth advantage','ours')){
        if(method == 'ours'){
            # conf_matrix <- table(dat$S3_pred, dat$S3)
            conf_matrix <- table(factor(dat$S3_pred, levels = c("FALSE", "TRUE")),factor(dat$S3,levels = c("FALSE", "TRUE")))

        }
        if(method == 'growth advantage'){
            # conf_matrix <- table(dat$S2, dat$S3)
            conf_matrix <- table(factor(dat$S2, levels = c("FALSE", "TRUE")),factor(dat$S3,levels = c("FALSE", "TRUE")))
        }
        TP <- conf_matrix['TRUE', 'TRUE']
        FP <- conf_matrix['TRUE', 'FALSE']
        TN <- conf_matrix['FALSE', 'FALSE']
        FN <- conf_matrix['FALSE', 'TRUE']

        recall <- TP / (TP + FN)
        accuracy <- (TP + TN) / (TP + FP + TN + FN)
        if(TP + FP > 0){
            # precision <- TP / (TP + FP)
            FDR <- FP / (TP + FP) 
        }else{
            # precision <- NA
            FDR <- NA
        }

        result_metrics[[length(result_metrics) + 1]] <- data.frame(method = method,proportion_threshold = i,
                                                                   recall = recall,accuracy = accuracy,FDR=FDR,
                                                                   n_actual_positive = TP + FN) # ,precision = precision
    }
}
result_metrics = bind_rows(result_metrics)

result_metrics_long = result_metrics %>%pivot_longer(cols = -c(method, proportion_threshold,n_actual_positive),names_to = "evaluation metrics",values_to = "score") %>% 
    mutate(n_actual_positive_label = ifelse(`evaluation metrics` == 'FDR',n_actual_positive,NA))

p = ggplot(result_metrics_long %>% filter(method=='ours'), aes(x = proportion_threshold, y = score, color = `evaluation metrics`)) +
  geom_line() + 
  geom_point(size = 1.5, fill = "white") + # shape = 21, 
  geom_label_repel(aes(label=n_actual_positive_label),size=2.7,label.size = 0.25,min.segment.length = 0.1,
                    arrow = grid::arrow(length = unit(0.02, "npc")),segment.size = 0.3
                    ) + 
  # geom_label_repel(aes(label=n_actual_positive),size=1) + 
  scale_y_continuous(limits = c(0, 1)) + 
  # scale_x_continuous(breaks=seq(0.05,prop_max,0.05),limits=c(0.04,prop_max+0.01))+
  scale_x_reverse(breaks=rev(seq(0.05,prop_max,0.05)),limits=rev(c(0.04,prop_max+0.01))) + 
  labs(title = str_c(model,' prediction'),
       y = "evaluation metrics", x= 'proportion threshold at T3') +
  # scale_color_locuszoom() 
  scale_color_manual(values=c("#BC3C29FF","#E18727FF", "#0072B5FF"))+ theme_classic() + # theme_light() + # + theme_light() +# + theme_minimal() + # theme_cowplot() +
  theme(legend.position = "bottom",
        axis.text=element_text(size=9),
        legend.text = element_text(size = 10),
        plot.title = element_text(size = 13, face = "plain", hjust = 0.5))

# Save the plot
pdf(save_file, width = 3, height = 3)
p
dev.off()