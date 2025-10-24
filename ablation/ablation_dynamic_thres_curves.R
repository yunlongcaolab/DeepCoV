library(tidyverse)
library(ggsci)
library(cowplot)
library(jsonlite)

data_dir = '<path/to/dir>'
prop_max = 0.5

thresholds <- seq(0.05, prop_max, by = 0.025) %>% round(3)

result_metrics <- list()
for(model in c('all_keep','esm2','no_backgrounds','no_dms_all')){
    print(model)
    for(i in thresholds){
        dat <- read_csv(str_c(data_dir,'/ablation/analysis/',model, '_Thres/S3prop_', i, '_S2ga_0.3.csv'),show_col_types = F)
        print(i)
        conf_matrix <- table(factor(dat$S3_pred, levels = c("FALSE", "TRUE")),factor(dat$S3,levels = c("FALSE", "TRUE")))

        TP <- conf_matrix['TRUE', 'TRUE']
        FP <- conf_matrix['TRUE', 'FALSE']
        TN <- conf_matrix['FALSE', 'FALSE']
        FN <- conf_matrix['FALSE', 'TRUE']

        recall <- TP / (TP + FN)
        accuracy <- (TP + TN) / (TP + FP + TN + FN)
        if(TP + FP > 0){
            FDR <- FP / (TP + FP) 
        }else{
            FDR <- NA
        }

        result_metrics[[length(result_metrics) + 1]] <- data.frame(model = model,proportion_threshold = i,
                                                                   recall = recall,accuracy = accuracy,FDR=FDR) 
    }
}
result_metrics = bind_rows(result_metrics)

##################

DFplot = result_metrics %>% mutate(ablations =case_when(model == 'all_keep' ~ 'full model',model == 'no_dms_all' ~ 'no DMS',
                                      model == 'esm2' ~ 'ESM-2 in place of ESM-MSA-1b',
                                      model == 'no_backgrounds' ~ 'no backgrounds'),
                           ablations = factor(ablations,levels = c('full model', 'no DMS', 'no backgrounds', 'ESM-2 in place of ESM-MSA-1b'))
                          )

theme_fontsize =  theme(axis.text=element_text(size=14),axis.title=element_text(size=15),
        legend.text = element_text(size = 13),legend.title = element_text(size = 15),legend.position = "bottom",
        plot.title = element_text(size = 15, face = "plain", hjust = 0.5)) 

p1 <- ggplot(DFplot %>% mutate(status = as.character(!is.na(FDR)),FDR = ifelse(is.na(FDR),1,FDR)), aes(x = proportion_threshold, y = FDR, color = ablations)) +
  geom_line(size=0.8) + 
  geom_point(aes(,shape=status),size = 2,fill = "white") +
  scale_shape_manual(name = "FDR status",values = c("FALSE" = 4, "TRUE" = 21)) + 
  scale_y_continuous(limits = c(0, 1)) + scale_x_continuous(breaks=seq(0.05,0.4,0.05),limits=c(0.04,0.41))+
  labs(title = "JN.1 era prediction", y = "False Discovery Rate", x= 'State3 threshold (proportion)') +
  scale_color_nejm() + theme_cowplot() + theme_fontsize
p2 <- ggplot(DFplot, aes(x = proportion_threshold, y = recall, color = ablations)) +
  geom_line(size=0.8) + 
  geom_point(shape = 21, size = 2, fill = "white") +
  scale_y_continuous(limits = c(0, 1)) + scale_x_continuous(breaks=seq(0.05,0.4,0.05),limits=c(0.04,0.41))+
  labs(title = "JN.1 era prediction", y = "Recall", x= 'State3 threshold (proportion)') +
  scale_color_nejm() + theme_cowplot() + theme_fontsize

pdf(str_glue('{data_dir}/ablation/plots/JN1_module_ablations_curves_dynamic_Thres.pdf'),width=5,height = 3)
p1
p2
# p3
dev.off()