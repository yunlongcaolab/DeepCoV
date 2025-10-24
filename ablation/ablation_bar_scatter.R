library(tidyverse)
library(ggsci)
library(cowplot)
library(lubridate)
library(caret)

data_dir = '<path/to/dir>'

model_levels <- c('full model', 'no DMS', 'no backgrounds', 'ESM-2 in place of ESM-MSA-1b')
manual_color = ggsci::pal_nejm()(4)
names(manual_color) = model_levels

case_1_rep <- read_csv(str_glue('{data_dir}/ablation/results/keep_all_rep1/TestMajor_metrics_keep_all_rep1.csv')) %>% mutate(model = 'full model')
case_2_rep <- read_csv(str_glue('{data_dir}/ablation/results/no_dms_all_rep1/TestMajor_metrics_no_dms_all_rep1.csv')) %>% mutate(model = 'no DMS')
case_3_rep <- read_csv(str_glue('{data_dir}/ablation/results/ESM2_rep1/TestMajor_metrics_ESM2_rep1.csv')) %>% mutate(model = 'ESM-2 in place of ESM-MSA-1b')
case_4_rep <- read_csv(str_glue('{data_dir}/ablation/results/no_backgrounds_rep1/TestMajor_metrics_no_backgrounds_rep1.csv')) %>% mutate(model = 'no backgrounds')

case_data_rep <- bind_rows(case_1_rep, case_2_rep, case_3_rep, case_4_rep)
case_data_rep <- case_data_rep %>% mutate(model = factor(model, levels = rev(model_levels), ordered = TRUE))

case_1_rep2 <- read_csv(str_glue('{data_dir}/ablation/results/keep_all_rep2/TestMajor_metrics_keep_all_rep2.csv')) %>% mutate(model = 'full model')
case_2_rep2 <- read_csv(str_glue('{data_dir}/ablation/results/no_dms_all_rep2/TestMajor_metrics_no_dms_all_rep2.csv')) %>% mutate(model = 'no DMS')
case_3_rep2 <- read_csv(str_glue('{data_dir}/ablation/results/ESM2_rep2/TestMajor_metrics_ESM2_rep2.csv')) %>% mutate(model = 'ESM-2 in place of ESM-MSA-1b')
case_4_rep2 <- read_csv(str_glue('{data_dir}/ablation/results/no_backgrounds_rep2/TestMajor_metrics_no_backgrounds_rep2.csv')) %>% mutate(model = 'no backgrounds')

case_data_rep2 <- bind_rows(case_1_rep2, case_2_rep2, case_3_rep2, case_4_rep2)
case_data_rep2 <- case_data_rep2 %>% mutate(model = factor(model, levels = rev(model_levels), ordered = TRUE))


case_1_rep3 <- read_csv(str_glue('{data_dir}/ablation/results/keep_all_rep3/TestMajor_metrics_keep_all_rep3.csv')) %>% mutate(model = 'full model')
case_2_rep3 <- read_csv(str_glue('{data_dir}/ablation/results/no_dms_all_rep3/TestMajor_metrics_no_dms_all_rep3.csv')) %>% mutate(model = 'no DMS')
case_3_rep3 <- read_csv(str_glue('{data_dir}/ablation/results/ESM2_rep3/TestMajor_metrics_ESM2_rep3.csv')) %>% mutate(model = 'ESM-2 in place of ESM-MSA-1b')
case_4_rep3 <- read_csv(str_glue('{data_dir}/ablation/results/no_backgrounds_rep3/TestMajor_metrics_no_backgrounds_rep3.csv')) %>% mutate(model = 'no backgrounds')

case_data_rep3 <- bind_rows(case_1_rep3, case_2_rep3, case_3_rep3, case_4_rep3)
case_data_rep3 <- case_data_rep3 %>% mutate(model = factor(model, levels = rev(model_levels), ordered = TRUE))

data_bind = bind_rows(case_data_rep,case_data_rep2,case_data_rep3) 

p = ggplot(data_bind, aes(x=model, y=TestMajor_rmse, fill=model))+
    stat_summary(fun='mean', geom='bar',show.legend = F,alpha = 0.3,linewidth=0.3)+
    stat_summary(fun = 'mean', geom = "text", aes(label = sprintf("%.3f", ..y..)),
                 position = position_nudge(y = 0.05), show.legend = FALSE) + 
    geom_point(position = position_jitter(0.4),size=1.5, alpha=0.5,shape=21, fill='white', show.legend = F)+
    theme_cowplot()+
    scale_color_manual(values=manual_color) + scale_fill_manual(values=manual_color) + 
    scale_y_continuous(limits=c(0,0.3))+
    labs(x = '', y = 'RMSE', title = '') +
    guides(color='none',fill='none')+coord_flip()

pdf(str_glue('{data_dir}/ablation/plots/TestMajor_RMSE_module_ablation_scatter.pdf'), width=4.5, height=2.5)
p
dev.off()
