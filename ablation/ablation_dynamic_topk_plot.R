library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggvenn)


# Set data directory
file_dir = '<path/to/ablation_dir>'

# top 1 - success rate
rank_eval_results = read_csv(str_glue('{file_dir}/analysis/ablation_success_predtop20_truthtop1.csv'))
summary_success = rank_eval_results %>% group_by(n_pred_top,n_truth_top) %>% summarize(`full model` = mean(`full.model`),`no DMS`=mean(`no.DMS`),`no historical proportion`=mean(`no.historical.proportion`),`esm2`=mean(`esm2`),`no backgrounds`=mean(`no.backgrounds`)) %>% 
    pivot_longer(-c('n_pred_top','n_truth_top'),names_to='methods',values_to='success_rate') %>% 
    filter(methods != 'no historical proportion') %>%
    mutate(methods = factor(methods,levels = c('full model', 'no DMS', 'no backgrounds','esm2')))
theme_fontsize =  theme(axis.text=element_text(size=12),axis.title=element_text(size=15), 
        legend.text = element_text(size = 10),legend.title = element_text(size = 10),# legend.position = "bottom",
        plot.title = element_text(size = 15, face = "plain", hjust = 0.5)) 
p = summary_success %>% ggplot(aes(x= n_pred_top,y=success_rate,color=methods)) + geom_point(size=0.8) + geom_line(alpha=0.7) + 
    theme_classic() + scale_color_nejm() + scale_y_continuous(expand = expansion()) + # scale_color_locuszoom() + 
    labs(x='k',y = 'prediction success rate',title=str_glue('Top 1 dominant strain prediction (ablation)')) + theme_fontsize

pdf(str_glue('{file_dir}/plots/ablation_success_rate_predtopK_truthtop1.pdf'), width=5, height=2.7)
p
dev.off()

# top k(1,3) - overlap rate
n_truth_top = 3
n_pred_top_max = 20
data = read_csv(str_glue('{file_dir}/analysis/ablation_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv'))
data_plot = data %>% group_by(n_pred_top,n_truth_top)  %>% filter(n_pred_top >= n_truth_top) %>%
    pivot_longer(-c('n_pred_top','n_truth_top','date'),names_to='methods',values_to='jaccard_index') %>% 
    group_by(n_pred_top,n_truth_top,methods) %>% summarize(mean_jaccard_index = mean(jaccard_index),sd_jaccard_index = sd(jaccard_index),sem_jaccard_index  = sd(jaccard_index) / sqrt(n())) %>% 
    mutate(methods = str_replace_all(methods,'\\.',' ')) %>%
    filter(methods != 'no historical proportion') %>%
    mutate(methods = factor(methods,levels = c('full model', 'no DMS', 'no backgrounds', 'no historical proportion', 'esm2')))

p = data_plot %>% ggplot(aes(x= n_pred_top,y=mean_jaccard_index,color=methods)) + geom_point(size=0.8) + geom_line(alpha=0.7) + 
    geom_errorbar(aes(ymin = pmax(mean_jaccard_index - sem_jaccard_index, 0),ymax = pmin(mean_jaccard_index + sem_jaccard_index,1)),width = 0.3, alpha = 0.5) +  
    theme_classic() + scale_color_nejm() + scale_y_continuous(expand = expansion()) + 
    labs(x='k',y = 'overlap ratio',title=str_glue('Top {n_truth_top} dominant strain prediction (ablation)')) + theme_fontsize 
pdf(str_glue('{file_dir}/plots/ablation_jaccard_index_predtopK_truthtop{n_truth_top}_sem.pdf'), width=5, height=2.7)
p
dev.off()

n_truth_top = 5
n_pred_top_max = 20
data = read_csv(str_glue('{file_dir}/analysis/ablation_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv'))
data_plot = data %>% group_by(n_pred_top,n_truth_top)  %>% filter(n_pred_top >= n_truth_top) %>%
    pivot_longer(-c('n_pred_top','n_truth_top','date'),names_to='methods',values_to='jaccard_index') %>% 
    group_by(n_pred_top,n_truth_top,methods) %>% summarize(mean_jaccard_index = mean(jaccard_index),sd_jaccard_index = sd(jaccard_index),sem_jaccard_index  = sd(jaccard_index) / sqrt(n())) %>% 
    mutate(methods = str_replace_all(methods,'\\.',' ')) %>%
    filter(methods != 'no historical proportion') %>%
    mutate(methods = factor(methods,levels = c('full model', 'no DMS', 'no backgrounds', 'no historical proportion', 'esm2')))

p = data_plot %>% ggplot(aes(x= n_pred_top,y=mean_jaccard_index,color=methods)) + geom_point(size=0.8) + geom_line(alpha=0.7) +  
    geom_errorbar(aes(ymin = pmax(mean_jaccard_index - sem_jaccard_index, 0),ymax = pmin(mean_jaccard_index + sem_jaccard_index,1)),width = 0.3, alpha = 0.5) +  
    theme_classic() + scale_color_nejm() + scale_y_continuous(expand = expansion()) + 
    labs(x='k',y = 'overlap ratio',title=str_glue('Top {n_truth_top} dominant strain prediction (ablation)')) + theme_fontsize 
pdf(str_glue('{file_dir}/plots/ablation_jaccard_index_predtopK_truthtop{n_truth_top}_sem.pdf'), width=5, height=2.7)
p
dev.off()
