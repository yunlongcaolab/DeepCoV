library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)
library(ggvenn)

# Set directories
analysis_data_dir <- '/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/analysis'
plot_save_dir <- '/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/plots'

# Ensure directories exist
dir.create(plot_save_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(analysis_data_dir, showWarnings = FALSE, recursive = TRUE)

# top 1 - success rate
rank_eval_results = read_csv(str_glue('{analysis_data_dir}/res_success_predtop20_truthtop1.csv'))
summary_success = rank_eval_results %>% group_by(n_pred_top,n_truth_top) %>% 
    summarize(EVEscape = mean(EVEscape),E2VD=mean(E2VD),DeepCoV=mean(DeepCoV),ga=mean(ga)) %>% 
    pivot_longer(-c('n_pred_top','n_truth_top'),names_to='methods',values_to='success_rate')
theme_fontsize =  theme(axis.text=element_text(size=13),axis.title=element_text(size=15),
        legend.text = element_text(size = 10),legend.title = element_text(size = 10),# legend.position = "bottom",
        plot.title = element_text(size = 15, face = "plain", hjust = 0.5)) 
p = summary_success %>% 
  ggplot(aes(x = n_pred_top, y = success_rate, color = methods)) + 
  geom_point(size = 0.8) + 
  geom_line(alpha = 0.7) + 
  theme_classic() + 
  scale_color_nejm() + 
  scale_y_continuous(expand = expansion()) + 
  labs(x = 'k', y = 'prediction success rate', 
       title = str_glue('Top 1 dominant strain prediction')) + 
  theme_fontsize + 
  theme(legend.position = 'bottom')
pdf(str_glue('{plot_save_dir}/success_rate_predtopK_truthtop1.pdf'),width=5,height=3)
p
dev.off()

# top k(1,3) - overlap rate
n_truth_top = 3
# n_truth_top = 5
n_pred_top_max = 20
data = read_csv(str_glue('{analysis_data_dir}/res_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv')) 
data_plot = data %>% 
  group_by(n_pred_top, n_truth_top) %>% 
  filter(n_pred_top >= n_truth_top) %>%
  pivot_longer(-c('n_pred_top', 'n_truth_top', 'date'), 
               names_to = 'methods', 
               values_to = 'jaccard_index') %>% 
  group_by(n_pred_top, n_truth_top, methods) %>% 
  summarize(
    mean_jaccard_index = mean(jaccard_index),
    n = dplyr::n(),
    sem_jaccard_index = sd(jaccard_index) / sqrt(n())
  )

p = data_plot %>% 
  ggplot(aes(x = n_pred_top, y = mean_jaccard_index, color = methods)) + 
  geom_point(size = 0.8) + 
  geom_line(alpha = 0.7) + 
  geom_errorbar(
    aes(ymin = pmax(mean_jaccard_index - sem_jaccard_index, 0),
        ymax = pmin(mean_jaccard_index + sem_jaccard_index, 1)),
    width = 0.3, 
    alpha = 0.5
  ) + 
  theme_classic() + 
  scale_color_nejm() + 
  scale_y_continuous(expand = expansion()) + 
  labs(x = 'k', 
       y = 'overlap ratio',
       title = str_glue('Top {n_truth_top} dominant strain prediction')) + 
  theme_fontsize + 
  theme(legend.position = 'bottom')
pdf(str_glue('{plot_save_dir}/jaccard_index_predtopK_truthtop{n_truth_top}.pdf'),width=5,height=3)
p
dev.off()

n_truth_top = 5
n_pred_top_max = 20
data = read_csv(str_glue('{analysis_data_dir}/res_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv'))
data_plot = data %>% 
  group_by(n_pred_top, n_truth_top) %>% 
  filter(n_pred_top >= n_truth_top) %>%
  pivot_longer(-c('n_pred_top', 'n_truth_top', 'date'), 
               names_to = 'methods', 
               values_to = 'jaccard_index') %>% 
  group_by(n_pred_top, n_truth_top, methods) %>% 
  summarize(
    mean_jaccard_index = mean(jaccard_index),
    n = dplyr::n(),
    sem_jaccard_index = sd(jaccard_index) / sqrt(n())
  )

p = data_plot %>% 
  ggplot(aes(x = n_pred_top, y = mean_jaccard_index, color = methods)) + 
  geom_point(size = 0.8) + 
  geom_line(alpha = 0.7) + 
  geom_errorbar(
    aes(ymin = pmax(mean_jaccard_index - sem_jaccard_index, 0),
        ymax = pmin(mean_jaccard_index + sem_jaccard_index, 1)),
    width = 0.3, 
    alpha = 0.5
  ) + 
  theme_classic() + 
  scale_color_nejm() + 
  scale_y_continuous(expand = expansion()) + 
  labs(x = 'k', 
       y = 'overlap ratio',
       title = str_glue('Top {n_truth_top} dominant strain prediction')) + 
  theme_fontsize + 
  theme(legend.position = 'bottom')
pdf(str_glue('{plot_save_dir}/jaccard_index_predtopK_truthtop{n_truth_top}.pdf'),width=5,height=3)
p
dev.off()

# Top k to k comparison
n_truth_top = 20
n_pred_top_max = 20

# Read and process data
data = read_csv(str_glue('{analysis_data_dir}/res_success_predtop{n_pred_top_max}_truthtop{n_truth_top}.csv')) 

data_plot = data %>% 
  group_by(n_pred_top, n_truth_top) %>% 
  filter(n_pred_top >= n_truth_top) %>%
  pivot_longer(-c('n_pred_top', 'n_truth_top', 'date'), 
               names_to = 'methods', 
               values_to = 'jaccard_index') %>% 
  group_by(n_pred_top, n_truth_top, methods) %>% 
  summarize(
    mean_jaccard_index = mean(jaccard_index),
    n = dplyr::n(),
    sem_jaccard_index = sd(jaccard_index) / sqrt(n())
  )

# Create the plot
p = data_plot %>% 
  ggplot(aes(x = n_pred_top, y = mean_jaccard_index, color = methods)) + 
  geom_point(size = 0.8) + 
  geom_line(alpha = 0.7) + 
  geom_errorbar(
    aes(ymin = pmax(mean_jaccard_index - sem_jaccard_index, 0),
        ymax = pmin(mean_jaccard_index + sem_jaccard_index, 1)),
    width = 0.3, 
    alpha = 0.5
  ) + 
  theme_classic() + 
  scale_color_nejm() + 
  scale_y_continuous(expand = expansion()) + 
  labs(x = 'k', 
       y = 'overlap ratio',
       title = 'Top k dominant strains prediction') + 
  theme_fontsize + 
  theme(legend.position = 'bottom')

# Save the plot
pdf(str_glue('{plot_save_dir}/jaccard_index_predtopK_truthtopK_sem.pdf'), width = 5, height = 3)
print(p)
dev.off()

