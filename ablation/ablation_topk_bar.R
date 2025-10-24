library(tidyverse)
library(cowplot)
library(ggsci)
library(stringr)

# Set file directory
file_dir = '<path/to/ablation_dir>'

# Read the metrics data
result_metrics = read_csv(str_glue('{file_dir}/analysis/Topk_ablation_result_metrics_10major_fullset.csv'))

theme_fontsize =  theme(axis.text=element_text(size=15),axis.title=element_text(size=17),
        legend.text = element_text(size = 7),legend.title = element_text(size = 7),legend.position = "bottom",
        plot.title = element_text(size = 18, face = "plain", hjust = 0.5))
model_levels <- c('full model', 'no DMS', 'no backgrounds', 'ESM-2 in place of ESM-MSA-1b')
manual_color = ggsci::pal_nejm()(4)
names(manual_color) = model_levels

# topk=10
topk=15
p1 = result_metrics %>% filter(k==topk) %>% mutate(method = factor(method,levels = rev(c('full model', 'no backgrounds', 'no DMS', 'ESM-2 in place of ESM-MSA-1b'))))%>% 
  ggplot(aes(x = method,y = FDR,color=method,fill=method)) + 
  geom_col(alpha = 0.3,linewidth=0.5) + geom_text(aes(label =round(FDR,5)),color='black',hjust=-0.2) + 
  theme_cowplot() +
  scale_color_manual(values=manual_color) + scale_fill_manual(values=manual_color) + 
  scale_y_continuous(limits=c(0,1))+
  labs(x = '', y = str_glue('FDR (k={topk})'), title = '') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+
  guides(color='none',fill='none')+ coord_flip() 

p2 = result_metrics %>% filter(k==topk) %>% mutate(method = factor(method,levels = rev(c('full model', 'no backgrounds', 'no DMS', 'ESM-2 in place of ESM-MSA-1b'))))%>% 
  ggplot(aes(x = method,y = recall,color=method,fill=method)) + 
  geom_col(alpha = 0.3,linewidth=0.5) + geom_text(aes(label =round(recall,5)),color='black', hjust = -0.2) + 
  theme_cowplot() +
  scale_color_manual(values=manual_color) + scale_fill_manual(values=manual_color) + 
  scale_y_continuous(limits=c(0,1))+
  labs(x = '', y = str_glue('recall (k={topk})'), title = '') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))+
  guides(color='none',fill='none')+ coord_flip() + scale_y_reverse()

pdf(str_glue('{file_dir}/plots/top{topk}_ablation_bar.pdf'), width=4, height=3)
p1
p2
dev.off()