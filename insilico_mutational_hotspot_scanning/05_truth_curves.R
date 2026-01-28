library(jsonlite)
library(tidyverse)
library(ggsci)

# dat_curve = read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/predict/results/rbd_single_JN1era/ValTestMajor_regres_outputs_labels-step-21502.csv')
file_dir <- '/lustre/grp/cyllab/share/evolution_prediction_dl'
name_mapper <- fromJSON(str_glue('{file_dir}/data/processed/to241030/rbd_name_mapper.json'), simplifyVector = T)
dat_curve <- read_csv(str_glue('{file_dir}/data/processed/to241030/rbd/2022-09-01/TestFull.csv')) %>% mutate(rbd_name_mut= name_mapper[rbd_name] %>% unname %>% as.character)

p <- ggplot(dat_curve %>% filter(rbd_name_mut %in%  c('XBB.1.5','EG.5','HK.3','JN.1+F456L','JN.1+R346T'),t0 < ymd('2024-09-01'))) +
  # geom_line(aes(x=col_date, y=col_output), color='darkred', size=0.4, alpha=0.8) + 
  geom_line(aes(x=t0, y=target_ratio_t0,color=rbd_name_mut), linewidth=0.5, alpha=0.5) +
  # geom_area(aes(x=t0, y=target_ratio_t0,fill=rbd_name_mut), size=1, alpha=0.8) +
  scale_x_date(date_breaks="2 month") + 
  scale_color_locuszoom() + 
  geom_vline(xintercept=ymd('2022-12-25'), linetype="dashed") +
  geom_vline(xintercept=ymd('2023-06-01'), linetype="dashed") +
  geom_vline(xintercept=ymd('2023-07-31'), linetype="dashed") +
  geom_vline(xintercept=ymd('2024-03-10'), linetype="dashed") +
  labs(x = 'date',y = 'proportion',color='') + 
  theme_classic() + 
  theme(plot.title = element_text(hjust=0.5),  
        axis.text.x=element_text(angle=45, vjust=1, hjust=1)) +
  coord_flip()

pdf(str_glue('{file_dir}/insilico_mutational_hotspot_scanning/results/True_curves.pdf'),height=6,width=4)
p
dev.off()