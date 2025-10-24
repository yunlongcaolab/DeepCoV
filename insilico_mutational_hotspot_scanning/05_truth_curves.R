library(jsonlite)
library(tidyverse)

dat_curve = read_csv('/lustre/grp/cyllab/yangsj/evo_pred/1article/predict/results/rbd_single_JN1era/ValTestMajor_regres_outputs_labels-step-21502.csv')

rbd_name_mapper <- fromJSON("/lustre/grp/cyllab/yangsj/evo_pred/0article/data/processed/241030/rbd_name_mapper.json", simplifyVector = T)

p <- ggplot(dat_curve %>% filter(rbd_name_mut %in%  c('XBB.1.5','EG.5','HK.3','JN.1+F456L','JN.1+R346T'),t0 < ymd('2024-05-01'))) +
  # geom_line(aes(x=col_date, y=col_output), color='darkred', size=0.4, alpha=0.8) + 
  geom_line(aes(x=t0, y=target_ratio_t0,color=rbd_name_mut), size=0.5, alpha=0.5) +
  # geom_area(aes(x=t0, y=target_ratio_t0,fill=rbd_name_mut), size=1, alpha=0.8) +
  scale_x_date(date_breaks="1 month") + 
  scale_color_locuszoom() + 
  geom_vline(xintercept=ymd('2022-11-01'), linetype="dashed") +
  geom_vline(xintercept=ymd('2022-11-12'), linetype="dashed") +
  geom_vline(xintercept=ymd('2023-01-22'), linetype="dashed") +
  # scale_x_datetime(date_breaks="2 months") + 
  labs(x = 'date',y = 'proportion',color='strain') + 
  theme_classic() + 
  theme(plot.title = element_text(hjust=0.5),  # 标题居中
        axis.text.x=element_text(angle=45, vjust=1, hjust=1)) +
  coord_flip()

pdf('',height=2,width=6)
p
dev.off()