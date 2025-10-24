library(tidyverse)
library(cowplot)
library(ggsci)

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

### JN.1 era
# dat = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/LocDiffMajor_regres_outputs_labels-step-36410.csv')) %>% mutate(t1_date = t0 + t1)
# save_file=str_glue('{data_dir}/spatiotemporal_evaluation/plots/geographic_JN1era.pdf')
# d_break="3 month"

### JN.1 era update
dat = read_csv(str_glue('{data_dir}/generalization/update/results/LocDiffMajor_regres_outputs_labels-step-36410.csv')) %>% mutate(t1_date = t0 + t1)
save_file=str_glue('{data_dir}/generalization/update/plots/geographic_JN1era_update.pdf')
d_break="2 week"

ps = list()
# dat = res_bg180
# dat=res_bg180_nodms
for(name in unique(dat$rbd_name_mut)){
    dat_draw = dat %>% filter(rbd_name_mut == name)
    if(name == 'BA.1'){dat_draw = dat_draw %>% filter(t0 < as.Date('2023-01-01'))}
    if(name == 'BQ.1.1'){dat_draw = dat_draw %>% filter(t0 < as.Date('2023-10-01'))}
    if(name == 'XBB.1.5'){dat_draw = dat_draw %>% filter(t0 < as.Date('2024-04-01'))}
    if(name == 'HK.3'){dat_draw = dat_draw %>% filter(t0 < as.Date('2024-06-01'))}
    p = ggplot(data = dat_draw) + 
        # geom_line(aes(x=t1_date,y=target_ratio_t1_output,color = location)) + 
        # geom_line(aes(x=t1_date,y=target_ratio_t1_label,color = location),linetype='dashed') + 
        # geom_line(aes(x=t0,y=target_ratio_t1_output,color = location)) + 
        # geom_line(aes(x=t0,y=target_ratio_t0,color = location),linetype='dashed') + 
        geom_line(aes(x=t0,y=target_ratio_t1_output,color = location),linetype='dashed') + 
        geom_line(aes(x=t0,y=target_ratio_t0,color = location)) + 
        scale_color_locuszoom() + 
        scale_x_date(date_breaks = d_break,date_labels = "%Y-%m-%d") + scale_y_continuous(expand = expansion()) + 
        theme_cowplot() + labs(x= 'date',y = 'proportion',title=name) + 
        theme(axis.text.x=element_text(angle=45,hjust=1))
    ps[[name]] = p  
}

pdf(save_file, width=5, height=3)
ps
dev.off()