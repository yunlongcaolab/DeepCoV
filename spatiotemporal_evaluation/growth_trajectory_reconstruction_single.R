library(tidyverse)
library(jsonlite)
library(lubridate)
library(ggsci)
library(caret)
library(cowplot)
theme_set(theme_cowplot())

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

# JN.1 era
name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"), simplifyVector = T)
res = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv')) %>%mutate(rbd_name_mut = name_mapper[rbd_name])
out_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/JN1era_lineages_tragectory_minor.pdf')
strain_draw = c('JN.1+F456L','JN.1+R346T','JN.1+K403R','JN.1+N417K','JN.1+H445P+F456L','KP.2+T346I','KP.3+K440R','KP.3+S408R')

# JN.1 era spike
# name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/241030/spike_name_mapper.json"), simplifyVector = T)
# res = read_csv(str_glue('{data_dir}/generalization/spike/results/TestFull_regres_outputs_labels-step-18485.csv')) %>%
#     mutate(spike_name_mut = name_mapper[spike_name]) %>%
#     mutate(spike_name_mut = ifelse(spike_name_mut == 'KP.3+S31-', 'KP.3.1.1', spike_name_mut),
#            spike_name_mut = ifelse(spike_name_mut == 'KP.2+S31-+H146Q', 'KP.2.3', spike_name_mut)) 
# out_file = str_glue('{data_dir}/generalization/spike/plots/JN1era_spike_lineages_tragectory_single.pdf')
# strain_draw = c('JN.1','KP.2','KP.3','KP.3.1.1','KP.2.3')

###
ps = list()
for(name in strain_draw){
    dat_draw = res %>% filter(rbd_name_mut == name)
    # dat_draw = res %>% filter(spike_name_mut == name)
    p = ggplot(data = dat_draw) + 
        # geom_line(aes(x=t0, y=target_ratio_t0), color = '#88A69D') + 
        # geom_line(aes(x=t0, y=target_ratio_t1_output), color = '#E7BA8F') + 
        # geom_area(aes(x=t0, y=target_ratio_t0), fill = '#88A69D', alpha = 0.5) + 
        # geom_area(aes(x=t0, y=target_ratio_t1_output), fill = '#E7BA8F', alpha = 0.5) + 
        geom_line(aes(x=t0, y=target_ratio_t0), color = '#468CBC') + 
        geom_line(aes(x=t0, y=target_ratio_t1_output), color = '#BA91A9') + 
        geom_area(aes(x=t0, y=target_ratio_t0), fill = '#468CBC', alpha = 0.5) +  #3778A5
        geom_area(aes(x=t0, y=target_ratio_t1_output), fill = '#BA91A9', alpha = 0.5) + 
        # scale_color_locuszoom() + 
        scale_x_date(date_breaks = "2 month", date_labels = "%Y-%m-%d") + 
        scale_y_continuous(limits = c(0, 0.25), expand = expansion()) + 
        theme_cowplot() + 
        labs(x = 'date', y = 'proportion', title = name) + 
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    ps[[name]] = p  
}

pdf(out_file,width=6,height=2.7)
ps
dev.off()