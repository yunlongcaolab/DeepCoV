library(tidyverse)
library(ggsci)
library(jsonlite)

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

tag='rbd'

### JN.1 era
# test_major = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/ValTestMajor_regres_outputs_labels-step-36410.csv'))
# save_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/corr_JN1era_valtest.pdf')

### JN.1 era update
# test_major = read_csv(str_glue('{data_dir}/generalization/update/results/TestMajor_regres_outputs_labels-step-36410.csv')) %>%
#   filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3','LP.8','LF.7','NB.1.8.1'))
# save_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/corr_JN1era_update_testmajor.pdf')

### JN.1 era spike
tag='spike'
spike_name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/spike_name_mapper.json"))
test = read_csv(str_glue('{data_dir}/generalization/spike/results/TestFull_regres_outputs_labels-step-18485.csv'))  %>%
    mutate(spike_name_mut = spike_name_mapper[spike_name] %>% unname %>% as.character)
test_major = test %>% filter(spike_name_mut %in% c('JN.1','KP.2','KP.3','KP.3+S31-','KP.2+S31-+H146Q')) %>% 
  mutate(spike_name_mut = ifelse(spike_name_mut == 'KP.3+S31-', 'KP.3.1.1', spike_name_mut),
         spike_name_mut = ifelse(spike_name_mut == 'KP.2+S31-+H146Q', 'KP.2.3', spike_name_mut)) 
save_file = str_glue('{data_dir}/generalization/spike/plots/corr_JN1era_spike.pdf')

### XBB era
# test_major = read_csv(str_glue('{data_dir}/generalization/XBB_era/results/TestMajor_regres_outputs_labels-step-38360.csv')) %>%
#   filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3','XBB','XBB.1.5','EG.5','HK.3'))
# save_file = str_glue('{data_dir}/generalization/XBB_era/plots/corr_XBBera_testmajor.pdf')

p =ggplot(test_major,aes(x = target_ratio_t1_label,y=target_ratio_t1_output,color = rbd_name_mut)) + #  # spike_name_mut
    geom_point(alpha=0.5,size=1.5) + theme_classic() + scale_color_d3(palette="category20c")+
    labs(x = 'actual proportion at t1',y = 'predicted proportion at t1',color = 'strain') + 
    theme(axis.text=element_text(size=20),axis.title=element_text(size=22),
        legend.text = element_text(size = 18),legend.title = element_text(size = 18),legend.position = "right",
        plot.title = element_text(size = 20, face = "plain", hjust = 0.5)) +
    scale_x_continuous(expand = expansion(),limits=c(0,0.9)) + scale_y_continuous(expand = expansion(),limits=c(0,0.9)) + 
    annotate(
      "text",x = min(test_major$target_ratio_t1_label) + 0.1,y = max(test_major$target_ratio_t1_output) - 0.1,
      label = paste0("pearson correlation = ", round(cor(test_major$target_ratio_t1_label,test_major$target_ratio_t1_output), 3)), size = 8,  hjust = 0 
    )

test_major_rising <- test_major %>%
  group_by(!!sym(str_glue('{tag}_name'))) %>%
  mutate(max_t0 = t0[which.max(target_ratio_t1_output)]) %>%
  filter(t0 <= max_t0)

# p =ggplot(test_major_rising,aes(x = target_ratio_t1_label,y=target_ratio_t1_output,color = !!sym(str_glue('{tag}_name_mut')))) + #  # spike_name_mut
#     geom_point(alpha=0.5,size=1.5) + theme_classic() + scale_color_d3(palette="category20c")+
#     labs(x = 'actual proportion at t1',y = 'predicted proportion at t1',color = 'strain') + 
#     theme(axis.text=element_text(size=20),axis.title=element_text(size=22),
#         legend.text = element_text(size = 18),legend.title = element_text(size = 18),legend.position = "right",
#         plot.title = element_text(size = 20, face = "plain", hjust = 0.5)) +
#     # scale_x_continuous(expand = expansion(),limits=c(0,0.9)) + scale_y_continuous(expand = expansion(),limits=c(0,0.9)) + 
#     scale_x_continuous(expand = expansion(),limits=c(0,0.4)) + scale_y_continuous(expand = expansion(),limits=c(0,0.3)) + 
#     annotate(
#       "text",x = min(test_major_rising$target_ratio_t1_label) + 0.1,y = max(test_major_rising$target_ratio_t1_output) - 0.1,
#       label = paste0("pearson correlation = ", round(cor(test_major_rising$target_ratio_t1_label,test_major_rising$target_ratio_t1_output), 3)), size = 8,  hjust = 0 
#     )

pdf(save_file,width=7,height=5)
p
dev.off()