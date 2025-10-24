library(tidyverse)
library(jsonlite)
library(lubridate)
library(ggsci)
library(caret)
library(cowplot)
theme_set(theme_cowplot())

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

### custom
tag='rbd'

### XBB era
# name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"), simplifyVector = T)
# res = read_csv(str_glue('{data_dir}/generalization/XBB_era/results/TestMajor_regres_outputs_labels-step-38360.csv')) %>%
#       filter(t0 >='2022-09-01') %>%
#       mutate(rbd_name_mut= name_mapper[rbd_name] %>% unname %>% as.character)
# out_file = str_glue('{data_dir}/generalization/XBB_era/plots/XBBera_lineages_tragectory.pdf')
# strain_draw = c('JN.1','KP.2','KP.3','XBB','XBB.1.5','EG.5','HK.3')
# colors = c(pal_locuszoom("default")(6),"#2B9B81FF")
# plot_width = 8

### JN.1 era
# name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"), simplifyVector = T)
# res = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv')) %>% 
#       mutate(rbd_name_mut= name_mapper[rbd_name] %>% unname %>% as.character)
# out_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/JN1era_lineages_tragectory.pdf')
# strain_draw = c('HK.3','BA.2.86','JN.1','KP.2','KP.3')
# colors = c(pal_locuszoom("default")(6),"#2B9B81FF")
# plot_width = 7

### JN.1 era update
# name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to250516/rbd_name_mapper.json"), simplifyVector = T)
# res = read_csv(str_glue('{data_dir}/generalization/update/results/TestMajor_regres_outputs_labels-step-36410.csv')) %>% 
#       filter(t0 >='2024-02-01') %>% 
#       mutate(rbd_name_mut= name_mapper[rbd_name] %>% unname %>% as.character)
# out_file = str_glue('{data_dir}/generalization/update/plots/JN1era_update_lineages_tragectory.pdf')
# strain_draw = c('JN.1','KP.2','KP.3','LF.7','LP.8','NB.1.8.1')
# colors = c(pal_locuszoom("default")(6),"#2B9B81FF")
# plot_width = 4.5

### JN.1 era spike
name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/spike_name_mapper.json"), simplifyVector = T)
res = read_csv(str_glue('{data_dir}/generalization/spike/results/TestFull_regres_outputs_labels-step-18485.csv')) %>% 
  mutate(spike_name_mut = name_mapper[spike_name] %>% unname %>% as.character) %>% 
  mutate(spike_name_mut = ifelse(spike_name_mut == 'KP.3+S31-', 'KP.3.1.1', spike_name_mut),
         spike_name_mut = ifelse(spike_name_mut == 'KP.2+S31-+H146Q', 'KP.2.3', spike_name_mut)) 
out_file = str_glue('{data_dir}/generalization/spike/plots/JN1era_spike_lineages_tragectory.pdf')
strain_draw = c('JN.1','KP.2','KP.3','KP.3.1.1','KP.2.3')
colors = c(pal_locuszoom("default")(6),"#2B9B81FF")
tag = 'spike'
plot_width = 6

print('data loaded.')

res = res %>% mutate(t1 = t0 + t1)  %>% mutate(t1_biweekly = floor_date(as.Date(t1), "1 week"))

res_week = res %>% group_by(!!sym(str_glue('{tag}_name_mut')),t1_biweekly,location) %>% 
    summarize(# target_isolates_t0 = sum(target_isolates_t0),total_isolates_t0 = sum(total_isolates_t0),
              target_ratio_t1_output_sd = sd(target_ratio_t1_output, na.rm = TRUE),
              target_ratio_t1_output = mean(target_ratio_t1_output),target_ratio_t1_label = mean(target_ratio_t1_label),target_ratio_t0 = mean(target_ratio_t0)
              ) 
res_week_errorbar = res_week %>% mutate(date = t1_biweekly) %>% 
    mutate(t1_output_min = target_ratio_t1_output - target_ratio_t1_output_sd,
           t1_output_max = target_ratio_t1_output + target_ratio_t1_output_sd)

p = res_week_errorbar %>% mutate(date = t1_biweekly-30) %>% 
  # filter(date < '2024-05-05',date > '2023-10-01') %>% # difference
  filter(!!sym(str_glue('{tag}_name_mut')) %in% strain_draw) %>% 
  ggplot() + 
  # geom_line(aes(x=date, y=target_ratio_t1_output, color=!!sym(str_glue('{tag}_name_mut')))) + # t0 
  # geom_line(aes(x=date, y=target_ratio_t0, color=!!sym(str_glue('{tag}_name_mut'))),linetype='dashed') + 
  geom_line(aes(x=date, y=target_ratio_t1_output, color=!!sym(str_glue('{tag}_name_mut'))),linetype='dashed') + # t0 
  geom_line(aes(x=date, y=target_ratio_t0, color=!!sym(str_glue('{tag}_name_mut')))) + 
  # geom_col(position='stack', alpha=alpha,color = 'white') +
  # geom_errorbar(aes(x=date, y=target_ratio_t1_output, color=lineage,ymin = t1_output_min, ymax = t1_output_max), width = 4) + 
  geom_ribbon(aes(x=date, y=target_ratio_t1_output, fill=!!sym(str_glue('{tag}_name_mut')),ymin = t1_output_min, ymax = t1_output_max), alpha=0.3) + 
  scale_color_manual(values=colors) + 
  scale_fill_manual(values=colors) + 
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m-%d",#  date_labels = "%Y.%b.%d", 
               expand=expansion()) +
  scale_y_continuous(expand=expansion())+ # ,limits=c(0,max=ymax)
  theme(axis.text.x=element_text(angle=45, vjust=1, hjust=1)
        ,legend.text = element_text(size = 9)
        # ,legend.key.size = unit(0.39, "cm")
        ,legend.title = element_blank()
        ,legend.position = 'top'
  ) + 
  labs(x='t0',y = 'percentage',color='lineage'
       # ,title = country
      ) 

pdf(out_file,height = 4,width=plot_width)
p 
dev.off()

### MAE & RMSE
# errors_test = res_week %>% rename(date = t1_biweekly) %>% group_by(date) %>% 
#     summarize(rmse = postResample(pred = target_ratio_t1_output, obs = target_ratio_t1_label)[1] %>% unname,
#               mae = postResample(pred = target_ratio_t1_output, obs = target_ratio_t1_label)[3] %>% unname)

## errors_case = res_week %>% rename(date = t1_biweekly) %>% filter(rbd_name_mut %in% strain_draw) %>% 
##     group_by(date) %>% 
##     summarize(rmse = postResample(pred = target_ratio_t1_output, obs = target_ratio_t1_label)[1] %>% unname,
##               mae = postResample(pred = target_ratio_t1_output, obs = target_ratio_t1_label)[3] %>% unname)
## errors_dataset = bind_rows(errors_case %>% mutate(dataset='major strains'),errors_test %>% mutate(dataset='all strains'))
# errors_dataset = errors_case %>% mutate(dataset='major strains')

# p1 = errors_dataset  %>% 
#     ggplot(aes(x = date,y = rmse,color=dataset)) + 
#     geom_point(size=0.8) + 
#     geom_line() + 
#     scale_x_date(date_breaks = "2 months", date_labels = "%Y-%m-%d",#  date_labels = "%Y.%b.%d", 
#     ) +
#     scale_y_continuous(breaks=seq(0,0.16,0.05),limits=c(0,0.18),expand=expansion())+ 
#     scale_color_locuszoom() + 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
#     labs(x = '',y='RMSE')
# p2 = errors_dataset  %>% 
#     ggplot(aes(x = date,y = mae,color=dataset)) + 
#     geom_point(size=0.8) + 
#     geom_line() + 
#     scale_x_date(date_breaks = "2 months",date_labels = "%Y-%m-%d",#  date_labels = "%Y.%b.%d", 
#     ) +
#     scale_y_continuous(breaks=seq(0,0.1,0.05),limits=c(0,0.1),expand=expansion())+ 
#     scale_color_locuszoom() + 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
#     labs(x = '',y='MAE')
# save_path = str_replace(out_file, "\\.pdf$", "_mae_rmse.pdf")
# pdf(save_path,height = 2.3,width=4.5)
# p1
# p2
# dev.off()


