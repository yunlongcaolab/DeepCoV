library(tidyverse)
library(ggsci)
library(jsonlite)
library(readr)
library(dplyr)
library(stringr)
library(cowplot)

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"), simplifyVector = TRUE)

res1 = read_csv(str_glue('{data_dir}/ablation/results/no_dms_all_rep1/TestFull_regres_outputs_labels_no_dms_all_rep1.csv')) %>% 
  mutate(rbd_name_mut = name_mapper[rbd_name] %>% unname %>% as.character) %>% 
  mutate(model='no DMS')

res2 = read_csv(str_glue('{data_dir}/ablation/results/keep_all_rep1/TestFull_regres_outputs_labels_keep_all_rep1.csv')) %>% 
  mutate(rbd_name_mut = name_mapper[rbd_name] %>% unname %>% as.character) %>% 
  mutate(model='full model')

res3 = read_csv(str_glue('{data_dir}/ablation/results/no_backgrounds_rep1/TestFull_regres_outputs_labels_no_backgrounds_rep1.csv')) %>% 
  mutate(rbd_name_mut = name_mapper[rbd_name] %>% unname %>% as.character) %>% 
  mutate(model='no backgrounds')
dat = bind_rows(res1,res2,res3)

# strain_select = 'KP.2'
strain_select = 'BA.2.86+K478E'

dat_plot = dat %>% filter(rbd_name_mut == strain_select )
p = ggplot(data = dat_plot # %>% filter(rbd_name_mut == 'KP.2')
          ) + 
        geom_line(aes(x=t0, y=target_ratio_t0),color='black') + 
        geom_line(aes(x=t0, y=target_ratio_t1_output,color=model)) + #,linetype=model
        scale_color_nejm() + 
        scale_x_date(date_breaks = "2 month", date_labels = "%Y-%m-%d") + 
        scale_linetype_manual() + 
        theme_cowplot() + 
        labs(x = 'date', y = 'proportion', title = str_glue('{strain_select} tragectory')) + 
        theme(axis.text.x = element_text(angle = 45, hjust = 1))

pdf(str_glue('{data_dir}/ablation/plots/{strain_select}_growth_reconstruction.pdf'), width=6, height=3)
p
dev.off()
  