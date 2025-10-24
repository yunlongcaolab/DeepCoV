library(tidyverse)
library(cowplot)
library(ggsci)
library(jsonlite)
library(lubridate)
library(ggrepel)

data_dir='<path/to/dir>'

res0 = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_keep_all/JN1-2024-03-10_regres_outputs_labels-step-36410.csv')) %>% mutate(model = 'full model')
res1 = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_esm2/JN1-2024-03-10_regres_outputs_labels-step-3641.csv')) %>% mutate(model = 'esm2')
res2 = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_backgrounds/JN1-2024-03-10_regres_outputs_labels-step-14564.csv')) %>% mutate(model = 'no backgrounds')
res3 = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_dms_all/JN1-2024-03-10_regres_outputs_labels-step-25487.csv'))  %>% mutate(model = 'no DMS')

trans = function(res,tag='rbd',min_thres=0,i='2024-03-10',location='Global',sites= c(331:531)){
res_mut =res %>% filter(!!sym(str_glue('{tag}_name_mut')) %>% str_detect(coll('+')) == T)  %>% 
           mutate(date =as.character(t0),
           base_strain = !!sym(str_glue('{tag}_name_mut')) %>% str_split(coll('+')) %>% map_chr(~.[[1]]),
           mut = !!sym(str_glue('{tag}_name_mut')) %>% str_split(coll('+')) %>% map_chr(~.[[2]]),site = mut %>% str_extract('\\d+') %>% as.numeric,
           aa_from = mut %>% str_extract('^[A-Z]'),aa_to = mut %>% str_extract('[A-Z-]$') )
date=res_mut %>% pull(date) %>% unique
dat = res_mut %>% filter(location == location,base_strain == base_strain) %>% 
    select(!!sym(str_glue('{tag}_name_mut')) ,base_strain,date,mut,site,aa_from,aa_to,target_ratio_t1_output) %>% arrange(-target_ratio_t1_output) 

aa_avg = dat %>% group_by(aa_to) %>% summarize(target_ratio_t1_output = mean(target_ratio_t1_output))
Vaa_avg = aa_avg$target_ratio_t1_output
names(Vaa_avg) = aa_avg$aa_to
Vaa_avg

dat_hm_ori = dat %>% mutate(score=target_ratio_t1_output - Vaa_avg[aa_to] %>% unname %>% as.numeric) %>% mutate(score = ifelse(score > min_thres,score,0),mut = ifelse(score > min_thres,mut,'')) 
data_score = dat_hm_ori %>% filter(date == i)
dat = data_score %>% group_by(site) %>% summarize(score = mean(score)) %>%  
        right_join(data.frame(site = sites)) %>% 
        mutate(score = ifelse(is.na(score),0,score))
return(dat)
}

p1 = trans(res0) %>% rename(`full model` = score) %>% full_join(trans(res1) %>% rename(`esm2` = score)) %>% 
    mutate(site_show=ifelse((esm2 > 1e-9|site %in% c(346,456)),site,''),is_hotspots=ifelse(site %in% c(346,456),'yes','no')) %>% 
    ggplot(aes(x=`full model`,y=`esm2`)) + geom_point(aes(color=is_hotspots)) + theme_classic() + geom_label_repel(aes(label=site_show)) + 
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank()) + 
    labs(x='score (full model)',y='score (esm2)') + 
    scale_color_manual(values=c('yes'='#BC3C29FF','no'='#20854EFF'))

p2 = trans(res0) %>% rename(`full model` = score) %>% full_join(trans(res2) %>% rename(`no backgrounds` = score)) %>% 
    mutate(site_show=ifelse((`no backgrounds` > 1e-3|site %in% c(346,456)),site,''),is_hotspots=ifelse(site %in% c(346,456),'yes','no')) %>% 
    ggplot(aes(x=`full model`,y=`no backgrounds`)) + geom_point(aes(color=is_hotspots)) + theme_classic() + geom_label_repel(aes(label=site_show))  +
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank()) + 
    labs(x='score (full model)',y='score (no backgrounds)') + 
    scale_color_manual(values=c('yes'='#BC3C29FF','no'='#E18727FF'))

p3 = trans(res0) %>% rename(`full model` = score) %>% full_join(trans(res3) %>% rename(`no DMS` = score)) %>% 
    mutate(site_show=ifelse((`no DMS` > 0.0001|site %in% c(346,456)),site,''),is_hotspots=ifelse(site %in% c(346,456),'yes','no')) %>% 
    ggplot(aes(x=`full model`,y=`no DMS`)) + geom_point(aes(color=is_hotspots)) + theme_classic() + geom_label_repel(aes(label=site_show)) + 
    theme(axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank()) + 
    labs(x='score (full model)',y='score (no DMS)') + 
    scale_color_manual(values=c('yes'='#BC3C29FF','no'='#0072B5FF'))

pdf(str_glue('{data_dir}/ablation/plots/ablation_mutscanning_scatter.pdf'), width=4.5, height=3)
p1
p2
p3
dev.off()