library(tidyverse)
library(ggseqlogo)
library(Biostrings)
library(ggsci)
# library(patchwork)
library(ggrepel)

draw_mut2site_score = function(data_score,title,site_show_normal = c(346,456,478,475),tag='rbd'){
    if(tag == 'rbd'){
        breaks_k = seq(330,530,5)
        limits_k = c(330,531)
        sites = c(331:531)
   }else if(tag == 'spike'){
        breaks_k = seq(0,310,5)
        limits_k = c(0,311)
        sites = c(1:331)
   }
    dat = data_score %>% group_by(site) %>% summarize(score = mean(score)) %>%  
        right_join(data.frame(site = sites)) %>% 
        mutate(score = ifelse(is.na(score),0,score))

   site_high = dat %>% filter(score > 0) %>% arrange(-score) %>% dplyr::slice(1:2) %>% pull(site) #  %>% dplyr::slice(1:3)  %>% dplyr::slice(1:10)
   print(site_high)
   site_show = c(site_show_normal,site_high)

   p = dat %>% ggplot(aes(x=site,y=score)) +
   geom_line(color="#A03429", size=0.8, alpha=0.8) + 
   geom_point(color="#A03429", shape=21)+ theme_classic() + theme(
    axis.text.y=element_blank(),
    axis.ticks.y=element_blank(),
    # axis.title=element_blank(),
    axis.text.x=element_text(angle=45, hjust=0.5, vjust=0.5)
  )+ scale_x_continuous(breaks=breaks_k,limits=limits_k) + 
    geom_label_repel(aes(label=ifelse(site %in% site_show,site, '')), min.segment.length = 0, direction="both", fill = alpha(c("white"),0.5), max.overlaps=50) +
    ggtitle(title)

    return(p)
}

### custom

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'
tag = 'rbd' # default

# JN.1 scanning (update model)
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-36410.csv'))
# postfix = ''
# base_strain = 'JN.1'
# location = 'Global'
# site_show_normal = c(346, 456)
# min_thres = 0.00
# nshow_site = 10 # logoplot

## JN.1 ablation scanning
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_esm2')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_esm2/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-5804.csv'))
# postfix = ' - esm2'
# # save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_keep_all')
# # res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_keep_all/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-3993.csv'))
# # postfix = ' - full model'
# # save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_backgrounds')
# # res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_backgrounds/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-3993.csv'))
# # postfix = ' - no backgrounds'
# # save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_bgratios')
# # res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_bgratios/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-36275.csv'))
# # postfix = ' - no historical proportion'
# # save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_dms_all')
# # res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_JN1mutscan_no_dms_all/JN1-2024-01-20to2024-03-31_regres_outputs_labels-step-1089.csv'))
# # postfix = ' - no DMS'

# base_strain = 'JN.1'
# location = 'Global'
# site_show_normal = c(346, 456)
# min_thres = 0.00
# nshow_site = 10 # logoplot

## KP.3 scanning in JN.1 era update
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_update')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_update/KP3-2024-08-01to2024-10-01_regres_outputs_labels-step-36410.csv'))
# postfix = ''
# base_strain = 'KP.3'
# location = 'Global'
# site_show_normal = c(435)
# min_thres = 0.00
# nshow_site = 10 # logoplot

## LF.7 scanning in JN.1 era update
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_update')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_JN1era_update/LF7-2024-08-01to2024-12-01_regres_outputs_labels-step-36410.csv'))
# postfix = ''
# base_strain = 'LF.7'
# location = 'Global'
# site_show_normal = c(475)
# min_thres = 0.00
# nshow_site = 10 # logoplot

# KP.3 scanning in JN.1 era spike
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/spike_single_JN1era')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/spike_single_JN1era/KP3-2024-06-01to2024-09-01_regres_outputs_labels-step-18485.csv')) %>%
#     filter(spike_name_mut %>% str_detect('ins') == FALSE)
# postfix = ''
# base_strain = 'KP.3'
# location = 'Global'
# site_show_normal = c(31)
# min_thres = 0.00
# nshow_site = 10 # logoplot
# tag = 'spike'

# XBB.1.5 scanning in XBB era
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera/XBB15-2023-05-01to2023-06-01_regres_outputs_labels-step-38360.csv'))
# base_strain = 'XBB.1.5'

# EG.5 scanning in XBB era
# save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera')
# res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera/EG5-2023-07-01to2023-09-01_regres_outputs_labels-step-38360.csv'))
# base_strain = 'EG.5'

# XBB scanning in XBB era
save_dir = str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera')
res = read_csv(str_glue('{data_dir}/insilico_mutational_hotspot_scanning/results/rbd_single_XBBera/XBB-2022-11-01to2023-01-01_regres_outputs_labels-step-38360.csv'))
base_strain = 'XBB'

postfix = ''
location = 'Global'
site_show_normal = c(486)
min_thres = 0.00
nshow_site = 10 # logoplot

###

if(dir.exists(save_dir) == FALSE){
    print(str_glue('create {save_dir}'))
    dir.create(save_dir, recursive = TRUE) # if dir.exists(save_dir)  
}

res_mut =res %>% filter(!!sym(str_glue('{tag}_name_mut')) %>% str_detect(coll('+')) == T)  %>% 
           mutate(date =as.character(t0),# date = as.Date(as.character(t0), format="%Y%m%d"),
           base_strain = !!sym(str_glue('{tag}_name_mut')) %>% str_split(coll('+')) %>% map_chr(~.[[1]]),
           mut = !!sym(str_glue('{tag}_name_mut')) %>% str_split(coll('+')) %>% map_chr(~.[[2]]),site = mut %>% str_extract('\\d+') %>% as.numeric,
           aa_from = mut %>% str_extract('^[A-Z]'),aa_to = mut %>% str_extract('[A-Z-]$') )
date=res_mut %>% pull(date) %>% unique
dat = res_mut %>% filter(location == location,base_strain == base_strain) %>% 
    select(!!sym(str_glue('{tag}_name_mut')) ,base_strain,date,mut,site,aa_from,aa_to,target_ratio_t1_output) %>% arrange(-target_ratio_t1_output) 

aa_avg = dat %>% group_by(aa_to) %>% summarize(target_ratio_t1_output = mean(target_ratio_t1_output))
Vaa_avg = aa_avg$target_ratio_t1_output# *2
# Vaa_avg = rep(0,20)
names(Vaa_avg) = aa_avg$aa_to
Vaa_avg

dat_hm_ori = dat %>% mutate(score=target_ratio_t1_output - Vaa_avg[aa_to] %>% unname %>% as.numeric) %>% mutate(score = ifelse(score > min_thres,score,0),mut = ifelse(score > min_thres,mut,'')) 

##############
### site plot
##############

ps_ori = list()
for(i in sort(unique(dat_hm_ori$date))){
    p = draw_mut2site_score(data_score = dat_hm_ori %>% filter(date == i),site_show_normal,
                # title = str_glue('{base_strain} singlemuts - {location} on {i} (original){postfix}')
                title = str_glue('{base_strain} scanning on {i} {postfix}'),tag=tag
                           )
    ps_ori[[i]] = p
}

pdf(str_glue('{save_dir}/hotspots_{base_strain}_{location}_period_{min(dat_hm_ori$date)}to{max(dat_hm_ori$date)}_min{min_thres}{postfix}.pdf'),width = 6,height = 2.5) # ,width = 9
ps_ori
dev.off()

##############
### logoplot
##############
colors <- c(
  "D"="#E60A0A", "E"="#E70B0B",
  "C"="#A6A600", "M"="#A6A600",
  "K"="#256CFF", "R"="#266DFF",
  "S"="#FA9600", "T"="#FB9701",
  "F"="#6565CC", "Y"="#6666CD", "W"="#6666CD", "H"="#6666CD",
  "N"="#218989", "Q"="#218989",
  "G"="#0F820F", "L"="#000000", "V"="#000000", "I"="#000000", "A"="#000000",
  "P"="#DC9682",
  "X"="#aaaaaa"# "-"="#aaaaaa"
)
cs1 = make_col_scheme(chars=names(colors), groups=names(colors),cols=colors)

ps_logo_ori = list()
for(i in sort(unique(dat_hm_ori$date))){
    print(i)
    show_site = dat_hm_ori %>% filter(date == i) %>% group_by(site) %>% summarize(score_sum = sum(score)) %>% 
        arrange(-score_sum) %>% dplyr::slice(1:nshow_site) %>% pull(site)
    dat_logo_ori = dat_hm_ori %>% filter(date == i) %>% arrange(-score) %>% filter(score > 0, site %in% show_site)%>% 
        arrange(site) %>% select(aa_to, site, score) %>% pivot_wider(names_from = 'site', values_from = 'score') %>% 
        column_to_rownames('aa_to') %>% mutate_all(~ifelse(is.na(.), 0, .)) %>% as.matrix()
    rownames(dat_logo_ori) = rownames(dat_logo_ori) %>% str_replace('-','X')
    print(rownames(dat_logo_ori))
    if(nrow(dat_logo_ori) > 1){
        p_logo_ori = ggseqlogo(dat_logo_ori, method = "custom", seq_type = "aa", col_scheme = cs1) +
            scale_x_continuous(breaks = 1:ncol(dat_logo_ori), labels = colnames(dat_logo_ori)) + 
          theme_bw() + 
          theme(panel.grid.major = element_blank(),panel.grid.minor = element_blank(),
            axis.text.y=element_blank(),axis.ticks.y=element_blank(),axis.title.y=element_blank(),axis.title.x=element_blank(),
            axis.text.x=element_text(angle=90, hjust=0.5, vjust=0.5,size=30))+
            labs(title = str_glue('{base_strain} singlemuts - {location} on {i} (original){postfix}'), 
                 x = "Position", y = "Score")
        
        ps_logo_ori[[as.character(i)]] = p_logo_ori
    }
}

pdf(str_glue('{save_dir}/logoplot_{base_strain}_{location}_{min(dat_hm_ori$date)}to{max(dat_hm_ori$date)}_min{min_thres}{postfix}_topshowSite{nshow_site}.pdf'),width = 5,height = 4)
ps_logo_ori
dev.off()