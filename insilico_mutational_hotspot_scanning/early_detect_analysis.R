library(tidyverse)
library(ggsci)
library(ggsignif)
# library(Rmisc)
library(Hmisc)
library(cowplot)
library(patchwork)

setwd('/Users/sibyl/Desktop/XieLab/05_model/0revision/analysis')
# dat_compare = read_csv('/Users/sibyl/Desktop/XieLab/05_model/0revision/analysis/dat_scanning_max_compare.csv')
dat_compare = read_csv('/Users/sibyl/Desktop/XieLab/05_model/0revision/analysis/dat_scanning_max_compare_more.csv')
# View(dat_compare)
dat_compare = dat_compare %>% filter(rbd_name_mut!='XBB.1.5+K478R') %>% 
  mutate(status = ifelse(status =='max','peak',status),
         `mutant harbouring hotspots` = case_when(
            rbd_name_mut=='XBB.1.5' ~ 'XBB.1.5 (XBB + S486P)',
            rbd_name_mut=='EG.5' ~ 'EG.5 (XBB.1.5 + F456L)',
            rbd_name_mut=='EG.5+L452R' ~ 'HV.1 (EG.5 + L452R)',
            rbd_name_mut=='KP.2' ~ 'KP.2 (JN.1 + R346T + F456L)',
            rbd_name_mut=='HK.3' ~ 'HK.3 (EG.5 + L455F)',
            TRUE~rbd_name_mut
          ))
p1 = dat_compare %>% # filter(rbd_name_mut %in% c('JN.1+R346T','JN.1+F456L')==FALSE) %>% 
  ggplot(aes(x = status,y = ratio,color = `mutant harbouring hotspots`)) + 
  geom_point(size = 2,alpha=0.5) + 
  # geom_point(size = 2,shape=21,alpha=0.5) + 
  geom_line(aes(group = rbd_name_mut),alpha=0.3) + 
  scale_y_continuous(limits=c(0,0.7))  + 
  theme_classic() + scale_color_npg()  + 
  geom_signif(comparisons = list(c(1,2)),
              tip_length = 0.05,color = 'black',step_increase = 0.07,
              # map_signif_level = TRUE,
              y_position = 0.65
              # annotation = c('a','b','c')
  ) + labs(x='proportion') + 
  theme(axis.title = element_text(size = 14),axis.text  = element_text(size = 12),
    legend.title = element_text(size = 13),legend.text  = element_text(size = 12),
    text = element_text(size = 12))

pdf('hotspots_overall_significance.pdf',height = 3.5,width = 7)
p1
dev.off()

dat_compare_period = read_csv('/Users/sibyl/Desktop/XieLab/05_model/0revision/analysis/dat_scanning_max_compare_more_pre10d.csv') %>%
  filter(rbd_name_mut!='XBB.1.5+K478R')

dat_compare_period = dat_compare_period %>% filter(rbd_name_mut!='XBB.1.5+K478R') %>% 
  mutate(status = ifelse(status =='max','peak',status),
         rbd_name_mut = case_when(
           rbd_name_mut=='XBB.1.5' ~ 'XBB.1.5 (XBB + S486P)',
           rbd_name_mut=='EG.5' ~ 'EG.5 (XBB.1.5 + F456L)',
           rbd_name_mut=='EG.5+L452R' ~ 'HV.1 (EG.5 + L452R)',
           rbd_name_mut=='KP.2' ~ 'KP.2 (JN.1 + R346T + F456L)',
           rbd_name_mut=='HK.3' ~ 'HK.3 (EG.5 + L455F)',
           TRUE~rbd_name_mut
         ))
ps = list()
colors = pal_npg()(length(unique(dat_compare_period$rbd_name_mut)))
names(colors) = unique(dat_compare_period$rbd_name_mut)
for(i in unique(dat_compare_period$rbd_name_mut)){
  dat_draw=dat_compare_period %>% filter(rbd_name_mut == i)
  p = dat_draw %>% 
    mutate(status = ifelse(status =='max','peak',status)) %>% 
    ggplot(aes(x = status,y = ratio)) + 
    # facet_wrap(~rbd_name_mut) + 
    geom_point(size = 0.2,shape=21,alpha=0.5,color = colors[i]) + 
    # geom_point(size = 2,shape=21,alpha=0.5) + 
    # geom_line(aes(group = rbd_name_mut),alpha=0.3) + 
    scale_y_continuous(limits=c(0,max(dat_draw$ratio)+0.1))  + 
    theme_classic() + scale_color_npg()  + 
    geom_signif(comparisons = list(c(1,2)),
                tip_length = 0.05,color = 'black',step_increase = 0.07,
                map_signif_level = TRUE
                # y_position = 0.65,
                # annotation = c('a','b','c')
    ) +labs(title=i) + theme(plot.title = element_text(size = 10))
  ps[[i]] = p
}


p2 = wrap_plots(ps, ncol = 3)

pdf('hotspots_strain_date_significance.pdf',height = 7,width = 7)
p2
dev.off()

  