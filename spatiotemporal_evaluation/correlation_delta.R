library(tidyverse)
library(ggsci)
library(jsonlite)

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

tag='rbd'

### JN.1 era
test_major = read_csv(str_glue('{data_dir}/predict/results/rbd_single_JN1era/ValTestMajor_regres_outputs_labels-step-36410.csv'))
save_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/corr_JN1era_valtest_delta.pdf')
show_limits = 0.6

### JN.1 era update
# test_major = read_csv(str_glue('{data_dir}/generalization/update/results/TestMajor_regres_outputs_labels-step-36410.csv')) %>%
#   filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3','LP.8','LF.7','NB.1.8.1'))
# save_file = str_glue('{data_dir}/spatiotemporal_evaluation/plots/corr_JN1era_update_testmajor_delta.pdf')
# show_limits = 0.5

candidate_cluster = c('JN.1','KP.2','KP.3','LP.8','LF.7','NB.1.8.1','BA.1','BA.2','BA.5','BF.7','XBB','BQ.1.1','XBB.1.5','EG.5','HK.3','BA.2.86')
candidate_color = pal_d3(palette="category20c")(length(candidate_cluster))
names(candidate_color) = candidate_cluster

test_major = test_major %>% mutate(delta_ratio_output = target_ratio_t1_output - target_ratio_t0,
                                   delta_ratio_label = target_ratio_t1_label - target_ratio_t0)

p =ggplot(test_major,aes(x = delta_ratio_label,y=delta_ratio_output,color = rbd_name_mut)) + #  # spike_name_mut
    geom_point(alpha=0.5,size=1.5) + theme_bw(base_size = 14) +
    scale_color_d3(palette="category20c")+
    labs(x = 'actual delta proportion (t1-t0)',y = 'predicted delta proportion (t1-t0)',color = 'strain') +
    theme(panel.grid.minor = element_blank(), # panel.grid.major = element_blank(), 
        axis.text=element_text(size=20),axis.title=element_text(size=22),
        legend.text = element_text(size = 18),legend.title = element_text(size = 18),legend.position = "right",
        plot.title = element_text(size = 20, face = "plain", hjust = 0.5)) +
    annotate(
      "text",x = -show_limits + 0.1,y = show_limits - 0.1,
      label = paste0("pearson correlation = ", round(cor(test_major$delta_ratio_label,test_major$delta_ratio_output), 3)), size = 8,  hjust = 0
    ) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, color = "grey50", linetype = "dotted", linewidth = 0.8) + 
    coord_fixed(ratio = 1, xlim = c(-show_limits,show_limits), ylim = c(-show_limits,show_limits)) 

pdf(save_file,width=7,height=5)
p
dev.off()