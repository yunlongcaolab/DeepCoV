library(tidyverse)
library(jsonlite)
library(lubridate)
library(ggsci)
library(caret)
library(cowplot)
theme_set(theme_cowplot())

data_dir = '/lustre/grp/cyllab/share/evolution_prediction_dl'

file_path = str_glue('{data_dir}/generalization/continue/results/TestMajor_regres_outputs_labels-step-54717.csv')
name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"), simplifyVector = T)

data = read_csv(file_path) %>% 
    mutate(rbd_name_mut= name_mapper[rbd_name] %>% unname %>% as.character) %>% mutate(t0_t1 = t1,t1 = t0+t1)

unique(data %>% filter(target_ratio_t1_output > 0.05) %>% pull(rbd_name_mut))

calculate_rmse <- function(df) {
  sqrt(mean((df$target_ratio_t1_label - df$target_ratio_t1_output)^2, na.rm = TRUE))
}

rmse_t0_t1 <- data %>%
  group_by(t0_t1) %>%
  summarise(rmse = calculate_rmse(cur_data()), .groups = "drop")

p <- ggplot(rmse_t0_t1, aes(x = t0_t1, y = rmse, fill = t0_t1)) +
  geom_col(color = "black", width = 0.7,alpha=0.3) +
  scale_y_continuous(expand=expansion())+
  # scale_x_continuous(expand=expansion())+
  # theme_cowplot() +
  labs(x = "days between t1 and t0", y = "RMSE") +
  scale_fill_distiller(palette = "YlGnBu") + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 11, angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 11),
    axis.title = element_text(size = 13),
    plot.title = element_text(size = 14, face = "plain", hjust = 0.5),
    plot.margin = margin(5, 5, 5, 5)
  )
pdf(str_glue('{data_dir}/generalization/continue/plots/rmse_JN1_era_continuous.pdf'), width=12, height=4)
p
dev.off()


plot_continuous = function(i = 'KP.3',date_start = '2024-04-01',days_try = 30,save_pdf = TRUE){
  ps=list()
  for (j in 0:(days_try - 1)) {
      date_start = as.Date(date_start)
      date_start_j <- date_start + days(j)

      df_j <- data %>%
        filter(rbd_name_mut == i, t0 == date_start_j)
      
      # p <- ggplot(df_j) +
      #   geom_line(aes(x=t1, y=target_ratio_t1_label), color = '#88A69D', linetype = "dashed", linewidth = 0.7) + 
      #   geom_line(aes(x=t1, y=target_ratio_t1_output), color = '#E7BA8F') + 
      #   geom_area(aes(x=t1, y=target_ratio_t1_output), fill = '#E7BA8F', alpha = 0.5) + 
      #   scale_x_date(date_breaks = "10 days",expand = expansion())  +
      #   scale_y_continuous(expand=expansion())+
      #   theme_classic() +
      #   theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
      #     plot.title = element_text(hjust = 0.5)) +
      #   labs(title = paste0(i, " predicted at ", date_start_j))
      df_j_past <- data %>% filter(rbd_name_mut == i, t0 >= date_start_j-days(30), t0 <= date_start_j)
      p <- ggplot(df_j) +
        # geom_line(aes(x=t1, y=target_ratio_t1_label), color = '#88A69D', linetype = "dashed", linewidth = 0.7) + 
        # geom_line(aes(x=t1, y=target_ratio_t1_output), color = '#E7BA8F') + 
        # geom_line(data=df_j_past,aes(x=t0, y=target_ratio_t0), color = '#88A69D') + 
        geom_line(aes(x=t1, y=target_ratio_t1_label), color = '#468CBC', linetype = "dashed", linewidth = 0.7) + 
        geom_line(aes(x=t1, y=target_ratio_t1_output), color = '#BA91A9') + 
        geom_line(data=df_j_past,aes(x=t0, y=target_ratio_t0), color = '#468CBC') + 
        geom_vline(xintercept = date_start_j, linetype = "dashed", color = "black", linewidth = 0.4) +
        scale_x_date(date_breaks = "10 days",expand = expansion())  +
        scale_y_continuous(expand=expansion())+
        theme_classic() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          plot.title = element_text(hjust = 0.5)) +
        labs(title = paste0(i, " predicted at ", date_start_j),y='proportion')
      ps[[j + 1]] <- p
  }
  if(save_pdf){
    pdf(str_glue('{data_dir}/generalization/continue/plots/trajectory_{i}_{date_start}_{days_try}.pdf'), width=6, height=2.7)
    for (p in ps) {print(p)}
    dev.off() 
    print('saved')
  }
  invisible(ps)
  # return(ps)
}

plot_continuous(i = 'KP.3',date_start = '2024-04-05',days_try = 10)
plot_continuous(i = 'KP.2',date_start = '2024-03-15',days_try = 10)
plot_continuous(i = 'JN.1',date_start = '2023-11-01',days_try = 10)
plot_continuous(i = 'HK.3',date_start = '2023-10-20',days_try = 10)
