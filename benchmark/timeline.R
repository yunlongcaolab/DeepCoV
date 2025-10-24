library(tidyverse)
library(jsonlite)
library(lubridate)
library(ggrepel)

data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl'
analysis_data_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/analysis'
plot_save_dir='/lustre/grp/cyllab/share/evolution_prediction_dl/benchmark/plots'

month_rankings <- function(df, date_col = "min_date",id_col = "accession_id",score_col = "evescape",window_months = 2) {
  ranked_dfs <- list()
  unique_dates <- unique(df %>% pull(!!sym(date_col)))
  
  for (i in 1:length(unique_dates)) {
    date = unique_dates[i]
    window_start <- date %m-% months(window_months/2)
    window_end <- date %m+% months(window_months/2)
    historical_data <- df %>%filter(!!sym(date_col) <= window_end & !!sym(date_col) >= window_start)
    historical_data <- historical_data %>% mutate(rank = rank(-!!sym(score_col), ties.method = "average"))
    current_ids <- df %>% filter(!!sym(date_col) == date) %>% pull(!!sym(id_col))
    current_ranks <- historical_data %>% filter(!!sym(id_col) %in% current_ids) %>%select(!!sym(date_col), !!sym(id_col), rank)
    ranked_dfs[[as.character(date)]] <- current_ranks
  }

  rank_df <- bind_rows(ranked_dfs)
  result <- df %>% left_join(rank_df, by = c(date_col, id_col)) %>% rename(!!paste0(score_col,"_rank_", window_months, "m") := rank)
  return(result)
}

form_rank_df = function(df,method = 'ours', rank_col = 'ours_rank', date_col='t0', major_strain=c('JN.1','KP.2','KP.3'),topn =topn, Event='rank'){
  df %>% filter(rbd_name_mut %in% major_strain) %>% 
    filter(!!sym(rank_col) < topn) %>% group_by(rbd_name_mut) %>% summarise(date_top = min(!!sym(date_col))) %>%
    complete(rbd_name_mut = major_strain, fill = list(date_top = NA)) %>% mutate(method = method, topn = topn, Event = str_c(rbd_name_mut,' ',Event))
}

rbd_name_mapper <- fromJSON(str_glue("{data_dir}/data/processed/to241030/rbd_name_mapper.json"))
meta <- read_csv(str_glue("{data_dir}/data/processed/to241030/meta241030.csv.gz"))

ours = read_csv(str_glue("{data_dir}/predict/results/rbd_single_JN1era/TestFull_regres_outputs_labels-step-36410.csv")) %>% 
    mutate(t1_date = t0 + t1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

evescape = read_csv(str_glue("{data_dir}/benchmark/EVEscape/EVEscape_scores_test_JN1era.csv"))
e2vd = read_csv(str_glue("{data_dir}/benchmark/E2VD/E2VD_scores_test_JN1era.csv"))
ga = read_csv(str_glue("{data_dir}/benchmark/growth_advantage/rbd_test_JN1era_ga.csv")) %>% 
    mutate(ga = as.numeric(ga_cov)) %>% 
    filter(ga < 1) %>% 
    mutate(rbd_name_mut = rbd_name_mapper[rbd_name] %>% unname %>% as.character)

meta_test = meta %>% filter(rbd_name %in% unique(ours$rbd_name),submit_date > ymd('2023-10-01'),submit_date < ymd('2024-09-01'))
evescape_score_mapper = evescape %>% select(rbd_name,`EVEscape score_pos`) %>% deframe() 
e2vd_score_mapper = e2vd %>% select(rbd_name,E2VD) %>% deframe() 
meta_test_scores = meta_test %>% mutate(EVEscape = evescape_score_mapper[rbd_name],E2VD = e2vd_score_mapper[rbd_name])

### rank
other_rank = meta_test_scores %>% month_rankings(., date_col = "submit_date",id_col = "epi_id",score_col = "EVEscape",window_months = 2) %>%
                     month_rankings(., date_col = "submit_date",id_col = "epi_id",score_col = "E2VD",window_months = 2)

### rank of dominant strains
topn1=10
topn2=3
bind_rank = bind_rows(
    ours %>% group_by(t0) %>% mutate(ours_rank = rank(-target_ratio_t1_output)) %>% 
    form_rank_df(.,method = 'ours', rank_col = 'ours_rank', date_col='t0', major_strain=c('JN.1','KP.2','KP.3'),topn =topn1, Event=str_glue('rank top{topn1}')),
    other_rank %>% form_rank_df(.,method = 'EVEscape', rank_col = 'EVEscape_rank_2m', date_col='submit_date', major_strain=c('JN.1','KP.2','KP.3'),topn =topn1, Event=str_glue('rank(2m) top{topn1}')),
    other_rank %>% form_rank_df(.,method = 'E2VD', rank_col = 'E2VD_rank_2m', date_col='submit_date', major_strain=c('JN.1','KP.2','KP.3'),topn =topn1, Event=str_glue('rank(2m) top{topn1}')),
    ours %>% group_by(t0) %>% mutate(ours_rank = rank(-target_ratio_t1_output)) %>% 
    form_rank_df(.,method = 'ours', rank_col = 'ours_rank', date_col='t0', major_strain=c('JN.1','KP.2','KP.3'),topn =topn2, Event=str_glue('rank top{topn2}')),
    other_rank %>% form_rank_df(.,method = 'EVEscape', rank_col = 'EVEscape_rank_2m', date_col='submit_date', major_strain=c('JN.1','KP.2','KP.3'),topn =topn2, Event=str_glue('rank(2m) top{topn2}')),
    other_rank %>% form_rank_df(.,method = 'E2VD', rank_col = 'E2VD_rank_2m', date_col='submit_date', major_strain=c('JN.1','KP.2','KP.3'),topn =topn2, Event=str_glue('rank(2m) top{topn2}')),
)

timeline_data = bind_rows(bind_rank,
    ours %>% filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3')) %>% 
    filter(target_ratio_t0 >= 0.2) %>% group_by(rbd_name_mut) %>% summarise(date_top = min(t0)) %>%
    mutate(method = 'truth', topn = NA, Event = str_c(rbd_name_mut,' > 20%')),
    ours %>% filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3')) %>% 
    filter(target_isolates_t0 > 10) %>% group_by(rbd_name_mut) %>% summarise(date_top = min(t0)) %>%
    mutate(method = 'truth', topn = NA, Event = str_c(rbd_name_mut,' sequences > 10')),
    ga %>% filter(rbd_name_mut %in% c('JN.1','KP.2','KP.3')) %>% 
    filter(ga > 0.3,target_isolates_t0 > 10) %>% group_by(rbd_name_mut) %>% summarise(date_top = min(t0)) %>%
    mutate(method = 'truth', topn = NA, Event = str_c(rbd_name_mut,' advantage > 30%'))
)
timeline_data = timeline_data %>% 
    mutate(direction = ifelse(rbd_name_mut %in% c('JN.1','KP.3'),1,-1),
          position = case_when(Event %>% str_detect(coll('rank(2m) top10'))==TRUE ~ 1,Event %>% str_detect(coll('rank top10'))==TRUE ~ 1,
                               Event %>% str_detect(coll('rank(2m) top3'))==TRUE ~ 2,Event %>% str_detect(coll('rank top3'))==TRUE ~ 2,
                               Event %>% str_detect(coll(' > 20%'))==TRUE ~ 1.5,Event %>% str_detect(coll('sequences > 10'))==TRUE ~ 0.5,
                               Event %>% str_detect(coll('advantage > '))==TRUE ~ 1,
                              )) %>% 
    mutate(date = case_when(is.na(date_top) == FALSE ~ date_top,
                            rbd_name_mut == 'JN.1' ~ timeline_data %>% filter(method=='truth', Event %>% str_detect(coll(' > 20%'))==TRUE) %>% filter(rbd_name_mut == 'JN.1') %>% pull(date_top),
                            rbd_name_mut == 'KP.2' ~ timeline_data %>% filter(method=='truth', Event %>% str_detect(coll(' > 20%'))==TRUE) %>% filter(rbd_name_mut == 'KP.2') %>% pull(date_top),
                            rbd_name_mut == 'KP.3' ~ timeline_data %>% filter(method=='truth', Event %>% str_detect(coll(' > 20%'))==TRUE) %>% filter(rbd_name_mut == 'KP.3') %>% pull(date_top)
                            ))
timeline_data %>% write_csv(str_glue('{data_dir}/benchmark/analysis/timeline_data.csv'))

# timeline_data  = read_csv('/lustre/grp/cyllab/yangsj/evo_pred/0article/benchmark/timeline_data.csv')

timeline_data_draw = timeline_data %>% mutate(position = position*direction,text_position = position+0.5*direction)%>%
    mutate(success = ifelse(is.na(date_top),'no','yes')) %>% mutate(success = ifelse(Event %>% str_detect('rank')==T,success,'others')) %>% 
    mutate(method = ifelse(method == 'ours','DeepCoV',method)) %>%
    mutate(method = factor(method,levels = c("truth","DeepCoV","EVEscape","E2VD")))  %>%
    mutate(text_position_x = as.Date(ifelse(success == 'no',date - days(70),date)))

month_buffer <- 1
month_date_range <- seq(min(timeline_data_draw$date) - months(month_buffer), max(timeline_data_draw$date) + months(month_buffer), by='month')
month_format <- format(month_date_range, '%b')
month_df <- data.frame(month_date_range, month_format)
year_date_range <- seq(min(timeline_data_draw$date) - months(month_buffer), max(timeline_data_draw$date) + months(month_buffer), by='year')
year_date_range <- as.Date(
intersect(ceiling_date(year_date_range, unit="year"),floor_date(year_date_range, unit="year")),
origin = "1970-01-01")
year_format <- format(year_date_range, '%Y')
year_df <- data.frame(year_date_range, year_format)

shape_mapper = c('no'=4,'yes'=20,'others'=21)
linetype_mapper = c('no'='dashed','yes'='solid','others'='solid')
method_colors = c("#A52514","#1F77B4FF","#5C9683") 

p = ggplot(timeline_data_draw,aes(x=date,y= position,color=rbd_name_mut, label=Event)) +labs(col="") +
    theme_classic() +
    geom_label_repel(data = timeline_data_draw, aes(x=text_position_x,y=text_position,label=Event),size=3.2,force=0.1)  +
    geom_segment(data=timeline_data_draw, aes(y=position,yend=0,xend=date,linetype=success), linewidth=0.3,color = 'black') + # 'gray70'
    geom_hline(yintercept=0, color = "black", linewidth=0.5) +
    geom_point(aes(y=position,shape=success), size=3) +
    scale_color_manual(values = method_colors) +
    scale_shape_manual(values=shape_mapper) +
    scale_linetype_manual(values=linetype_mapper) +
    geom_text(data=month_df, aes(x=month_date_range,y=-0.25,label=month_format),size=2.5,vjust=0.5, color='black') + #, angle=90
    geom_text(data=year_df, aes(x=year_date_range,y=-0.6,label=year_format, fontface="bold"),size=2.5, color='black') +
    theme(axis.line.x =element_blank(),axis.line.y=element_blank(),axis.text.y=element_blank(),axis.text.x =element_blank(),
        axis.title.x=element_blank(),axis.title.y=element_blank(),
        axis.ticks.y=element_blank(),axis.ticks.x =element_blank(),
        strip.background = element_blank(),strip.text = element_text(size=12),
    legend.position = "top",legend.title = element_blank()) +guides(shape='none',linetype='none') + facet_wrap(~method,ncol=1)


pdf(str_glue('{plot_save_dir}/timeline.pdf'), height=8, width=5)
p
dev.off()