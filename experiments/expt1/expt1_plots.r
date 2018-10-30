###############################################################################
### plotting expt1 results (preliminary + limited, as of oct30/2018) ##########
###############################################################################
# 
# TODO: 
#   - integrate metadata into results df 
#   - better results df format more generally 
#   - encapsulate various plots to apply across results files 
#   - ... 
###############################################################################

lefftpack::lazy_setup()

results_files <- list(
  prelim_MNBvsFFNN = "results/prelim_results-MNBvsFFNN.csv")

plot_outfiles <- list(
  prelim_MNBvsFFNN = "results/prelim_results-MNBvsFFNN-plot.pdf")


prelim_res <- read.csv(results_files$prelim_MNBvsFFNN, as.is=TRUE) %>% 
  reshape2::melt(id.vars=c("clf", "lbin")) %>% 
  mutate(variable = as.character(variable)) %>% 
  # TODO: fix spacing in legend items (use this hack for now)
  mutate(clf = case_when(
    clf=="clfA" ~" multinomial naive bayes ", 
    clf=="clfB" ~ " FF neural net w/one hidden layer ")) %>%
  mutate(lbin = as.numeric(gsub("bin", "", lbin))) %>% 
  rename(metric=variable, length_bin=lbin) %>% dplyr::as_tibble()


prelim_res_plot <- prelim_res %>% 
  ggplot(aes(x=length_bin, y=value, color=clf)) + 
  geom_point() + geom_line() + 
  facet_wrap(~metric, scales="fixed") + 
  theme(legend.position="top") + 
  scale_y_continuous(limits=c(.6, 1), breaks=seq(from=.6, to=1, by=.1)) +
  labs(title="classification peformance on IMDB sentiment dataset", 
       subtitle="Naive Bayes versus simple neural net", 
       x="review length quartile (within IMDB sentiment dataset)", 
       y="metric on 25k review test set", 
       caption=paste("features from unigram count-DTM (max 10k vocab)", 
                     "(preliminary -- will produce more general results soon!)",
                     sep="\n"))

ggsave(plot_outfiles$prelim_MNBvsFFNN, prelim_res_plot, 
       width=10, height=8, units="in")


