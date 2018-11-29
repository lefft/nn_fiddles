### can run interactively or from shell via `$Rscript expt1_plot_results.r`

### TODO:
#   - make note of length quartile ranges and n's
#   - maybe include overall scores as well 
#   - apply this to another dataset 
#   - configure y-axis automatically!
#   - combine color and shape legends!!! 

library(magrittr)
library(ggplot2)


# columns of results_file are:
#   c('clf', 'lbin', 'f1', 'accuracy', 'precision', 'recall')
results_file <- 'results_nov29/results-nov29.csv'
plot_outfile <- 'results_nov29/results-nov29_plot.pdf'


yax_limits <- c(.7, 1)
yax_breaks <- seq(from=min(yax_limits), to=max(yax_limits), by=.1)


res <- read.csv(results_file, stringsAsFactors=FALSE)

res$lbin <- as.integer(gsub('q', '', res$lbin))
res$clf_type <- ifelse(grepl('hidden', res$clf), 'neural net', 'NB/LR')

res <- res %>% reshape2::melt(id.vars=c('clf', 'lbin', 'clf_type'))



out <- res %>% 
 ggplot(aes(x=lbin, y=value, color=clf, fill=clf, shape=clf_type)) + 
 geom_point(size=3, alpha=.5) + geom_line() + 
 facet_wrap(~variable, scales='free') + 
 scale_y_continuous(limits=yax_limits, breaks=yax_breaks, expand=c(0,.01)) + 
 scale_x_continuous(expand=c(.02,.01)) + 
 labs(x='IMDB review length quartile', y='score', 
      title='Expt 1: Sentiment classification performance by text length',
      subtitle='neural nets, naive bayes, logistic regression') + 
 theme_minimal() + 
 theme(axis.line=element_line(color='gray'))


ggsave(plot_outfile, out, width=10, height=7, units='in')
message('wrote results plot to: `', plot_outfile, '`')

