# source: http://selbydavid.com/2018/01/09/neural-network/
lefftpack::lazy_setup()

lefftpack::import(makemedat_, from='nn_phonxe.r')
dat <- makemedat_(n=200, cycles=1, x1_sd=.15, x2_sd=.15, seed=6933)

lefftpack::import(train_, from='nn_phonxe.r')
menet <- train_(hidden=5, learn_rate=1e-2, iters=1e4)

lefftpack::import(grid_, from='nn_phonxe.r')
megrid <- grid_(menet)

lefftpack::import(plotg_, from='nn_phonxe.r')
plotg_(megrid, fixed=TRUE)

ggsave('blaowwie3.pdf', width=4, height=4, units='in')
#


plot(sigmoid, from=-10, to=10)


# sometimes u will get error bc all preds are the same so levels off in grid_
# seems to fuck up if too high of learning rates are tried...
for (iters in as.integer(10^(3:4))){
  for (rate in c(.001,.005,.01)){ #(10^((-3):(-2)))){
    for (hidden in as.integer(c(5,10,20))){
      pnam <- paste0('plots/hoopi/nn_', 
                     hidden, 'hidden-', 
                     rate, 'rate-', 
                     iters, 'iters-',
                     '.pdf')
      message('\n*************************\nmaking `', pnam, '`...\n')
      danet <- train_(hidden, rate, iters)
      dagrid <- grid_(danet)
      plotg_(dagrid, fixed=TRUE)
      ggsave(pnam, width=4, height=4, units='in')
    }
  }
}






### DEV AREA FOR BAQSABALL -------
if (FALSE){

source('playaz.r')
dat <- hoop_clf
dat <- rbind(dat, 
             list(72, 160, 'fc'), list(90, 350, 'bc'),
             list(71, 165, 'bc'), list(90, 350, 'fc'))
# dat$x1 <- as.numeric(scale(dat$x1))
# dat$x2 <- as.numeric(scale(dat$x2))
ggplot2::qplot(dat$x1, dat$x2, color=dat$y, alpha=.5, size=4)

## UGH IT IS LIKE TOO GOD MAYBE?! ACCURACY 1...
lefftpack::import(train_, from='nn_phonxe.r')
menet <- train_(hidden=5, learn_rate=.05, iters=100)

lefftpack::import(grid_, from='nn_phonxe.r')
megrid <- grid_(menet)

lefftpack::import(plotg_, from='nn_phonxe.r')
plotg_(megrid, fixed=TRUE)

ggsave('blaowwie2.pdf', width=4, height=4, units='in')
}