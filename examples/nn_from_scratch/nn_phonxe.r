### original gahy's phonxe ----------------------------------------------------

# sigmoid (activation func(?))
sigmoid <- function(x){ 1 / (1 + exp(-x)) }

# feed forward 
feedfwd <- function(x, w1, w2) {
  z1 <- cbind(1, x) %*% w1
  h <- sigmoid(z1)
  z2 <- cbind(1, h) %*% w2
  list(output=sigmoid(z2), h=h)
}

# backpropogation 
backprop <- function(x, y, y_hat, w1, w2, h, learn_rate) {
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)
  dh  <- (y_hat - y) %*% t(w2[-1, , drop=FALSE])
  dw1 <- t(cbind(1, x)) %*% (h * (1 - h) * dh)
  
  w1 <- w1 - learn_rate * dw1
  w2 <- w2 - learn_rate * dw2
  
  list(w1=w1, w2=w2)
}

# training routine 
train <- function(x, y, hidden=5, learn_rate=1e-2, iters=1e4) {
  d <- ncol(x) + 1
  w1 <- matrix(rnorm(d * hidden), d, hidden)
  w2 <- matrix(rnorm(hidden + 1))
  for (i in 1:iters) {
    ff <- feedfwd(x, w1, w2)
    bp <- backprop(x, y,
                        y_hat=ff$output,
                        w1, w2,
                        h=ff$h,
                        learn_rate=learn_rate)
    w1 <- bp$w1; w2 <- bp$w2
    if (i %% as.integer(iters/5) == 0) message("at i=", i)
  }
  list(output=ff$output, w1=w1, w2=w2)
}



### me wrappers around original gaihye's phonxe -------------------------------
train_ <- function(...){
  require(magrittr)
  train(data.matrix(dat[,c('x1','x2')]), dat[,c('y')]=='yes', ...) %T>% 
  { .$output %>% (function(x) mean((x > .5) == (dat$y=='yes'))) %>%
      message('prop correct(?!) preds: `', ., '`') }
}

grid_ <- function(net, expand=.5, by=.05){
  grid <- expand.grid(  # was expand=1 by=.25 originally
    x1=seq(min(dat$x1)-expand, max(dat$x1)+expand, by=by), 
    x2=seq(min(dat$x2)-expand, max(dat$x2)+expand, by=by))
  
  ff_grid <- feedfwd(
    x=data.matrix(grid[, c('x1', 'x2')]),
    w1=net[['w1']], w2=net[['w2']])
  
  grid$y <- factor((ff_grid$output > .5) * 1,
                   labels=levels(dat$y))
  
  attributes(grid) <- c( # experimentalze!!
    attributes(grid), acc=mean((net$output > .5)==(dat$y=='yes')))
  
  # print(head(grid, n=4))
  # message('grid w dims: `', list(dim(grid)), '`, first few rows above')
  return(grid)
}

plotg_ <- function(grid, fixed=TRUE){
  require(ggplot2)
  ggplot(dat) + aes(x1, x2, colour=y) +
    geom_point(alpha=.75, size=rel(4)) +
    scale_x_continuous(expand=c(0,0)) + 
    scale_y_continuous(expand=c(0,0)) + 
    annotate("label", x=min(dat$x1), y=max(dat$x2), 
             hjust=.35, vjust=-.5, fill='darkgray', color='black', alpha=1,
             label=paste0('accuracy: ', round(attributes(grid)$acc, 2))) +
    geom_point(data=grid, size=rel(2), alpha=.15) +
    labs(x=expression(x[1]), y=expression(x[2])) + 
    if (fixed) coord_fixed() else coord_cartesian()
}


### me lil fonq to load da datarre --------------------------------------------
makemedat_ <- function(n, cycles, x1_sd, x2_sd, seed=6933){
  set.seed(seed)
  spirals <- mlbench::mlbench.spirals(n, cycles, sd=0)
  dat <- dplyr::data_frame(
    x1 = spirals$x[,1] + rnorm(100, sd=x1_sd), 
    x2 = spirals$x[,2] + rnorm(100, sd=x2_sd), 
    y  = as.factor(ifelse(spirals$classes=="1", "yes", "no")))
  print(ggplot2::qplot(dat$x1, dat$x2, color=dat$y, 
                       alpha=.5, size=4, show.legend=FALSE))
  print(summary(glm(y ~ x1 + x2, family=binomial, data=dat)))
  return(dat)
}



# this gets clunky but wd be kinda naice 
# nn <- list(
#   sigmoid=sigmoid, feedfwd=feedfwd, backprop=backprop, train=train)
# rm(list=c('sigmoid', 'feedfwd', 'backprop', 'train'))


### load the data w/o miphonqqe
# set.seed(6933)
# spirals <- mlbench::mlbench.spirals(n=200, cycles=.6, sd=0)
# 
# dat <- dplyr::data_frame(
#   x1 = spirals$x[,1] + rnorm(100, sd=x1_sd), 
#   x2 = spirals$x[,2] + rnorm(100, sd=x2_sd), 
#   y  = as.factor(ifelse(spirals$classes=="1", "yes", "no")))
# 
# print(ggplot2::qplot(
#   dat$x1, dat$x2, color=dat$y, alpha=.5, size=4, show.legend=FALSE))
# print(summary(glm(y ~ x1 + x2, family=binomial, data=dat)))



