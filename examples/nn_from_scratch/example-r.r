###############################################################################
### TODO: 
###   - streamline below w original nn.r/nn_phonxe.r scripts 
###   - see subfolder notes for further instructions 
### 
### original starting point/source: 
###   - https://selbydavid.com/2018/01/09/neural-network
###############################################################################


##### MAIN FUNXE ############################################################## 

### 1. sigmoid func --------------
sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}
### 2. feed fwd ------------------
feedfwd <- function(x, w1, w2){
  z1 <- cbind(1, x) %*% w1
  h <- sigmoid(z1)
  z2 <- cbind(1, h) %*% w2
  return(list(output=sigmoid(z2), h=h))
}
### 3. backprop ------------------
backprop <- function(x, y, y_hat, w1, w2, h, learn_rate){
  dw2 <- t(cbind(1, h)) %*% (y_hat - y)          # [FILL IN NAME OF THIS STEP] 
  dh <- (y_hat - y) %*% t(w2[-1, , drop=FALSE])  # [FILL IN NAME OF THIS STEP] 
  dw1 <- t(cbind(1, x)) %*% (h * (1 - h) * dh)   # [FILL IN NAME OF THIS STEP] 
  w1 <- w1 - learn_rate * dw1 # update w1 
  w2 <- w2 - learn_rate * dw2 # update w2 
  return(list(w1=w1, w2=w2))  # return updated weights 
}
### 4. train routine -------------
train <- function(x, y, hidden=5, learn_rate=1e-2, iters=1e4){
  d <- ncol(x) + 1                         # `d`+1-many rows for `w1` 
  w1 <- matrix(rnorm(d*hidden), d, hidden) # init w1, `d*hidden`-many vals 
  w2 <- matrix(rnorm(hidden + 1))          # init w2, `hidden+1`-many vals 
  for (i in 1:iters){                      # for each iter i:
    ff <- feedfwd(x, w1, w2)                 # feedfwd input `x` w `w1, w2` 
    bp <- backprop(x, y,                     # backprop w `x`, `y`:
                   y_hat=ff$output,            # preds: ff output 
                   w1, w2,                     # current weights `w1`, `w2` 
                   h=ff$h,                     # `h` = feedfwd `h` 
                   learn_rate=learn_rate)      # with `learn_rate` 
    w1 <- bp$w1; w2 <- bp$w2                 # update weights w backprop vals 
    if (i %% as.integer(iters/5) == 0)       # progress msg 
      message("at i=", i)
  }
  return(list(output=ff$output, w1=w1, w2=w2)) # return final ff and weights 
}


##### SOME UTILS ############################################################## 
makemedat_ <- function(n, cycles, x1_sd, x2_sd, seed=6933){
  set.seed(seed)
  spirals <- mlbench::mlbench.spirals(n, cycles, sd=0)
  dat <- dplyr::data_frame(
    x1 = spirals$x[,1] + rnorm(n/2, sd=x1_sd), 
    x2 = spirals$x[,2] + rnorm(n/2, sd=x2_sd), 
    y  = as.factor(ifelse(spirals$classes=="1", "yes", "no")))
  print(ggplot2::qplot(dat$x1, dat$x2, color=dat$y, 
                       alpha=.5, size=4, show.legend=FALSE))
  print(summary(glm(y ~ x1 + x2, family=binomial, data=dat)))
  return(dat)
}



##### GO THRU TRAIN ROUTINE STEP BY STEP ###################################### 
dat <- makemedat_(n=100, cycles=.5, x1_sd=.05, x2_sd=.05, seed=6933); print(dat)

x <- data.matrix(dat[, c('x1','x2')])
y <- dat[, 'y']=='yes' 

hidden <- 5       # num columns 
learn_rate <- .1  # [**FILL IN DESC**] 
iters <- 10       # num train iters 


### STEP 1: get number of rows for hidden layers[??] 
(d <- ncol(x) + 1)                         # `d`+1-many rows for `w1` 

### STEP 2: initialize weights 
(w1 <- matrix(rnorm(d*hidden), d, hidden)) # init w1, `d*hidden`-many vals 
(w2 <- matrix(rnorm(hidden + 1)))          # init w2, `hidden+1`-many vals 


for (i in 1:iters){                        # for each iter i: 
  ff <- feedfwd(x, w1, w2)                   # feedfwd input `x` w `w1, w2` 
  bp <- backprop(x, y,                       # backprop `x`, `y`:
                 y_hat=ff$output,              # preds: ff output 
                 w1, w2,                       # current weights `w1`, `w2` 
                 h=ff$h,                       # `h` = feedfwd `h` 
                 learn_rate=learn_rate)        # with `learn_rate` 
  w1 <- bp$w1; w2 <- bp$w2                 # update weights w backprop vals 
}

# final ff and weights 
(out <- list(output=ff$output, w1=w1, w2=w2))

lefftpack::lazy_setup()
dplyr::data_frame(y=y[,1], out=out$output[,1]) %>% 
  group_by(y) %>% summarize(
    min = min(out), max = max(out), 
    mean = mean(out), median = median(out))


##### in/out dims for each func 

# if dim(x) = 100x2, dim(w1) = 5x3, dim(w2) = 6x1: 
#   dim(feedfwd(x, w1, w2)) = 

# matmult %*%: 
#  e.g. 100x3 * 3x5 ==> 100x5 






##### CLASSIFY SOME DATA ###################################################### 
lefftpack::lazy_setup()
dat <- makemedat_(n=100, cycles=.5, x1_sd=.05, x2_sd=.05, seed=6933)
# train(x = data.matrix(dat[, c('x1','x2')]), y = dat[, 'y']=='yes', 
#       hidden=2, learn_rate=.01, iters=1e3)

x <- data.matrix(dat[, c('x1','x2')])
y <- dat[, 'y']=='yes' # cbind(x, y)[1:10, ]

nn <- train(x, y, hidden=2, learn_rate=.1, iters=100)





##### BREAK DOWN `nn_phonxe.r` ################################################
################################################ BREAK DOWN `nn_phonxe.r` ##### 
##### BREAK DOWN `nn_phonxe.r` ################################################
################################################ BREAK DOWN `nn_phonxe.r` ##### 
##### BREAK DOWN `nn_phonxe.r` ################################################
################################################ BREAK DOWN `nn_phonxe.r` ##### 




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


