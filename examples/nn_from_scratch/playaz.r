lefftpack::lazy_setup()

# # first read the data + fix colnames
# dat <- read.csv("playerbiostats.csv") %>% select(
#   id=PLAYER_ID, name=PLAYER_NAME, age=AGE, ht=PLAYER_HEIGHT_INCHES, 
#   wt=PLAYER_WEIGHT, gp=GP, pts=PTS, reb=REB, ast=AST) %>% unique() %>% 
#   mutate(id = sprintf("id%07d", id)) %>% arrange(desc(ht)) %>% mutate(pos = "")
# write to disk + add positions by hand 
# # not run write.csv(dat, "temp.csv", row.names=F)

# read in labeled data + compute per-game stats + toss count cols 
dat <- dplyr::as_data_frame(read.csv("playaz.csv")) %>% 
  mutate(ppg = pts/gp, rpg = reb/gp, apg = ast/gp) %>% 
  mutate_if(is.double, round, digits=1) %>% 
  select(-pts, -reb, -ast) %>% 
  mutate(pos = factor(pos, levels=c("pg","sg","sf","pf","c")))

sapply(dat, function(col) sum(is.na(col)))
dat %>% group_by(pos) %>% summarize_at(vars(age,ht,wt,gp,ppg,rpg,apg), mean)
summarize_at(group_by(dat, pos), vars(age,ht,wt,gp,ppg,rpg,apg), mean)
sapply(c("ht", "wt", "gp", "ppg", "rpg", "apg"), function(cname){
  round(tapply(dat[[cname]], list(dat$pos), mean), 1)})

hoop_clf <- dplyr::data_frame(
  x1 = dat$ht, x2 = dat$wt, 
  y = as.factor(ifelse(dat$pos %in% c("pg","sg"), "bc", "fc")))

rm(dat)
