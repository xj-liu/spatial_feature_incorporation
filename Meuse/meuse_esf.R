options(stringsAsFactors = F)
source("../esf_funcs.R")

# Data ----
library(ggplot2)
library(dplyr)
library(sf)
data("meuse", package = "sp")
# remove NA rows (3)
meuse.sf <- st_as_sf(meuse, coords = c("x", "y"), crs = 28992) %>% na.omit() %>% 
  st_transform(meuse, crs=4326)
rm(meuse)
meuse.sf <- select(meuse.sf, -c("cadmium","copper","lead", "dist.m"))


# Spatial CV: train/test splitting ----
library(mlr)
outer_n <- 5
inner_n <- 3
cvmethod <- "CV"

set.seed(123) # Set seeds
# Resampling description (outer folds)
rdesc <- makeResampleDesc(cvmethod, iters = outer_n)
# Regression task definition
spatial.task <- makeRegrTask(data = st_drop_geometry(meuse.sf), target = "zinc", 
                             coordinates = as.data.frame(st_coordinates(meuse.sf)))
# Resampling instance (outer folds)
outer.rin <- makeResampleInstance(rdesc, task = spatial.task)
rm(rdesc, spatial.task)

# Generating inner folds
inner <- lapply(1:length(outer.rin$train.inds), function(i) {
  idx.train <- outer.rin$train.inds[[i]]
  idx.test <- outer.rin$test.inds[[i]]
  set.seed(123)
  rdesc <- makeResampleDesc(cvmethod, iters = inner_n)
  spatial.task <- makeRegrTask(data = st_drop_geometry(meuse.sf[idx.train,]), target = "zinc", 
                               coordinates = as.data.frame(st_coordinates(meuse.sf[idx.train,])))
  inner.rin <- makeResampleInstance(rdesc, task = spatial.task)
  list(train.inds = inner.rin$train.inds, test.inds = inner.rin$test.inds)
})


# CV: evaluation ----
library(glmnet)
library(FNN)
library(ranger)
library(spmoran)

res <- lapply(1:outer_n, function(oi) {
  res.rmse <- NULL
  params <- 2:7
  # inner folds eval
  for(i in 1:length(params)){
    cv.res <- lapply(1:inner_n, function(cvi) {
      hold_eval(param = params[i],
                train.id = inner[[oi]]$train.inds[[cvi]], test.id = inner[[oi]]$test.inds[[cvi]], 
                target.var = "zinc", data.sf = meuse.sf, lasso.fold = 10, prox = FALSE)
    })
    res.rmse[i] <- aggregate(unlist(cv.res), by = list(rep(1:4, times = inner_n)), mean)[3, 2]
  }
  print(paste0("Done: inner tuning of outer fold ", oi))
  flush.console()
  # outer eval
  rmse.outer <- unlist(hold_eval(param = params[which.min(res.rmse)],
                                 train.id = outer.rin$train.inds[[oi]], test.id = outer.rin$test.inds[[oi]], 
                                 target.var = "zinc", data.sf = meuse.sf, lasso.fold = 10, prox = FALSE))
  print(paste0("Done: outer fold ", oi, ". RMSE: ", rmse.outer[3]))
  flush.console()
  return(rmse.outer)
})
# Average results from outer folds
avg.nested <- aggregate(unlist(res), by = list(rep(1:4, times = outer_n)), mean)[3, 2]


# Final non-spatial model: tuning ----
# 5-fold CV
set.seed(456)
rdesc <- makeResampleDesc(cvmethod, iters = 5)
spatial.task <- makeRegrTask(data = st_drop_geometry(meuse.sf), target = "zinc", 
                             coordinates = as.data.frame(st_coordinates(meuse.sf)))
tune.rin <- makeResampleInstance(rdesc, task = spatial.task)
rm(rdesc, spatial.task)

params <- 2:7
tune.res <- sapply(1:length(params), function(i) {
  cv.res <- lapply(1:5, function(cvi) {
    hold_eval(param = params[i], 
              train.id = tune.rin$train.inds[[cvi]], test.id = tune.rin$test.inds[[cvi]], 
              target.var = "zinc", data.sf = meuse.sf, lasso.fold = 10, 
              prox = FALSE, spatial = T)
  })
  aggregate(unlist(cv.res), by = list(rep(1:4, times = 5)), mean)[3, 2]
})
params[which.min(tune.res)]

# Final model: training ----
train.esfs <- meigen_full(as.matrix(st_coordinates(meuse.sf)))
esf.coef <- esf_coeffs(train.esfs$sf, meuse.sf$zinc, 
                       as.data.frame(st_coordinates(meuse.sf)), folds = 10)
esfsub <- as.data.frame(train.esfs$sf)[, esf.coef != 0, drop=FALSE]
# Combine the original features with esf eigenvectors
meuse.final <- dplyr::bind_cols(meuse.sf, esfsub)

# Use the parameter setting with the lowest RMSE
set.seed(1111)
final.model <- ranger::ranger(x = dplyr::select(st_drop_geometry(meuse.final), -c("zinc")), 
                              y = meuse.sf$zinc, 
                              mtry = params[which.min(tune.res)], num.trees = 200, num.threads = 2)
pred <- predict(final.model, dplyr::select(st_drop_geometry(meuse.final), -c("zinc"))) %>% ranger::predictions()
rmse.train <- ModelMetrics::rmse(actual = meuse.final$zinc, predicted = pred)


# Final model: Moran ----
library(spdep)
# Creating neighboring list
nb <- knearneigh(coordinates(as(meuse.sf, "Spatial")), k = 5) %>% knn2nb()
# Moran's I (1000 Monte-Carlo simulation)
mc <- moran.mc(meuse.sf$zinc - pred, nb2listw(nb), nsim = 999)







