options(stringsAsFactors = F)

# Lag helper funcs
source("../lag_funcs.R")

# Data ----
library(ggplot2)
library(dplyr)
library(sf)

housing <- read.csv("../Data/houses1990.csv") %>% 
  mutate(bedroomsAvg = bedrooms / households, 
         roomsAvg = rooms / households) %>% 
  select(-c("bedrooms", "rooms"))

# Transform to 'sf' object
housing.sf <- st_as_sf(housing, coords = c("longitude", "latitude"), crs = 4326)

unique_idx <- as.data.frame(st_coordinates(housing.sf), row.names = seq_along(housing.sf)) %>% 
  unique() %>% row.names() %>% as.numeric()

housing.sf <- housing.sf[unique_idx, ]
housing <- housing[unique_idx, ]

# CV: train/test splitting ----
library(mlr)
outer_n <- 5
inner_n <- 3
cvmethod <- "CV"
set.seed(123) # Set seeds
# Resampling description (outer folds)
rdesc <- makeResampleDesc(cvmethod, iters = outer_n)
# Regression task definition
spatial.task <- makeRegrTask(data = st_drop_geometry(housing.sf), target = "houseValue", 
                             coordinates = as.data.frame(st_coordinates(housing.sf)))
# Resampling instance (outer folds)
outer.rin <- makeResampleInstance(rdesc, task = spatial.task)
rm(rdesc, spatial.task)

# Generating inner folds
inner <- lapply(1:length(outer.rin$train.inds), function(i) {
  idx.train <- outer.rin$train.inds[[i]]
  idx.test <- outer.rin$test.inds[[i]]
  set.seed(123)
  rdesc <- makeResampleDesc(cvmethod, iters = inner_n)
  spatial.task <- makeRegrTask(data = st_drop_geometry(housing.sf[idx.train,]), target = "houseValue", 
                               coordinates = as.data.frame(st_coordinates(housing.sf[idx.train,])))
  inner.rin <- makeResampleInstance(rdesc, task = spatial.task)
  list(train.inds = inner.rin$train.inds, test.inds = inner.rin$test.inds)
})


# CV: evaluation ----
library(glmnet)
library(FNN)
library(ranger)
library(foreach)
library(parallel)
library(doParallel)

# Multi-core computing
cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)

res <- lapply(1:outer_n, function(oi) {
  # inner folds eval
  params <- 2:8 # 'mtry' candidates
  ex_pkgs <- c("sf", "FNN", "ModelMetrics", "caret", "ranger", "dplyr", "glmnet", "mlr")
  vars <- c(ls(), ls(envir = globalenv()))
  res.rmse <- lapply(1:length(params), function(i){
    cv.res <- foreach(cvi = 1:inner_n, .combine = "c", 
                      .packages = ex_pkgs, .export = vars) %dopar%{
                        hold_eval(param = params[i], lag.vec = c(5, 10, 15, 50), 
                                  train.id = inner[[oi]]$train.inds[[cvi]], 
                                  test.id = inner[[oi]]$test.inds[[cvi]], 
                                  target.var = "houseValue", data.sf = housing.sf, lasso.fold = 10, 
                                  spatial = TRUE)
                      }
    aggregate(unlist(cv.res), by = list(rep(1:4, times = inner_n)), mean)[3, 2]
  })
  print(paste0("Done: inner tuning of outer fold ", oi))
  flush.console()
  # outer eval
  rmse.outer <- unlist(hold_eval(param = params[which.min(res.rmse)], lag.vec = c(5, 10, 15, 50),
                                 train.id = outer.rin$train.inds[[oi]], test.id = outer.rin$test.inds[[oi]], 
                                 target.var = "houseValue", data.sf = housing.sf, lasso.fold = 10))
  print(paste0("Done: outer fold ", oi, ". RMSE: ", rmse.outer[3]))
  flush.console()
  return(rmse.outer)
})
stopCluster(cl)

# Average results from outer folds
avg.nested <- aggregate(unlist(res), by = list(rep(1:4, times = outer_n)), mean)[3, 2]


# Final lag model: tuning ----
# 5-fold CV
set.seed(456)
rdesc <- makeResampleDesc(cvmethod, iters = 5)
spatial.task <- makeRegrTask(data = st_drop_geometry(housing.sf), target = "houseValue", 
                             coordinates = as.data.frame(st_coordinates(housing.sf)))
tune.rin <- makeResampleInstance(rdesc, task = spatial.task)
rm(rdesc, spatial.task)

params <- 2:6 # 'mtry' candidates
ex_pkgs <- c("sf", "FNN", "ModelMetrics", "caret", "ranger", "dplyr", "glmnet", "mlr")

cl <- makeCluster(detectCores()-1)
registerDoParallel(cl)
tune.res <- sapply(1:length(params), function(i) {
  vars <- c(ls(), ls(envir = globalenv()))
  cv.res <- foreach(cvi = 1:outer_n, .combine = "c", 
                    .packages = ex_pkgs, .export = vars) %dopar%{
                      hold_eval(params[i], lag.vec = c(5, 10, 15, 50), 
                                train.id = tune.rin$train.inds[[cvi]], 
                                test.id = tune.rin$test.inds[[cvi]], 
                                target.var = "houseValue", data.sf = housing.sf, 
                                lasso.fold = 10, spatial = TRUE)
                    }
  # Averaging the testing RMSE
  aggregate(unlist(cv.res), by = list(rep(1:4, times = 5)), mean)[3, 2]
})
stopCluster(cl)

params[which.min(tune.res)]


# Final lag model: training ----
lags <- multi_lag(housing.sf$houseValue, as.matrix(st_coordinates(housing.sf)), 
                  k.vec = c(5, 10, 15, 50))
lag.coef <- lag_coeffs(lags, housing.sf$houseValue, as.data.frame(st_coordinates(housing.sf)), 
                       folds = 10)
lagsub <- lags[, lag.coef != 0, drop=FALSE]
# Combine the original features with lag features
housing.final <- dplyr::bind_cols(housing.sf, lagsub)

set.seed(1111)
# Use the parameter setting with the lowest RMSE
final.model <- ranger::ranger(x = dplyr::select(st_drop_geometry(housing.final), -c("houseValue")), 
                              y = housing.final$houseValue, 
                              mtry = params[which.min(tune.res)], num.trees = 200, num.threads = 2)

pred <- predict(final.model, dplyr::select(st_drop_geometry(housing.final), -c("houseValue"))) %>% ranger::predictions()
rmse.train <- ModelMetrics::rmse(actual = housing.final$houseValue, predicted = pred)


# Final lag model: Moran ----
library(spdep)
# Creating neighboring list
nb <- FNN::get.knn(as.matrix(st_coordinates(housing.sf)), 5)$nn.index %>% 
  apply(1, list) %>% unlist(recursive = F)
attr(nb, "class") <- "nb"
# nb <- knearneigh(coordinates(as(housing.sf, "Spatial")), k = 5) %>% knn2nb()
# Moran's I (1000 Monte-Carlo simulation)
moran.mc(housing.sf$houseValue - pred, nb2listw(nb), nsim = 999)



