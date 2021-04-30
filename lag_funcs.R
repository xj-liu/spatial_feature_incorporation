# Generate multiple spatial lag features
# @target: a numeric vector of the target variable
# @coor: coordinates of the data samples with known target values
# @k.vec: the 'k' values used for generating the sptial weight matrix
# @dist.pow: the power of inverse distance 
# @query.coor: the coordinates of the samples with unknown target values. If provided, the function returns the spatial lag features of the query locations. If not provided, the lag features of the 'coor' locations are returned.
# The function returns the spatail lag featues
multi_lag <- function(target, coor, k.vec, dist.pow = 0, query.coor = NULL){
  k_lag_generate <- function(k, vals, locs, w.pow = 0, query.locs = NULL){
    if(is.null(query.locs)){
      neigh <- FNN::get.knn(locs, k)
    } else{
      neigh <- FNN::get.knnx(locs, query.locs, k)
    }
    
    if(w.pow == 0){
      lag <- apply(neigh$nn.index, 1, function(x) mean(vals[x]))
    } else{
      frac <- 1/neigh$nn.dist^w.pow # divided by 0
      w <- sweep(frac, 1, rowSums(frac), "/")
      lag <- rowSums(t(apply(neigh$nn.index, 1, function(x) vals[x])) * w)
    }
    return(lag)
  }
  
  lag.ls <- lapply(k.vec, k_lag_generate, vals = target, locs = coor, w.pow = dist.pow, 
                   query.locs = query.coor)
  lags <- as.data.frame(lag.ls, col.names = paste("lag_k", k.vec, "_d", dist.pow, sep = ""))
  return(lags)
}

# LASSO fitting of spatial lag features 
# @lags.df: data frame of the spatial lag features derived from 'multi_lag' function
# @target: a numeric vector of the target variable
# @coor: coordinates of the data samples 
# @folds: the CV folds used for lasso
# @cvmethod: "CV" indicating the type of cross-validation
# This function returns the LASSO coefficients corresponding to the spatial lag features
lag_coeffs <- function(lags.df, target, coor, folds = 20, cvmethod = 'CV'){
  d <- cbind(lags.df, target = target)
  set.seed(123) # Set seeds
  # Resampling description (outer folds)
  rdesc <- makeResampleDesc(cvmethod, iters = folds)
  # Regression task definition
  spatial.task <- makeRegrTask(data = d, target = "target", 
                               coordinates = coor)
  # Resampling instance (outer folds)
  rin <- makeResampleInstance(rdesc, task = spatial.task)
  rm(rdesc, spatial.task)
  fold.idx <- seq(length(target))
  for(i in seq(folds)){
    fold.idx[seq(length(target)) %in% rin$test.inds[[i]]] <- i
  }
  cvfit <- cv.glmnet(as.matrix(lags.df), y = target, type.measure = "mse", 
                     foldid = fold.idx, alpha = 1)
  coeff_mtx <- coef(cvfit, s = "lambda.1se")
  coeffs <- rep(0, length.out = ncol(lags.df))
  coeffs[coeff_mtx@i[-1]] <- coeff_mtx@x[-1]
  #lag_agg <- rowSums(sweep(lags.df, 2, coeffs, "*"))
  return(coeffs)
}


# Evaluate a random forest model with spatial lag features based on a provided hyper-parameter and the training/testing split
# @lag.vec: the 'k' values used for generating the sptial weight matrix
# @train.id: indices of the training samples
# @test.id: indices of the testing samples
# @target.var: column name of the target variable in the data frame
# @data.sf: the data ('sf' data frame)
# @lasso.cv: "CV" indicating the type of cross-validation for LASSO
# @lasso.fold: the CV folds used for LASSO
# @spatial: if the spatial lag features are calculated? If false, the parameters 'lag.vec', 'lasso.cv', 'lasso.fold'are ignored.
# This function returns a data frame of the RMSE and R-squared values corresponding to the training and testing sets
hold_eval <- function(param, lag.vec = c(5, 10, 15, 50), train.id, test.id, 
                      target.var, data.sf, 
                      lasso.cv = "CV", lasso.fold = 20, 
                      spatial = TRUE){
  # train.idx <- train.id[[fold]]
  train.sf <- data.sf[train.id,]
  if(spatial){
    train.lags <- multi_lag(train.sf[[target.var]], as.matrix(st_coordinates(train.sf)), 
                            k.vec = lag.vec)
    # save lag coefficients
    lag.coef <- lag_coeffs(train.lags, train.sf[[target.var]], 
                           coor = as.data.frame(st_coordinates(train.sf)), 
                           folds = lasso.fold, cvmethod = lasso.cv)
    
    train.lagsub <- train.lags[, lag.coef > 1e-3, drop=FALSE]
    train.sf <- dplyr::bind_cols(train.sf, train.lagsub)  
    # print(colnames(train.lagsub))
    # flush.console()
  }

  
  test.sf <- data.sf[test.id,]
  if(spatial){
    test.lags <- multi_lag(train.sf[[target.var]], as.matrix(st_coordinates(train.sf)), 
                           k.vec = lag.vec, query.coor = as.matrix(st_coordinates(test.sf)))
    test.lagsub <- test.lags[, lag.coef > 1e-3, drop=FALSE]
    test.sf <- dplyr::bind_cols(test.sf, test.lagsub)
    # print("Test lag calculation done.")
    # flush.console()
  }
  m <- ranger::ranger(x = st_drop_geometry(train.sf)[, colnames(st_drop_geometry(train.sf)) != target.var], 
                      y = train.sf[[target.var]], 
                      mtry = param, num.trees = 200, num.threads = 2)
  # build trees in parallel
  # m <- foreach(ntree = rep(50, 4), .combine = randomForest::combine,
  #              .multicombine=TRUE, .packages = 'randomForest') %dopar%{
  #                randomForest(as.formula(paste(target.var, "~ .")), 
  #                             data = st_drop_geometry(train.sf), mtry = param, ntree = ntree)
  #              }
  
  pred.train <- predict(m, dplyr::select(st_drop_geometry(train.sf), -all_of(target.var))) %>% ranger::predictions()
  pred.test <- predict(m, dplyr::select(st_drop_geometry(test.sf), -all_of(target.var))) %>% ranger::predictions()
  return(list(model.rmse = ModelMetrics::rmse(actual = train.sf[[target.var]], predicted = pred.train), 
              model.r2 = caret::postResample(as.data.frame(train.sf[[target.var]]), pred.train)[2],
              rmse = ModelMetrics::rmse(actual = test.sf[[target.var]], predicted = pred.test),
              r2 = caret::postResample(as.data.frame(test.sf[[target.var]]), pred.test)[2]))
}


