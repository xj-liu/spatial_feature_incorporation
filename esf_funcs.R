# LASSO fitting of eigenvectors 
# @eigens.df: data frame of the eigenvectors derived from 'spmoran' package
# @target: a numeric vector of the target variable
# @coor: coordinates of the data samples 
# @folds: the CV folds used for lasso
# @cvmethod: "CV" indicating the type of cross-validation
# This function returns the LASSO coefficients corresponding to the eigenvector features
esf_coeffs <- function(eigens.df, target, coor, folds = 20, cvmethod = 'CV'){
  d <- cbind(eigens.df, target = target) %>% as.data.frame()
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
  cvfit <- cv.glmnet(as.matrix(eigens.df), y = target, type.measure = "mse", nfolds = folds, alpha = 1)
  coeff_mtx <- coef(cvfit, s = "lambda.1se")
  coeffs <- rep(0, length.out = ncol(eigens.df))
  coeffs[coeff_mtx@i[-1]] <- coeff_mtx@x[-1]
  return(coeffs)
}


# Evaluate a random forest model with eigenvector features based on a provided hyper-parameter and the training/testing split
# @param: 'mtry' parameter used for random forest
# @train.id: indices of the training samples
# @test.id: indices of the testing samples
# @target.var: column name of the target variable in the data frame
# @data.sf: the data ('sf' data frame)
# @lasso.cv: "CV" indicating the type of cross-validation for LASSO
# @lasso.fold: the CV folds used for LASSO
# @prox: if the eigenvectors should be approximated? If true, use 'meigen_f' in 'spmoran' package; if false, use 'meigen' instead.
# @model: type of kernel to model spatial dependence (see 'meigen' function of 'spmoran' package)
# @spatial: if the eigenvector features are calculated? If false, the parameters 'lasso.cv', 'lasso.fold', 'prox', 'model' are ignored.
# This function returns a data frame of the RMSE and R-squared values corresponding to the training and testing sets
hold_eval <- function(param, train.id, test.id, target.var, data.sf, 
                      lasso.cv = "CV", lasso.fold = 20, prox = TRUE, 
                      model = "exp", spatial = TRUE){

  train.sf <- data.sf[train.id,]
  if(spatial){
    if(prox){
      train.esfs <- meigen_f(as.matrix(st_coordinates(train.sf)), model = model)
    } else {
      train.esfs <- meigen_full(as.matrix(st_coordinates(train.sf)), model = model)
    }
    # save lag coefficients
    esf.coef <- esf_coeffs(train.esfs$sf, train.sf[[target.var]], 
                           coor = as.data.frame(st_coordinates(train.sf)), 
                           folds = lasso.fold, cvmethod = lasso.cv)
    
    train.esfsub <- as.data.frame(train.esfs$sf)[, esf.coef > 1e-3, drop=FALSE]
    if(!is.null(train.esfsub))
      train.sf <- dplyr::bind_cols(train.sf, train.esfsub)
  }
  
  
  test.sf <- data.sf[test.id,]
  if(spatial){
    test.esfs <- meigen0(train.esfs, as.matrix(st_coordinates(test.sf)))
    test.esfsub <- as.data.frame(test.esfs$sf)[, esf.coef > 1e-3, drop=FALSE]
    if(!is.null(test.esfsub))
      test.sf <- dplyr::bind_cols(test.sf, test.esfsub)
  }
  
  m <- ranger::ranger(x = st_drop_geometry(train.sf)[, colnames(st_drop_geometry(train.sf)) != target.var], 
                      y = train.sf[[target.var]], 
                      mtry = param, num.trees = 200, num.threads = 2)
  
  pred.train <- predict(m, dplyr::select(st_drop_geometry(train.sf), -all_of(target.var))) %>% ranger::predictions()
  pred.test <- predict(m, dplyr::select(st_drop_geometry(test.sf), -all_of(target.var))) %>% ranger::predictions()
  return(list(model.rmse = ModelMetrics::rmse(actual = train.sf[[target.var]], predicted = pred.train), 
              model.r2 = caret::postResample(as.data.frame(train.sf[[target.var]]), pred.train)[2],
              rmse = ModelMetrics::rmse(actual = test.sf[[target.var]], predicted = pred.test),
              r2 = caret::postResample(as.data.frame(test.sf[[target.var]]), pred.test)[2]))
}




