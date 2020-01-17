  rm(list = ls())
  # install.packages("randomForest")
  # install.packages("adabag")
  # install.packages("gbm")
  library(tree)
  library(randomForest)
  library(adabag)
  library(gbm)
  library(caret)
  library(MASS)
  
  ## train/test partition
  set.seed(23)
  n <- nrow(iris)
  indtrain <- sample(1:n, round(0.75*n))  # indices for train
  indtest <- setdiff(1:n, indtrain)  # indices for test
  
  ## Single Tree:
  t <- tree(Species ~., iris, subset = indtrain, control = tree.control(length(indtrain), mincut = 1, minsize = 2, mindev = 0))
  ## Prediction for test
  pred.t.test <- predict(t, iris[indtest, ], type = "class")
  ## Prediction for train
  pred.t.train <- predict(t, iris[indtrain, ], type = "class")
  ## Accuracy
  c(sum(diag(table(pred.t.test, iris$Species[indtest]))) / length(indtest), sum(diag(table(pred.t.train, iris$Species[indtrain]))) / length(indtrain))
  
  ## Bagging: Random Forests
  rf <- randomForest(Species ~., iris , subset = indtrain, ntree = 500)
  ## Prediction for test
  pred.rf.test <- predict(rf, iris[indtest, ])
  ## Prediction for train
  pred.rf.train <- predict(rf, iris[indtrain, ])
  ## Accuracy
  c(sum(diag(table(pred.rf.test, iris$Species[indtest]))) / length(indtest), sum(diag(table(pred.rf.train, iris$Species[indtrain]))) / length(indtrain))
  
  ## Boosting: Adaptive Boosting (AdaBoost)
  ## 20 trees (mfinal)
  ab <- boosting(Species ~., iris[indtrain, ], mfinal = 20, control=rpart.control(minsplit = 2, minbucket = 1, cp = 0.01))  
  ## Prediction for test
  pred.ab.test <- predict(ab, iris[indtest, ])
  ## Prediction for train
  pred.ab.train <- predict(ab, iris[indtrain, ])
  ## Accuracy
  c(sum(diag(table(pred.ab.test$class, iris$Species[indtest]))) / length(indtest), sum(diag(table(pred.ab.train$class, iris$Species[indtrain]))) / length(indtrain))
  
  ## Boosting: Gradient Boosting
  gb <- gbm(Species~., data=iris[indtrain, ], n.trees=1000, interaction.depth=20, shrinkage = 0.01)
  ## Prediction for test
  pred.gb.test <- predict(object = gb, newdata = iris[indtest, ], n.trees = 1000, type = "response")
  ## Prediction for train
  pred.gb.train <- predict(object = gb, newdata = iris[indtrain, ], n.trees = 1000, type = "response")
  ## Accuracy
  c(sum(diag(table(attributes(pred.gb.test)$dimnames[[2]][apply(pred.gb.test, FUN = which.max, MARGIN = 1)], iris$Species[indtest]))) / length(indtest), sum(diag(table(attributes(pred.gb.test)$dimnames[[2]][apply(pred.gb.train, FUN = which.max, MARGIN = 1)], iris$Species[indtrain]))) / length(indtrain))
  
  gb.cv <- gbm(Species~., data=iris[indtrain, ], n.trees=1000, interaction.depth=20, shrinkage = 0.01, cv.folds = 4)
  ntree_opt_cv <- gbm.perf(gb.cv, method = "cv")
  ntree_opt_oob <- gbm.perf(gb.cv, method = "OOB")
  
  print(ntree_opt_cv)
  print(ntree_opt_oob)
  
  gb <- gbm(Species~., data=iris[indtrain, ], n.trees=ntree_opt_cv, interaction.depth=20, shrinkage = 0.01)
  print(gb)
  summary(gb)
  ## Prediction for test
  pred.gb.test <- predict(object = gb, newdata = iris[indtest, ], n.trees = ntree_opt_cv, type = "response")
  ## Prediction for train
  pred.gb.train <- predict(object = gb, newdata = iris[indtrain, ], n.trees = ntree_opt_cv, type = "response")
  ## Accuracy
  c(sum(diag(table(attributes(pred.gb.test)$dimnames[[2]][apply(pred.gb.test, FUN = which.max, MARGIN = 1)], iris$Species[indtest]))) / length(indtest), sum(diag(table(attributes(pred.gb.test)$dimnames[[2]][apply(pred.gb.train, FUN = which.max, MARGIN = 1)], iris$Species[indtrain]))) / length(indtrain))
  
  #############################################################################
  rm(list = ls())
  # install.packages("randomForest")
  # install.packages("adabag")
  # install.packages("gbm")
  #### Installing XGBoost - Recommended
  # install.packages("drat", repos="https://cran.rstudio.com")
  # drat:::addRepo("dmlc")
  # install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
  #### Installing XGBoost - from CRAN
  # install.packages("xgboost")
  library(tree)
  library(randomForest)
  library(adabag)
  library(gbm)
  library(caret)
  library(MASS)
  library(xgboost)
  
  load("/home/sixto/Dropbox/M1966_DataMining/datasets/meteo.RData")
  ## Keeping the first 10 years (3650 days) for this example
  n <- 3650
  y <- y[1:n]
  x <- x[1:n, ]
  
  ## train/test partition
  set.seed(23)
  indtrain <- sample(1:n, round(0.75*n))  # indices for train
  indtest <- setdiff(1:n, indtrain)  # indices for test
  
  ## binary occurrence (1/0)
  occ <- y
  occ[which(y < 1)] <- 0
  occ[which(y >= 1)] <- 1
  
  ## dataframe for occurrence
  df.occ <- data.frame(y.occ = as.factor(occ), predictors = x)
  
  ## Single Tree:
  t <- tree(y.occ ~., df.occ, subset = indtrain, control = tree.control(length(indtrain), mincut = 1, minsize = 2, mindev = 0))
  ## Prediction for test
  pred.t.test <- predict(t, df.occ[indtest, ], type = "class")
  ## Prediction for train
  pred.t.train <- predict(t, df.occ[indtrain, ], type = "class")
  ## Accuracy
  c(sum(diag(table(pred.t.test, df.occ$y.occ[indtest]))) / length(indtest), sum(diag(table(pred.t.train, df.occ$y.occ[indtrain]))) / length(indtrain))
  confusionMatrix(pred.t.test, df.occ$y.occ[indtest])
  confusionMatrix(pred.t.train, df.occ$y.occ[indtrain])
  
  ## Bagging: Random Forests
  rf <- randomForest(y.occ ~., df.occ, subset = indtrain, ntree = 500)
  ## Prediction for test
  pred.rf.test <- predict(rf, df.occ[indtest, ])
  ## Prediction for train
  pred.rf.train <- predict(rf, df.occ[indtrain, ])
  ## Accuracy
  c(sum(diag(table(pred.rf.test, df.occ$y.occ[indtest]))) / length(indtest), sum(diag(table(pred.rf.train, df.occ$y.occ[indtrain]))) / length(indtrain))
  confusionMatrix(pred.rf.test, df.occ$y.occ[indtest])
  confusionMatrix(pred.rf.train, df.occ$y.occ[indtrain])
  
  ## Boosting: Adaptive Boosting (AdaBoost)
  ## 20 trees (mfinal)
  ab <- boosting(y.occ ~., df.occ[indtrain, ], mfinal = 20)
  ## Prediction for test
  pred.ab.test <- predict(ab, df.occ[indtest, ])
  ## Prediction for train
  pred.ab.train <- predict(ab, df.occ[indtrain, ])
  ## Accuracy
  c(sum(diag(table(pred.ab.test$class, df.occ$y.occ[indtest]))) / length(indtest), sum(diag(table(pred.ab.train$class, df.occ$y.occ[indtrain]))) / length(indtrain))
  confusionMatrix(as.factor(pred.ab.test$class), df.occ$y.occ[indtest])
  confusionMatrix(as.factor(pred.ab.train$class), df.occ$y.occ[indtrain])
  
  ## Boosting: Gradient Boosting
  ## dataframe for occurrence: GBM function doesn't work properly with factors. See:
  ## https://stackoverflow.com/questions/21198007/gbm-model-generating-na-results
  df.gb.occ <- df.occ
  df.gb.occ$y.occ <- as.character(df.gb.occ$y.occ)
  
  gb.cv <- gbm(y.occ ~., data=df.gb.occ[indtrain, ], n.trees=1000, interaction.depth=20, shrinkage = 0.01, cv.folds = 10)
  ntree_opt_cv <- gbm.perf(gb.cv, method = "cv")
  ntree_opt_oob <- gbm.perf(gb.cv, method = "OOB")
  
  print(ntree_opt_cv)
  print(ntree_opt_oob)
  
  gb <- gbm(y.occ ~., data=df.gb.occ[indtrain, ], n.trees=ntree_opt_cv, interaction.depth=20, shrinkage = 0.01)
  print(gb)
  summary(gb)
  ## Prediction for test
  pred.gb.test <- predict(object = gb, newdata = df.gb.occ[indtest, ], n.trees = ntree_opt_cv, type = "response")
  ## Prediction for train
  pred.gb.train <- predict(object = gb, newdata = df.gb.occ[indtrain, ], n.trees = ntree_opt_cv, type = "response")
  ## Accuracy
  pred.gb.test.bin <- as.factor(ifelse(pred.gb.test>mean(as.numeric(df.gb.occ$y.occ[indtrain])),1,0))
  pred.gb.train.bin <- as.factor(ifelse(pred.gb.train>mean(as.numeric(df.gb.occ$y.occ[indtrain])),1,0))
  c(sum(diag(table(pred.gb.test.bin, df.gb.occ$y.occ[indtest]))) / length(indtest), sum(diag(table(pred.gb.train.bin, df.gb.occ$y.occ[indtrain]))) / length(indtrain))
  
  ###################################################################################
  library(MASS)
  
  n <- nrow(Boston)
  # train/test partition
  indtrain <- sample(1:n, round(0.75*n))  # indices for train
  indtest <- setdiff(1:n, indtrain)  # indices for test
  
  # RF
  rf <- randomForest(medv ~., Boston , subset = indtrain)
  # RF configuration?
  
  # OOB error?
  plot(rf$mse, type = "l", xlab = "no. trees", ylab = "OOB error")
  grid()
  
  ## fitting mtry
  ntree <- which(rf$mse == min(rf$mse))
  
  # OOB error?
  err.oob <- c()
  for (mtry in 1:13) {
    rf.mtry <- randomForest(medv ~., Boston, subset = indtrain, ntree = ntree, mtry = mtry)  
    err.oob[mtry] <- rf.mtry$mse[ntree]
  }
  
  # test error?
  err.test <- c()
  for (mtry in 1:13) {
    rf.mtry <- randomForest(medv ~., Boston, subset = indtrain, ntree = ntree, mtry = mtry)
    pred.mtry <- predict(rf.mtry, Boston[indtest, ])
    err.test[mtry] <- mean((pred.mtry - Boston$medv[indtest])^2)
  }
  
  # OOB vs. test errors
  matplot(1:13 , cbind(err.oob, err.test), type = "b", pch = 19 ,   
          lty = 1, col = c("black", "red"),
          ylab = "MSE", xlab = "no. predictors")
  grid()
  legend("topright", c("OOB", "test"), lty = 1, col = c("black", "red"))
  
  ## fitting mtry with caret
  library(caret)
  rf.caret <- train(medv ~., Boston, subset = indtrain,
                    method = "rf", ntree = ntree, 
                    tuneGrid = expand.grid(mtry = 1:13))
  plot(rf.caret)
  
  ## predictors' importance
  rf.opt <- train(medv ~., Boston, subset = indtrain,
                  method = "rf", ntree = ntree, 
                  tuneGrid = expand.grid(mtry = 6),
                  importance = T)
  plot(varImp(rf.opt))
  
  ### AdaBoost:
  install.packages("adabag")
  library(adabag)
  
  # AdaBoost with 20 trees (mfinal)
  ab <- boosting(y.occ ~., df.occ[indtrain, ], mfinal = 20)  
  
  # train errors as a function of number of trees
  plot(errorevol(ab, df.occ[indtrain, ]))
  grid()
  
  # we can pick and draw individual trees
  plot(ab$trees[[1]])
  text(ab$trees[[1]], pretty = F)
  
  ## prediction for test
  pred.ab <- predict(ab, df.occ[indtest, ])
  # test error
  1 - sum(diag(table(pred.ab$class, df.occ$y.occ[indtest]))) / length(indtest)
  
  
  
  #############################################################################
  
  simpleGBMModel <- gbm(Survived~.-PassengerId - Name - Ticket, distribution="bernoulli", data=Train, n.trees=1000, interaction.depth=4, shrinkage = 0.01, cv.folds = 2)
  
  print(simpleGBMModel)
  summary(simpleGBMModel)
  
  ### Gradient Boosting:
  GradientBoostingModel <- gbm(formula = response~., distribution = "bernoulli", data = train, n.trees = 1000)
  
  Train <- read.csv("/home/sixto/Documentos/Docencia/M1966_DataMining/titanic/train.csv")
  
  set.seed(77850)
  indTrain <- createDataPartition(y=Train$Survived, p = 750/891, list = FALSE)
  Train <- Train[indTrain,]
  Test <- Train[-indTrain,]
  
  simpleGBMModel <- gbm(Survived~.-PassengerId-Name-Ticket, distribution="bernoulli", data=Train, n.trees=1000, interaction.depth=4,shrinkage = 0.01)
  print(simpleGBMModel)
  summary(simpleGBMModel)
  
  simpleGBMModel <- gbm(Survived~.-PassengerId - Name - Ticket, distribution="bernoulli", data=Train, n.trees=1000, interaction.depth=4, shrinkage = 0.01, cv.folds = 2)
  
  ntree_opt_cv <- gbm.perf(simpleGBMModel, method = "cv")
  ntree_opt_oob <- gbm.perf(simpleGBMModel, method = "OOB")
  
  Predictions <- predict(object = simpleGBMModel, newdata = test, n.trees = ntree_opt_cv, type = "response")
  Predictions <- predict(object = simpleGBMModel, newdata = test, n.trees = ntree_opt_cv, type = "response")
  
  PredictionsBinaries <- as.factor(ifelse(Predictions>0.7,1,0))
  testing$Survived <- as.factor(testing$Survived)
  confusionMatrix(PredictionsBinaries, testing$Survived)
  
  GBM_pred_testing <- prediction(Predictions, testing$Survived)
  GBM_ROC_testing <- performance(GBM_pred_testing, "tpr", "fpr")
  plot(GBM_ROC_testing)
  plot(GBM_ROC_testing, add=TRUE, col="green")
  legend("right", legend=c("GBM"), col=c("green"),lty=1:2, cex=0.6)
  
  auc.tmp <- performance(GBM_pred_testing, "auc")
  gbm_auc_testing <- as.numeric(auc.tmp@y.values)
  gbm_auc_testing
  
  #############################################################################
  # RF configuration: no. of trees? no. of predictors 
  ## considered at each node?
  str(rf)
  
  # OOB error
  plot(rf$err.rate[, 1], type = "b", xlab = "no trees", ylab = "OBB error") 
  grid()
  # prediction for test
  pred <- predict(rf, iris[indtest, ])
  # accuracy
  sum(diag(table(pred, iris$Species[indtest]))) / length(indtest)
  
  #############################################################################
  # comparison with a single tree
  library(tree)
  t <- tree(Species ~., iris, subset = indtrain)
  # prediction for test
  pred.t <- predict(t, iris[indtest, ], type = "class")
  # accuracy
  sum(diag(table(pred.t, iris$Species[indtest]))) / length(indtest)
  #############################################################################
  
  ############################################################################
  ### Gradient Boosting:
  install.packages("gbm")
  library(gbm)
  library(caret)
  GradientBoostingModel <- gbm(formula = response~., distribution = "bernoulli", data = train, n.trees = 1000)
  
  Train <- read.csv("/home/sixto/Documentos/Docencia/M1966_DataMining/titanic/train.csv")
  
  set.seed(77850)
  indTrain <- createDataPartition(y=Train$Survived, p = 750/891, list = FALSE)
  Train <- Train[indTrain,]
  Test <- Train[-indTrain,]
  
  simpleGBMModel <- gbm(Survived~.-PassengerId-Name-Ticket, distribution="bernoulli", data=Train, n.trees=1000, interaction.depth=4,shrinkage = 0.01)
  print(simpleGBMModel)
  summary(simpleGBMModel)
  
  selectMethod("as.factor", "item.vector")
  
  simpleGBMModel <- gbm(Survived~.-PassengerId - Name - Ticket, distribution="bernoulli", data=Train, n.trees=1000, interaction.depth=4, shrinkage = 0.01, cv.folds = 2)
  
  ntree_opt_cv <- gbm.perf(simpleGBMModel, method = "cv")
  ntree_opt_oob <- gbm.perf(simpleGBMModel, method = "OOB")
  
  print(ntree_opt_cv)
  print(ntree_opt_oob)
  
  Predictions <- predict(object = simpleGBMModel, newdata = test, n.trees = ntree_opt_cv, type = "response")
  
  PredictionsBinaries <- as.factor(ifelse(Predictions>0.7,1,0))
  testing$Survived <- as.factor(testing$Survived)
  confusionMatrix(PredictionsBinaries, testing$Survived)
  
  GBM_pred_testing <- prediction(Predictions, testing$Survived)
  GBM_ROC_testing <- performance(GBM_pred_testing, "tpr", "fpr")
  plot(GBM_ROC_testing)
  plot(GBM_ROC_testing, add=TRUE, col="green")
  legend("right", legend=c("GBM"), col=c("green"),lty=1:2, cex=0.6)
  
  auc.tmp <- performance(GBM_pred_testing, "auc")
  gbm_auc_testing <- as.numeric(auc.tmp@y.values)
  gbm_auc_testing
  
