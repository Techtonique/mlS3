# mlS3 

R package `mlS3` providing an S3 interface to Machine Learning packages.

[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/mlsS3/index.html)

## Install 

```R
remotes::install_github("Techtonique/mlS3")
```

## Classification example

```R
library(mlS3)

# =============================================================================
# Classification examples (no leakage)
# =============================================================================
set.seed(123)

# --- Binary classification: iris setosa vs versicolor ---
iris_bin <- iris[iris$Species != "virginica", ]
X_bin <- iris_bin[, 1:4]
y_bin <- droplevels(iris_bin$Species)

# Split into train/test
idx_bin <- sample(nrow(X_bin), 0.7 * nrow(X_bin))
X_bin_train <- X_bin[idx_bin, ]
y_bin_train <- y_bin[idx_bin]
X_bin_test  <- X_bin[-idx_bin, ]
y_bin_test  <- y_bin[-idx_bin]

# xgboost
mod <- wrap_xgboost(X_bin_train, y_bin_train,
                    nrounds = 50, objective = "binary:logistic", verbose = 0)
pred_bin_xgboost <- predict(mod, newx = X_bin_test, type = "class")
acc_xgboost <- mean(pred_bin_xgboost == y_bin_test)

# glmnet
mod <- wrap_glmnet(X_bin_train, y_bin_train, family = "binomial")
pred_bin_glmnet <- predict(mod, newx = X_bin_test, type = "class")
acc_glmnet <- mean(pred_bin_glmnet == y_bin_test)

cat("Accuracy (xgboost): ", acc_xgboost, "\n")
cat("Accuracy (glmnet): ", acc_glmnet, "\n")


# --- Multiclass classification: iris all species ---
X_multi <- iris[, 1:4]
y_multi <- iris$Species

# Split into train/test
idx_multi <- sample(nrow(X_multi), 0.7 * nrow(X_multi))
X_multi_train <- X_multi[idx_multi, ]
y_multi_train <- y_multi[idx_multi]
X_multi_test  <- X_multi[-idx_multi, ]
y_multi_test  <- y_multi[-idx_multi]

# lightgbm
mod <- wrap_lightgbm(X_multi_train, y_multi_train,
                     params = list(objective = "multiclass",
                                   num_class = 3, verbose = -1),
                     nrounds = 150)
pred_multi_lightgbm <- predict(mod, newx = X_multi_test, type = "class")
acc_lightgbm <- mean(pred_multi_lightgbm == y_multi_test)

# ranger
mod <- wrap_ranger(X_multi_train, y_multi_train, num.trees = 100L)
pred_multi_ranger <- predict(mod, newx = X_multi_test, type = "class")
acc_ranger <- mean(pred_multi_ranger == y_multi_test)

# svm
mod <- wrap_svm(X_multi_train, y_multi_train, kernel = "radial")
pred_multi_svm <- predict(mod, newx = X_multi_test, type = "class")
acc_svm <- mean(pred_multi_svm == y_multi_test)

cat("Accuracy (lightgbm): ", acc_lightgbm, "\n")
cat("Accuracy (ranger): ", acc_ranger, "\n")
cat("Accuracy (svm): ", acc_svm, "\n")
```

## Regression example

```R
# =============================================================================
# Regression examples (mtcars)
# =============================================================================
X_reg <- mtcars[, -1]
y_reg <- mtcars$mpg

# Split into train/test
set.seed(123)
idx_reg <- sample(nrow(X_reg), 0.7 * nrow(X_reg))
X_reg_train <- X_reg[idx_reg, ];  y_reg_train <- y_reg[idx_reg]
X_reg_test  <- X_reg[-idx_reg, ]; y_reg_test  <- y_reg[-idx_reg]

# xgboost
mod <- wrap_xgboost(X_reg_train, y_reg_train,
                    nrounds = 50, objective = "reg:squarederror", verbose = 0)
pred_reg_xgboost <- predict(mod, newx = X_reg_test)
rmse_xgboost <- sqrt(mean((pred_reg_xgboost - y_reg_test)^2))

# lightgbm
mod <- wrap_lightgbm(X_reg_train, y_reg_train,
                     params = list(objective = "regression", verbose = -1),
                     nrounds = 50)
pred_reg_lightgbm <- predict(mod, newx = X_reg_test)
rmse_lightgbm <- sqrt(mean((pred_reg_lightgbm - y_reg_test)^2))

# glmnet
mod <- wrap_glmnet(X_reg_train, y_reg_train, alpha = 0)
pred_reg_glmnet <- predict(mod, newx = X_reg_test)
rmse_glmnet <- sqrt(mean((pred_reg_glmnet - y_reg_test)^2))

# svm
mod <- wrap_svm(X_reg_train, y_reg_train)
pred_reg_svm <- predict(mod, newx = X_reg_test)
rmse_svm <- sqrt(mean((pred_reg_svm - y_reg_test)^2))

# ranger
mod <- wrap_ranger(X_reg_train, y_reg_train, num.trees = 100L)
pred_reg_ranger <- predict(mod, newx = X_reg_test)
rmse_ranger <- sqrt(mean((pred_reg_ranger - y_reg_test)^2))

cat("RMSE (xgboost): ", rmse_xgboost, "\n")
cat("RMSE (lightgbm): ", rmse_lightgbm, "\n")
cat("RMSE (glmnet): ", rmse_glmnet, "\n")
cat("RMSE (svm): ", rmse_svm, "\n")
cat("RMSE (ranger): ", rmse_ranger, "\n")
```


