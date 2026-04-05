X_bin   <- as.matrix(iris[iris$Species != "virginica", 1:4])
y_bin   <- droplevels(iris[iris$Species != "virginica", "Species"])
X_multi <- as.matrix(iris[, 1:4])
y_multi <- iris$Species
X_reg   <- as.matrix(mtcars[, -1])
y_reg   <- mtcars$mpg

is_prob_matrix <- function(m, k) {
  is.matrix(m) && ncol(m) == k && all(abs(rowSums(m) - 1) < 1e-6)
}

test_that("wrap_ranger binary classification", {
  skip_if_not_installed("ranger")
  mod <- wrap_ranger(X_bin, y_bin, num.trees = 50L)
  expect_s3_class(mod, "wrap_ranger")
  expect_equal(mod$task, "classification")
  expect_s3_class(predict(mod, newx = X_bin, type = "class"), "factor")
  expect_true(is_prob_matrix(predict(mod, newx = X_bin, type = "prob"), 2L))
})

test_that("wrap_ranger multiclass", {
  skip_if_not_installed("ranger")
  mod <- wrap_ranger(X_multi, y_multi, num.trees = 50L)
  expect_true(is_prob_matrix(predict(mod, newx = X_multi, type = "prob"), 3L))
})

test_that("wrap_ranger regression", {
  skip_if_not_installed("ranger")
  mod <- wrap_ranger(X_reg, y_reg, num.trees = 50L)
  expect_equal(mod$task, "regression")
  expect_true(is.numeric(predict(mod, newx = X_reg)))
})

test_that("wrap_xgboost binary", {
  skip_if_not_installed("xgboost")
  mod <- wrap_xgboost(X_bin, y_bin, nrounds = 20,
                      objective = "binary:logistic", verbose = 0)
  expect_s3_class(predict(mod, newx = X_bin, type = "class"), "factor")
  expect_true(is_prob_matrix(predict(mod, newx = X_bin, type = "prob"), 2L))
})

test_that("wrap_xgboost multiclass", {
  skip_if_not_installed("xgboost")
  mod <- wrap_xgboost(X_multi, y_multi, nrounds = 20,
                      objective = "multi:softprob", num_class = 3, verbose = 0)
  expect_true(is_prob_matrix(predict(mod, newx = X_multi, type = "prob"), 3L))
})

test_that("wrap_xgboost regression", {
  skip_if_not_installed("xgboost")
  mod <- wrap_xgboost(X_reg, y_reg, nrounds = 20,
                      objective = "reg:squarederror", verbose = 0)
  expect_equal(mod$task, "regression")
  expect_true(is.numeric(predict(mod, newx = X_reg)))
})

test_that("wrap_lightgbm binary", {
  skip_if_not_installed("lightgbm")
  mod <- wrap_lightgbm(X_bin, y_bin,
                       params = list(objective = "binary", verbose = -1),
                       nrounds = 20)
  expect_true(is_prob_matrix(predict(mod, newx = X_bin, type = "prob"), 2L))
})

test_that("wrap_lightgbm regression", {
  skip_if_not_installed("lightgbm")
  mod <- wrap_lightgbm(X_reg, y_reg,
                       params = list(objective = "regression", verbose = -1),
                       nrounds = 20)
  expect_equal(mod$task, "regression")
  expect_true(is.numeric(predict(mod, newx = X_reg)))
})

test_that("wrap_glmnet binary", {
  skip_if_not_installed("glmnet")
  mod <- wrap_glmnet(X_bin, y_bin, family = "binomial")
  expect_s3_class(predict(mod, newx = X_bin, type = "class"), "factor")
  expect_true(is_prob_matrix(predict(mod, newx = X_bin, type = "prob"), 2L))
})

test_that("wrap_glmnet regression", {
  skip_if_not_installed("glmnet")
  mod <- wrap_glmnet(X_reg, y_reg, alpha = 0)
  expect_equal(mod$task, "regression")
  expect_true(is.numeric(predict(mod, newx = X_reg)))
})

test_that("wrap_svm binary", {
  skip_if_not_installed("e1071")
  mod <- wrap_svm(X_bin, y_bin, kernel = "radial")
  expect_s3_class(predict(mod, newx = X_bin, type = "class"), "factor")
  expect_true(is_prob_matrix(predict(mod, newx = X_bin, type = "prob"), 2L))
})

test_that("wrap_svm regression", {
  skip_if_not_installed("e1071")
  mod <- wrap_svm(X_reg, y_reg)
  expect_equal(mod$task, "regression")
  expect_true(is.numeric(predict(mod, newx = X_reg)))
})
