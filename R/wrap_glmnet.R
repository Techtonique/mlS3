#' S3 wrapper for glmnet
#'
#' Fits a `glmnet` penalized regression model with a consistent
#' interface. Supports regression and binary classification.
#'
#' @param x A matrix or data.frame of features.
#' @param y A factor or character vector for classification, numeric for regression.
#' @param ... Additional arguments passed to [glmnet::glmnet()].
#'   Pass `family = "binomial"` for binary classification.
#' @return An object of class `wrap_glmnet` with fields:
#'   \item{fit}{The fitted glmnet model.}
#'   \item{levels}{Class levels (NULL for regression).}
#'   \item{task}{"classification" or "regression".}
#' @note Multiclass (`family = "multinomial"`) is not yet supported.
#'   For lambda selection, a specific `s` value can be passed to `predict()`.
#'   By default the midpoint of the lambda path is used. For optimal lambda,
#'   use [glmnet::cv.glmnet()] externally and pass `s = fit$lambda.min`.
#' @examples
#' \donttest{
#' X <- iris[iris$Species != "virginica", 1:4]
#' y <- droplevels(iris[iris$Species != "virginica", "Species"])
#' mod <- wrap_glmnet(X, y, family = "binomial")
#' predict(mod, newx = X, type = "class")
#' predict(mod, newx = X, type = "prob")
#' }
#' @export
wrap_glmnet <- function(x, y, ...) {
  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("Package 'glmnet' required. Install with: install.packages('glmnet')")
  x   <- as.matrix(x)
  fit <- glmnet::glmnet(x = x, y = y, ...)
  .wrap(fit, y, "wrap_glmnet")
}

#' @rdname wrap_glmnet
#' @param object A fitted `wrap_glmnet` object.
#' @param newx A matrix or data.frame of new observations.
#' @param type `"class"` (default) for class labels, `"prob"` for a probability
#'   matrix. Ignored for regression.
#' @param s Lambda value for prediction. Defaults to the midpoint of the
#'   lambda path. Pass `s = cv_fit$lambda.min` if using [glmnet::cv.glmnet()].
#' @export
predict.wrap_glmnet <- function(object, newx, type = c("class", "prob"),
                                s = NULL, ...) {
  newx <- as.matrix(newx)
  lvls <- object$levels
  # lambda.min only exists on cv.glmnet — use midpoint of lambda path
  if (is.null(s))
    s <- object$fit$lambda[length(object$fit$lambda) %/% 2L]
  if (object$task == "regression")
    return(drop(predict(object$fit, newx = newx, s = s, type = "response")))
  # Binomial: type="response" gives P(class=2)
  raw   <- drop(predict(object$fit, newx = newx, s = s, type = "response"))
  probs <- cbind(1 - raw, raw)
  colnames(probs) <- lvls
  .format_output(probs, match.arg(type), lvls, object$task)
}

#' @rdname wrap_glmnet
#' @export
print.wrap_glmnet <- function(x, ...) {
  fam <- tryCatch(x$fit$call$family, error = function(e) "gaussian")
  cat("wrap_glmnet\n")
  cat("  task    :", x$task, "\n")
  if (!is.null(x$levels))
    cat("  classes :", paste(x$levels, collapse = ", "), "\n")
  cat("  family  :", deparse(fam), "\n")
  cat("  lambdas :", length(x$fit$lambda), "\n")
  invisible(x)
}
