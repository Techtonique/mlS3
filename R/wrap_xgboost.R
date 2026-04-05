#' S3 wrapper for xgboost
#'
#' Fits an `xgboost` model with a consistent interface.
#' Supports binary classification, multiclass classification, and regression.
#'
#' @param x A matrix or data.frame of features.
#' @param y A factor or character vector for classification, numeric for regression.
#' @param ... Additional arguments passed to [xgboost::xgboost()].
#'   The `objective` argument is required for classification
#'   (e.g. `"binary:logistic"`, `"multi:softprob"`).
#' @return An object of class `wrap_xgboost` with fields:
#'   \item{fit}{The fitted xgboost model.}
#'   \item{levels}{Class levels (NULL for regression).}
#'   \item{task}{"classification" or "regression".}
#'   \item{objective}{The xgboost objective string, stored at fit time.}
#' @examples
#' \donttest{
#' X <- as.matrix(iris[iris$Species != "virginica", 1:4])
#' y <- droplevels(iris[iris$Species != "virginica", "Species"])
#' mod <- wrap_xgboost(X, y, nrounds = 50, objective = "binary:logistic", verbose = 0)
#' predict(mod, newx = X, type = "class")
#' predict(mod, newx = X, type = "prob")
#' }
#' @export
wrap_xgboost <- function(x, y, ...) {
  if (!requireNamespace("xgboost", quietly = TRUE))
    stop("Package 'xgboost' required. Install with: install.packages('xgboost')")
  x     <- as.matrix(x)
  args  <- list(...)
  y_fit <- if (.is_classif(y)) as.numeric(.to_factor(y)) - 1L else as.numeric(y)
  fit   <- xgboost::xgboost(data = x, label = y_fit, ...)
  .wrap(fit, y, "wrap_xgboost", objective = args$objective)
}

#' @rdname wrap_xgboost
#' @param object A fitted `wrap_xgboost` object.
#' @param newx A matrix or data.frame of new observations.
#' @param type `"class"` (default) for class labels, `"prob"` for a probability
#'   matrix. Ignored for regression.
#' @export
predict.wrap_xgboost <- function(object, newx, type = c("class", "prob"), ...) {
  newx <- as.matrix(newx)
  raw  <- predict(object$fit, newx)
  if (object$task == "regression") return(raw)
  obj  <- object$objective
  if (is.null(obj))
    stop("objective not stored -- refit with wrap_xgboost()")
  lvls <- object$levels
  k    <- length(lvls)
  if (grepl("binary", obj)) {
    probs <- cbind(1 - raw, raw)
    colnames(probs) <- lvls
    return(.format_output(probs, match.arg(type), lvls, object$task))
  }
  if (grepl("multi", obj)) {
    probs <- matrix(raw, nrow = nrow(newx), ncol = k, byrow = TRUE)
    colnames(probs) <- lvls
    return(.format_output(probs, match.arg(type), lvls, object$task))
  }
  raw
}

#' @rdname wrap_xgboost
#' @export
print.wrap_xgboost <- function(x, ...) {
  cat("wrap_xgboost\n")
  cat("  task     :", x$task, "\n")
  if (!is.null(x$levels))
    cat("  classes  :", paste(x$levels, collapse = ", "), "\n")
  cat("  objective:", x$objective %||% x$fit$params$objective, "\n")
  cat("  rounds   :", x$fit$niter, "\n")
  invisible(x)
}
