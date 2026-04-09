#' S3 wrapper for lightgbm
#'
#' Fits a `lightgbm` model with a consistent interface.
#' Supports binary classification, multiclass classification, and regression.
#'
#' @param x A matrix or data.frame of features.
#' @param y A factor or character vector for classification, numeric for regression.
#' @param ... Additional arguments passed to [lightgbm::lgb.train()].
#'   Pass `params = list(objective = "binary")` for binary classification,
#'   `params = list(objective = "multiclass", num_class = k)` for multiclass,
#'   or `params = list(objective = "regression")` for regression.
#' @return An object of class `wrap_lightgbm` with fields:
#'   \item{fit}{The fitted lgb.Booster model.}
#'   \item{levels}{Class levels (NULL for regression).}
#'   \item{task}{"classification" or "regression".}
#'   \item{objective}{The lightgbm objective string, stored at fit time.}
#' @examples
#' \donttest{
#' X <- iris[, 1:4]
#' y <- iris$Species
#' mod <- wrap_lightgbm(X, y,
#'   params = list(objective = "multiclass", num_class = 3, verbose = -1),
#'   nrounds = 50)
#' predict(mod, newx = X, type = "class")
#' predict(mod, newx = X, type = "prob")
#' }
#' @export
wrap_lightgbm <- function(x, y, ...) {
  if (!requireNamespace("lightgbm", quietly = TRUE))
    stop("Package 'lightgbm' required. Install with: install.packages('lightgbm')")
  x     <- as.matrix(x)
  args  <- list(...)
  y_fit <- if (.is_classif(y)) as.numeric(.to_factor(y)) - 1L else as.numeric(y)
  ds    <- lightgbm::lgb.Dataset(data = x, label = y_fit)
  fit   <- lightgbm::lgb.train(data = ds, ...)
  obj   <- args$params$objective %||% args$objective
  .wrap(fit, y, "wrap_lightgbm", objective = obj)
}

#' @rdname wrap_lightgbm
#' @param object A fitted `wrap_lightgbm` object.
#' @param newx A matrix or data.frame of new observations.
#' @param type `"class"` (default) for class labels, `"prob"` for a probability
#'   matrix. Ignored for regression.
#' @export
predict.wrap_lightgbm <- function(object, newx, type = c("class", "prob"), ...) {
  newx <- as.matrix(newx)
  raw  <- predict(object$fit, newx)
  if (object$task == "regression") return(raw)
  obj  <- object$objective
  if (is.null(obj))
    stop("objective not stored -- refit with wrap_lightgbm()")
  lvls <- object$levels
  k    <- length(lvls)
  if (grepl("binary", obj)) {
    probs <- cbind(1 - raw, raw)
    colnames(probs) <- lvls
    return(.format_output(probs, match.arg(type), lvls, object$task))
  }
  if (grepl("multiclass", obj)) {
    probs <- matrix(raw, nrow = nrow(newx), ncol = k, byrow = TRUE)
    colnames(probs) <- lvls
    return(.format_output(probs, match.arg(type), lvls, object$task))
  }
  raw
}

#' @rdname wrap_lightgbm
#' @export
print.wrap_lightgbm <- function(x, ...) {
  cat("wrap_lightgbm\n")
  cat("  task     :", x$task, "\n")
  if (!is.null(x$levels))
    cat("  classes  :", paste(x$levels, collapse = ", "), "\n")
  cat("  objective:", x$objective, "\n")
  invisible(x)
}
