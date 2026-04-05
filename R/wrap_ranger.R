#' S3 wrapper for ranger
#'
#' Fits a `ranger` random forest with a consistent interface.
#' Supports both classification (factor `y`) and regression (numeric `y`).
#'
#' @param x A matrix or data.frame of features.
#' @param y A factor or character vector for classification, numeric for regression.
#' @param ... Additional arguments passed to [ranger::ranger()].
#' @return An object of class `wrap_ranger` with fields:
#'   \item{fit}{The fitted ranger model.}
#'   \item{levels}{Class levels (NULL for regression).}
#'   \item{task}{"classification" or "regression".}
#' @examples
#' \donttest{
#' X <- as.matrix(iris[, 1:4])
#' y <- iris$Species
#' mod <- wrap_ranger(X, y, num.trees = 100L)
#' predict(mod, newx = X, type = "class")
#' predict(mod, newx = X, type = "prob")
#' }
#' @export
wrap_ranger <- function(x, y, ...) {
  if (!requireNamespace("ranger", quietly = TRUE))
    stop("Package 'ranger' required. Install with: install.packages('ranger')")
  x <- .std_colnames(.coerce_newx(x, "data.frame"))
  if (.is_classif(y)) y <- .to_factor(y)
  df  <- data.frame(y = y, x)
  fit <- ranger::ranger(y ~ ., data = df, probability = .is_classif(y), ...)
  .wrap(fit, y, "wrap_ranger")
}

#' @rdname wrap_ranger
#' @param object A fitted `wrap_ranger` object.
#' @param newx A matrix or data.frame of new observations.
#' @param type `"class"` (default) for class labels, `"prob"` for a probability
#'   matrix. Ignored for regression.
#' @export
predict.wrap_ranger <- function(object, newx, type = c("class", "prob"), ...) {
  newx <- .std_colnames(.coerce_newx(newx, "data.frame"))
  raw  <- predict(object$fit, data = newx)$predictions
  if (object$task == "regression") return(as.numeric(raw))
  colnames(raw) <- object$levels
  .format_output(raw, match.arg(type), object$levels, object$task)
}

#' @rdname wrap_ranger
#' @export
print.wrap_ranger <- function(x, ...) {
  cat("wrap_ranger\n")
  cat("  task    :", x$task, "\n")
  if (!is.null(x$levels))
    cat("  classes :", paste(x$levels, collapse = ", "), "\n")
  cat("  trees   :", x$fit$num.trees, "\n")
  cat("  features:", x$fit$num.independent.variables, "\n")
  invisible(x)
}
