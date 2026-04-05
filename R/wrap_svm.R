#' S3 wrapper for e1071 SVM
#'
#' Fits an `e1071` support vector machine with a consistent
#' interface. Supports classification and regression.
#'
#' @param x A matrix or data.frame of features.
#' @param y A factor or character vector for classification, numeric for regression.
#' @param ... Additional arguments passed to [e1071::svm()].
#'   `probability = TRUE` is set automatically for classification; do not
#'   override this if you need `type = "prob"` predictions.
#' @return An object of class `wrap_svm` with fields:
#'   \item{fit}{The fitted svm model.}
#'   \item{levels}{Class levels (NULL for regression).}
#'   \item{task}{"classification" or "regression".}
#' @examples
#' \donttest{
#' X <- as.matrix(iris[, 1:4])
#' y <- iris$Species
#' mod <- wrap_svm(X, y, kernel = "radial")
#' predict(mod, newx = X, type = "class")
#' predict(mod, newx = X, type = "prob")
#' }
#' @export
wrap_svm <- function(x, y, ...) {
  if (!requireNamespace("e1071", quietly = TRUE))
    stop("Package 'e1071' required. Install with: install.packages('e1071')")
  x   <- as.matrix(x)
  fit <- e1071::svm(x = x, y = y, probability = .is_classif(y), ...)
  .wrap(fit, y, "wrap_svm")
}

#' @rdname wrap_svm
#' @param object A fitted `wrap_svm` object.
#' @param newx A matrix or data.frame of new observations.
#' @param type `"class"` (default) for class labels, `"prob"` for a probability
#'   matrix. Ignored for regression.
#' @export
predict.wrap_svm <- function(object, newx, type = c("class", "prob"), ...) {
  newx <- as.matrix(newx)
  type <- match.arg(type)
  if (object$task == "regression")
    return(as.numeric(predict(object$fit, newdata = newx)))
  if (type == "prob") {
    raw   <- predict(object$fit, newdata = newx, probability = TRUE)
    probs <- attr(raw, "probabilities")
    if (is.null(probs))
      stop("Model was not trained with probability = TRUE. Refit with wrap_svm().")
    probs <- probs[, object$levels, drop = FALSE]
    return(probs)
  }
  # type == "class": svm returns factor directly
  predict(object$fit, newdata = newx)
}

#' @rdname wrap_svm
#' @export
print.wrap_svm <- function(x, ...) {
  cat("wrap_svm\n")
  cat("  task   :", x$task, "\n")
  if (!is.null(x$levels))
    cat("  classes:", paste(x$levels, collapse = ", "), "\n")
  cat("  kernel :", x$fit$kernel, "\n")
  invisible(x)
}
