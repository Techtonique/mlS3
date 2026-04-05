# TRUE if y should be treated as a classification target
#' @keywords internal
.is_classif <- function(y) is.factor(y) || is.character(y)

# Coerce character y to factor; leave factors unchanged
#' @keywords internal
.to_factor <- function(y) {
  if (is.character(y)) y <- as.factor(y)
  y
}

# Convert a probability matrix to a factor of predicted class labels
# by picking the column with the highest probability per row
#' @keywords internal
.prob_to_class <- function(probs, levels) {
  factor(levels[max.col(probs)], levels = levels)
}

# Build the standard wrap_* return object, storing metadata alongside the fit
#' @keywords internal
.wrap <- function(fit, y, class_name, objective = NULL) {
  task   <- if (.is_classif(y)) "classification" else "regression"
  levels <- if (task == "classification") levels(.to_factor(y)) else NULL
  structure(list(fit = fit, levels = levels, task = task, objective = objective),
            class = class_name)
}

# Coerce newx to either a matrix or a data.frame as required by the backend
#' @keywords internal
.coerce_newx <- function(newx, to = c("matrix", "data.frame")) {
  to <- match.arg(to)
  if (to == "matrix")     return(as.matrix(newx))
  if (to == "data.frame") return(as.data.frame(newx))
}

# Rename columns to X1, X2, ... so formula-based backends (ranger) always
# see the same names at predict time as at fit time
#' @keywords internal
.std_colnames <- function(x) {
  colnames(x) <- paste0("X", seq_len(ncol(x)))
  x
}

# Route predict output to the correct format:
#   regression -> pass numeric vector through unchanged
#   type="prob"  -> return named probability matrix as-is
#   type="class" -> convert probability matrix to factor of class labels
#' @keywords internal
.format_output <- function(probs, type, levels, task) {
  if (task == "regression") return(probs)
  type <- match.arg(type, c("class", "prob"))
  if (type == "prob")  return(probs)
  if (type == "class") return(.prob_to_class(probs, levels))
}

# NULL-coalescing operator: return a if non-NULL, otherwise b
#' @keywords internal
`%||%` <- function(a, b) if (!is.null(a)) a else b
