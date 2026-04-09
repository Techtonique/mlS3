#' Wrap caret models for mlS3
#'
#' Minimal wrapper around caret::train with no tuning.
#' Hyperparameters can be passed via ... as named arguments.
#'
#' @param X Feature matrix or data frame
#' @param y Response vector
#' @param method caret model method (default "rf")
#' @param ... Named hyperparameters (e.g., mtry = 3, ntree = 500)
#'
#' @return Object with class "mlS3_caret"
#' @export
#'
#' @examples
#' \dontrun{
#' # Only runs if caret is installed
#' mod <- wrap_caret(X_train, y_train, method = "rf", mtry = 3)
#' pred <- predict(mod, newx = X_test)
#' }
wrap_caret <- function(X, y, method = "rf", ...) {
  # Check if caret is available - gives helpful error if not
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop(
      "Package 'caret' is required for wrap_caret().\n",
      "Please install it with: install.packages('caret')\n",
      "Alternatively, use other mlS3 wrappers like wrap_xgboost() or wrap_ranger()"
    )
  }

  # Infer task
  task <- if (is.numeric(y)) "regression" else "classification"

  # Ensure factor for classification
  if (task == "classification" && !is.factor(y)) y <- factor(y)

  # Convert ... to tuneGrid if not empty
  dots <- list(...)
  tuneGrid <- if (length(dots) > 0) as.data.frame(dots) else NULL

  # Combine X and y into a data.frame
  df <- data.frame(X, y = y)

  # Train the model
  mod <- caret::train(
    y ~ .,
    data = df,
    method = method,
    trControl = caret::trainControl(method = "none"),
    tuneGrid = tuneGrid
  )

  # Return S3-style object
  structure(
    list(
      model = mod,
      task = task,
      method = method,
      parameters = dots
    ),
    class = "wrap_caret"
  )
}

#' Predict method for mlS3 caret wrapper
#'
#' @param object Object from wrap_caret
#' @param newx New features (matrix or data frame)
#' @param type Prediction type: "raw" (default), "class", "prob", or NULL
#' @param ... Additional arguments to caret::predict.train
#'
#' @return Vector or matrix of predictions
#' @export
predict.wrap_caret <- function(object, newx, type = NULL, ...) {
  # Check caret availability for prediction too
  if (!requireNamespace("caret", quietly = TRUE)) {
    stop("Package 'caret' is required for predictions from wrap_caret models")
  }

  # Map common mlS3 types to caret types
  caret_type <- if (is.null(type) || type == "class") {
    "raw"
  } else if (type == "prob") {
    if (object$task == "regression") {
      warning("Probability predictions not available for regression tasks, using 'raw'")
      "raw"
    } else {
      "prob"
    }
  } else {
    type
  }

  # Make predictions
  caret::predict.train(object$model, newdata = newx, type = caret_type, ...)
}

#' Print method for wrap_caret objects
#'
#' @param x Object from wrap_caret
#' @param ... Additional arguments
#' @export
print.wrap_caret <- function(x, ...) {
  cat("\n=== mlS3 caret wrapper ===\n")
  cat("Method:", x$method, "\n")
  cat("Task:", x$task, "\n")
  cat("Model class:", class(x$model$finalModel)[1], "\n")

  # Show parameters used
  if (length(x$parameters) > 0) {
    cat("\nParameters:\n")
    param_df <- data.frame(
      Parameter = names(x$parameters),
      Value = unlist(x$parameters),
      row.names = NULL
    )
    print(param_df, row.names = FALSE)
  } else {
    cat("\nParameters: (caret defaults - may involve tuning)\n")
  }

  # Show training info
  cat("\nTraining data shape:", nrow(x$model$trainingData), "rows,",
      ncol(x$model$trainingData) - 1, "features\n")

  # Show call
  cat("\nCall:", deparse(x$call), "\n")

  invisible(x)
}
