run_grid_search <- function(model_function, hyper_grid, input, k, DV, ML,
                             validationInput = NULL,
                             validationDV = NULL,
                             seed = 123
) {
  
  ## ----- TASK DETECTION -----
  task <- if (is.factor(DV)) "classification" else "regression"
  
  results <- dplyr::bind_rows(
    lapply(1:nrow(hyper_grid), function(j) {
      
      params <- as.list(hyper_grid[j, , drop = FALSE])
      
      base_args <- list(
        input = input,
        k = k,
        DV = DV,
        ML = ML,
        seed = seed,
        validationInput = validationInput,
        validationDV = validationDV
      )
      
      x <- do.call(model_function, c(base_args, params))
      
      ## ----- SUMMARY BY TASK -----
      if (task == "classification") {
        
        result <- data.frame(
          CV_Accuracy = mean(x[, 1]),
          CV_Balanced_Accuracy = mean(x[, 2]),
          CV_MCC = mean(x[, 3]),
          CV_Avg_F1 = mean(x[, 4])
        )
        
      } else {
        
        result <- data.frame(
          CV_RMSE = mean(x[, 1]),
          CV_MAE  = mean(x[, 2]),
          CV_R2   = mean(x[, 3])
        )
      }
      
      dplyr::bind_cols(result, hyper_grid[j, , drop = FALSE])
    })
  )
  
  results
}

Function_RF <- function(input, k, DV, ML, 
                            validationInput = NULL,
                            validationDV = NULL,
                            seed = 123, probability = FALSE,  
                            num.trees = 500, 
                            mtry = floor(sqrt(ncol(input))), 
                            min.node.size = 1, 
                            max.depth = 0, 
                            replace = TRUE, 
                            sample.fraction = 1,
                            externalV = TRUE, 
                            repetitions = 1,  
                            loss = NULL,
                            type = "ratio") {
  
  if (is.null(colnames(input))) {
    colnames(input) <- paste0("x", 1:ncol(input))
  }
  
  set.seed(seed)
  
  ## ----- TASK DETECTION -----
  task <- if (is.factor(DV)) "classification" else "regression"
  is_classif <- task == "classification"
  
  ## ----- DEFAULT LOSS FOR DALEX -----
  if (is.null(loss)) {
    loss <- if (is_classif) {
      DALEX::loss_cross_entropy
    } else {
      DALEX::loss_root_mean_square
    }
  }
  
  ## ----- MODE SELECTION -----
  if (is.null(validationInput)) {
    folds <- cvTools::cvFolds(NROW(input), K = k)
    K_eff <- k
  } else {
    K_eff <- 1
  }
  
  dt  <- data.frame()
  dt2 <- data.frame(matrix(nrow = ncol(input), ncol = K_eff))
  
  ## ----- Helper: pseudo-probabilities from votes (classification only) -----
  vote_to_prob <- function(model, newdata) {
    votes <- predict(model, newdata, predict.all = TRUE)$predictions
    class_levels <- model$forest$levels
    class_codes  <- seq_along(class_levels)
    probs <- sapply(class_codes, function(i) rowMeans(votes == i))
    colnames(probs) <- class_levels
    as.data.frame(probs)
  }
  
  for (i in 1:K_eff) {
    
    ## ----- TRAIN / VALIDATION SPLIT -----
    if (is.null(validationInput)) {
      train_idx <- folds$subsets[folds$which != i]
      valid_idx <- folds$subsets[folds$which == i]
      
      train <- input[train_idx, ]
      validation <- input[valid_idx, ]
      DV_train <- DV[train_idx]
      DV_valid <- DV[valid_idx]
      
    } else {
      train <- input
      validation <- validationInput
      DV_train <- DV
      DV_valid <- validationDV
    }
    
    ##############################
    #### PERFORMANCE (externalV)
    ##############################
    
    if (externalV) {
      
      set.seed(seed)
      model <- ML(
        y = DV_train, x = train,
        num.trees = num.trees, mtry = mtry,
        min.node.size = min.node.size, max.depth = max.depth,
        replace = replace, sample.fraction = sample.fraction,
        probability = probability
      )
      
      ## ==========================================================
      ## ================= CLASSIFICATION (UNCHANGED) =============
      ## ==========================================================
      
      if (is_classif) {
        
        ## ---------- BINARY CASE ----------
        if (length(levels(DV)) < 3) {
          
          if (probability) {
            pred_obj <- predict(model, validation, type = "response")
            probs <- pred_obj$predictions
            if (is.matrix(probs)) {
              pred <- colnames(probs)[max.col(probs)]
            } else {
              pred <- ifelse(probs > 0.5, levels(DV)[2], levels(DV)[1])
            }
            pred <- factor(pred, levels = levels(DV))
          } else {
            pred_obj <- predict(model, validation)
            pred <- pred_obj$predictions
          }
          
          class_table <- table("Predictions" = pred,
                               "Actual" = DV_valid)
          message(paste0("K-fold / validation completed: ", i))
          print(class_table)
          
          df <- caret::confusionMatrix(class_table, mode = "everything")
          dt[i, 1] <- df$overall[1]
          dt[i, 2] <- df$byClass[11]
          
          actual <- rep(1:nrow(class_table), each = nrow(class_table))
          predicted <- rep(1:nrow(class_table), times = nrow(class_table))
          values <- as.vector(class_table)
          dm <- data.table::data.table(actual = rep(actual, values),
                                       predicted = rep(predicted, values))
         # dm$actual <- factor(dm$actual, labels = levels(DV))
         #  dm$predicted <- factor(dm$predicted, labels = levels(DV))
          
          dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                               labels = levels(DV))
          
          dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                                 labels = levels(DV))
          
          if (length(unique(dm$predicted)) > 1 && length(unique(dm$actual)) > 1) {
            dt[i, 3] <- as.numeric(yardstick::mcc(dm, actual, predicted)[3])
          } else {
            dt[i, 3] <- NA
          }
          
          dt[i, 4] <- ((2 * df$byClass[1] * df$byClass[3]) /
                         (df$byClass[1] + df$byClass[3]) +
                         (2 * df$byClass[2] * df$byClass[4]) /
                         (df$byClass[2] + df$byClass[4])) / 2
          
          dt[i, 5] <- (2 * df$byClass[1] * df$byClass[3]) /
            (df$byClass[1] + df$byClass[3])
          dt[i, 6] <- (2 * df$byClass[2] * df$byClass[4]) /
            (df$byClass[2] + df$byClass[4])
          
          colnames(dt)[1] <- "Accuracy"
          colnames(dt)[2] <- "Balanced Accuracy"
          colnames(dt)[3] <- "MCC"
          colnames(dt)[4] <- "Avg. F1"
          colnames(dt)[5] <- paste0("F1 ", levels(DV)[1])
          colnames(dt)[6] <- paste0("F1 ", levels(DV)[2])
          
          result <- dt
          
        } else {
          
          ## ---------- MULTICLASS CASE ----------
          if (probability) {
            pred_obj <- predict(model, validation, type = "response")
            probs <- pred_obj$predictions
            pred <- colnames(probs)[max.col(probs)]
            pred <- factor(pred, levels = levels(DV))
          } else {
            pred_obj <- predict(model, validation)
            pred <- pred_obj$predictions
          }
          
          class_table <- table("Predictions" = pred,
                               "Actual" = DV_valid)
          message(paste0("K-fold / validation completed: ", i))
          print(class_table)
          
          df <- caret::confusionMatrix(class_table, mode = "everything")
          p <- as.data.frame(df$byClass)
          
          dt[i, 1] <- df$overall[1]
          dt[i, 2] <- mean(p[, 6])
          
          actual <- rep(1:nrow(class_table), each = nrow(class_table))
          predicted <- rep(1:nrow(class_table), times = nrow(class_table))
          values <- as.vector(class_table)
          dm <- data.table::data.table(actual = rep(actual, values),
                                       predicted = rep(predicted, values))
         # dm$actual <- factor(dm$actual, labels = levels(DV))
        #  dm$predicted <- factor(dm$predicted, labels = levels(DV))
         
          dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                              labels = levels(DV))
          
          dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                                 labels = levels(DV))
          
          if (length(unique(dm$predicted)) > 1 && length(unique(dm$actual)) > 1) {
            dt[i, 3] <- as.numeric(yardstick::mcc(dm, actual, predicted)[3])
          } else {
            dt[i, 3] <- NA
          }
          
          dt[i, 4] <- mean(p$F1, na.rm = TRUE)
          
          for (z in 1:nlevels(DV)) {
            dt[i, 4 + z] <- p$F1[z]
            colnames(dt)[4 + z] <- paste0("F1 ", levels(DV)[z])
          }
          
          colnames(dt)[1:4] <- c("Accuracy", "Balanced Accuracy", "MCC", "Avg. F1")
          result <- dt
        }
        
      } else {
        
        ## ==========================================================
        ## ===================== REGRESSION =========================
        ## ==========================================================
        
        pred <- predict(model, validation)$predictions
        
        dt[i, 1] <- yardstick::rmse_vec(DV_valid, pred)
        dt[i, 2] <- yardstick::mae_vec(DV_valid, pred)
        dt[i, 3] <- yardstick::rsq_vec(DV_valid, pred)
        
        message(paste0("K-fold / validation completed: ", i))
        # print(dt)
        
        colnames(dt)[1:3] <- c("RMSE", "MAE", "R2")
        
        # print(dt)
        result <- dt
      }
      
    } else {
      
      ##############################
      #### DALEX / INGREDIENTS
      ##############################
      
      set.seed(seed)
      model <- ML(
        y = DV_train, x = train,
        num.trees = num.trees, mtry = mtry,
        min.node.size = min.node.size, max.depth = max.depth,
        replace = replace, sample.fraction = sample.fraction,
        probability = probability
      )
      
      y_for_explain <- if (is_classif) {
        factor(DV_valid, levels = levels(DV))
      } else {
        DV_valid
      }
      
      pred_fun <- function(m, newdata) {
        if (is_classif) {
          if (probability) {
            as.data.frame(predict(m, newdata, type = "response")$predictions)
          } else {
            vote_to_prob(m, newdata)
          }
        } else {
          predict(m, newdata)$predictions
        }
      }
      
      expl <- DALEX::explain(
        model = model,
        data = validation,
        y = y_for_explain,
        predict_function = pred_fun,
        verbose = FALSE,
        label = "ranger RF"
      )
      
      set.seed(seed)
      fi_obj <- ingredients::feature_importance(
        expl,
        B = repetitions,
        type = type,
        loss_function = loss
      )
      
      message(paste0("K-fold / validation completed: ", i))
      
      fi_obj <- fi_obj[!fi_obj$variable %in% c("_baseline_", "_full_model_"), ]
      imp <- aggregate(dropout_loss ~ variable, fi_obj, mean)
      x <- imp[order(imp$variable, decreasing = FALSE), ]
      
      dt2[, i] <- x$dropout_loss
      row.names(dt2) <- x$variable
      result <- dt2
    }
  }
  
  result
}

Function_NB <- function(input, k, DV, ML,
                         validationInput = NULL,
                         validationDV = NULL,
                         seed = 123, laplace = 0.5,
                         externalV = TRUE,
                         loss = DALEX::loss_cross_entropy,
                         repetitions = 1, type = "ratio",
                         dataframe = NULL,
                         factor = FALSE) {
  
  if (is.null(colnames(input))) {
    colnames(input) <- paste0("x", 1:ncol(input))
  }
  set.seed(seed)
  
  if (factor && is.null(dataframe)) {
    stop("When factor = TRUE, you must provide `dataframe`.")
  }
  
  if (factor && nrow(input) != nrow(dataframe)) {
    stop("`input` and `dataframe` must have the same number of rows.")
  }
  
  ## ----- MODE SELECTION -----
  if (is.null(validationInput)) {
    folds <- cvTools::cvFolds(NROW(input), K = k)
    K_eff <- k
  } else {
    K_eff <- 1
  }
  
  dt  <- data.frame()
  dt2 <- data.frame(matrix(nrow = ncol(input), ncol = K_eff))
  
  for (i in seq_len(K_eff)) {
    
    ## ----- TRAIN / VALIDATION SPLIT -----
    if (is.null(validationInput)) {
      train_idx <- folds$subsets[folds$which != i]
      valid_idx <- folds$subsets[folds$which == i]
      
      train <- input[train_idx, ]
      validation <- input[valid_idx, ]
      DV_train <- DV[train_idx]
      DV_valid <- DV[valid_idx]
      
      if (factor) {
        dataframe_train <- dataframe[train_idx, , drop = FALSE]
        dataframe_valid <- dataframe[valid_idx, , drop = FALSE]
        terms_obj <- terms(~ ., data = dataframe_train)
      }
      
    } else {
      train <- input
      validation <- validationInput
      DV_train <- DV
      DV_valid <- validationDV
      
      if (factor) {
        dataframe_train <- dataframe
        dataframe_valid <- dataframe
        terms_obj <- terms(~ ., data = dataframe_train)
      }
    }
    
    ##############################
    #### PERFORMANCE (externalV)
    ##############################
    
    if (externalV) {
      
      set.seed(seed)
      model <- ML(y = DV_train, x = train, laplace = laplace)
      
      ## ---------- BINARY CASE ----------
      if (length(levels(DV)) < 3) {
        
        pred <- predict(model, validation, type = "class")
        pred <- factor(pred, levels = levels(DV))
        
        class_table <- table("Predictions" = pred,
                             "Actual" = DV_valid)
        message(paste0("K-fold / validation completed: ", i))
        print(class_table)
        
        df <- caret::confusionMatrix(class_table, mode = "everything")
        dt[i, 1] <- df$overall[1]
        dt[i, 2] <- df$byClass[11]
        
        actual <- rep(seq_len(nrow(class_table)), each = nrow(class_table))
        predicted <- rep(seq_len(nrow(class_table)), times = nrow(class_table))
        values <- as.vector(class_table)
        
        dm <- data.table::data.table(
          actual = rep(actual, values),
          predicted = rep(predicted, values)
        )
       # dm$actual <- factor(dm$actual, labels = levels(DV))
      #  dm$predicted <- factor(dm$predicted, labels = levels(DV))
        dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                            labels = levels(DV))
        
        dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                               labels = levels(DV))
        
        dt[i, 3] <- if (length(unique(dm$predicted)) > 1 &&
                        length(unique(dm$actual)) > 1)
          as.numeric(yardstick::mcc(dm, actual, predicted)[3]) else NA
        
        dt[i, 4] <- ((2 * df$byClass[1] * df$byClass[3]) /
                       (df$byClass[1] + df$byClass[3]) +
                       (2 * df$byClass[2] * df$byClass[4]) /
                       (df$byClass[2] + df$byClass[4])) / 2
        
        dt[i, 5] <- (2 * df$byClass[1] * df$byClass[3]) /
          (df$byClass[1] + df$byClass[3])
        dt[i, 6] <- (2 * df$byClass[2] * df$byClass[4]) /
          (df$byClass[2] + df$byClass[4])
        
        colnames(dt)[1:6] <- c(
          "Accuracy", "Balanced Accuracy", "MCC",
          "Avg. F1",
          paste0("F1 ", levels(DV)[1]),
          paste0("F1 ", levels(DV)[2])
        )
        
        result <- dt
        
      } else {
        
        ## ---------- MULTICLASS CASE ----------
        pred <- predict(model, validation, type = "class")
        pred <- factor(pred, levels = levels(DV))
        
        class_table <- table("Predictions" = pred,
                             "Actual" = DV_valid)
        message(paste0("K-fold / validation completed: ", i))
        print(class_table)
        
        df <- caret::confusionMatrix(class_table, mode = "everything")
        p <- as.data.frame(df$byClass)
        
        dt[i, 1] <- df$overall[1]
        dt[i, 2] <- mean(p[, 6])
        
        actual <- rep(seq_len(nrow(class_table)), each = nrow(class_table))
        predicted <- rep(seq_len(nrow(class_table)), times = nrow(class_table))
        values <- as.vector(class_table)
        
        dm <- data.table::data.table(
          actual = rep(actual, values),
          predicted = rep(predicted, values)
        )
       
        # dm$actual <- factor(dm$actual, labels = levels(DV))
        # dm$predicted <- factor(dm$predicted, labels = levels(DV))
        
        dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                            labels = levels(DV))
        
        dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                               labels = levels(DV))
        
        dt[i, 3] <- if (length(unique(dm$predicted)) > 1 &&
                        length(unique(dm$actual)) > 1)
          as.numeric(yardstick::mcc(dm, actual, predicted)[3]) else NA
        
        dt[i, 4] <- mean(p$F1, na.rm = TRUE)
        
        for (z in seq_len(nlevels(DV))) {
          dt[i, 4 + z] <- p$F1[z]
          colnames(dt)[4 + z] <- paste0("F1 ", levels(DV)[z])
        }
        
        colnames(dt)[1:4] <- c("Accuracy", "Balanced Accuracy", "MCC", "Avg. F1")
        result <- dt
      }
      
    } else {
      
      ##############################
      #### DALEX / INGREDIENTS
      ##############################
      
      set.seed(seed)
      model <- ML(
        y = DV_train,
        x = train,
        laplace = laplace
      )
      
      y_for_explain <- factor(DV_valid, levels = levels(DV))
      
      pred_fun <- function(m, newdata) {
        if (factor) {
          newdata <- model.matrix(terms_obj, data = newdata)[, -1, drop = FALSE]
        }
        probs <- predict(m, newdata, type = "prob")
        if (is.data.frame(probs)) probs <- as.matrix(probs)
        probs <- probs[, levels(DV), drop = FALSE]
        storage.mode(probs) <- "numeric"
        probs
      }
      
      expl <- DALEX::explain(
        model = model,
        data = if (factor) dataframe_valid else validation,
        y = y_for_explain,
        predict_function = pred_fun,
        verbose = FALSE,
        label = "naivebayes NB",
        residual_function = function(...) NA,
        model_info = DALEX::model_info(
          model = model,
          type  = "classification"
        )
      )
      
      fi_obj <- ingredients::feature_importance(
        expl,
        B = repetitions,
        type = type,
        loss_function = loss
      )
      
      message(paste0("K-fold / validation completed: ", i))
      
      fi_obj <- fi_obj[!fi_obj$variable %in% c("_baseline_", "_full_model_"), ]
      imp <- aggregate(dropout_loss ~ variable, fi_obj, mean)
      imp <- imp[order(imp$variable), ]
      
      if (i == 1) {
        dt2 <- data.frame(matrix(
          nrow = nrow(imp),
          ncol = K_eff
        ))
        rownames(dt2) <- imp$variable
      }
      
      row.names(dt2) <- imp$variable
      dt2[imp$variable, i] <- imp$dropout_loss
      result <- dt2
    }
  }
  
  result
}

Function_NN <- function(input, k, DV, ML,
                          validationInput = NULL,
                          validationDV = NULL,
                          seed = 123, max_features,
                          units1 = units,
                          units2 = units,
                          units3 = units,
                          hidden2 = NA,
                          hidden3 = NA,
                          scaling = TRUE, norm = TRUE, norm_type = "batch",
                          rate1 = 0, rate2 = 0, rate3 = 0,
                          regularizer1_l1 = 0, regularizer1_l2 = 0,
                          regularizer2_l1 = 0, regularizer2_l2 = 0,
                          regularizer3_l1 = 0, regularizer3_l2 = 0,
                          batch_size = batch_size,
                          epochs = epochs, learning_rate = 0.001,
                          activation_hidden = "relu",
                          optimizer_type = "rmsprop",
                          use_init = FALSE,
                          externalV = TRUE,
                          repetitions = 1,
                          loss = NULL,
                          type = "ratio",
                          dataframe = NULL,
                          factor = FALSE) {
  
  if (is.null(colnames(input))) {
    colnames(input) <- paste0("x", 1:ncol(input))
  }
  
  set.seed(seed)
  
  ## ----- TASK DETECTION -----
  task <- if (is.factor(DV)) "classification" else "regression"
  is_classif <- task == "classification"
  binary_case <- is_classif && length(levels(DV)) < 3
  
  if (is.null(loss)) {
    loss <- if (is_classif) {
      DALEX::loss_cross_entropy
    } else {
      DALEX::loss_root_mean_square
    }
  }
  
  if (is.null(validationInput)) {
    folds <- cvTools::cvFolds(NROW(input), K = k)
    K_eff <- k
  } else {
    K_eff <- 1
  }
  
  dt  <- data.frame()
  dt2 <- NULL
  
  make_reg <- function(l1, l2) {
    if (l1 > 0 || l2 > 0) regularizer_l1_l2(l1 = l1, l2 = l2) else NULL
  }
  
  add_norm <- function(model) {
    if (norm) {
      if (norm_type == "batch") layer_batch_normalization(model)
      else if (norm_type == "layer") layer_layer_normalization(model)
    }
  }
  
  hidden_init <- if (use_init) "he_normal" else NULL
  output_init <- if (use_init) "glorot_uniform" else NULL
  
  for (i in seq_len(K_eff)) {
    
    if (is.null(validationInput)) {
      train_idx <- folds$subsets[folds$which != i]
      valid_idx <- folds$subsets[folds$which == i]
      
      DV_train <- DV[train_idx]
      DV_valid <- DV[valid_idx]
      
      if (factor) {
        dataframe_train <- dataframe[train_idx, , drop = FALSE]
        dataframe_valid <- dataframe[valid_idx, , drop = FALSE]
        dv_obj <- caret::dummyVars(~ ., data = dataframe_train, fullRank = FALSE)
        trainK <- predict(dv_obj, dataframe_train)
        validationK <- predict(dv_obj, dataframe_valid)
      } else {
        trainK <- input[train_idx, ]
        validationK <- input[valid_idx, ]
      }
      
    } else {
      DV_train <- DV
      DV_valid <- validationDV
      
      if (factor) {
        dataframe_train <- dataframe
        dataframe_valid <- dataframe
        dv_obj <- caret::dummyVars(~ ., data = dataframe_train, fullRank = FALSE)
        trainK <- predict(dv_obj, dataframe_train)
        validationK <- predict(dv_obj, dataframe_valid)
      } else {
        trainK <- input
        validationK <- validationInput
      }
    }
    
    if (scaling) {
      mean_tr <- apply(trainK, 2, mean)
      std_tr  <- apply(trainK, 2, sd)
      trainK <- scale(trainK, center = mean_tr, scale = std_tr)
      validationK <- scale(validationK, center = mean_tr, scale = std_tr)
      trainK[is.na(trainK)] <- 0
      validationK[is.na(validationK)] <- 0
    }
    
    tensorflow::set_random_seed(seed)
    modelCV <- ML()
    
    layer_dense(modelCV, units = units1, activation = NULL,
                input_shape = c(ncol(trainK)),
                use_bias = !norm,
                kernel_initializer = hidden_init,
                kernel_regularizer = make_reg(regularizer1_l1, regularizer1_l2))
    add_norm(modelCV)
    layer_activation(modelCV, activation_hidden)
    layer_dropout(modelCV, rate = rate1)
    
    if (!is.na(hidden2)) {
      layer_dense(modelCV, units = units2, activation = NULL,
                  use_bias = !norm,
                  kernel_initializer = hidden_init,
                  kernel_regularizer = make_reg(regularizer2_l1, regularizer2_l2))
      add_norm(modelCV)
      layer_activation(modelCV, activation_hidden)
      layer_dropout(modelCV, rate = rate2)
    }
    
    if (!is.na(hidden3)) {
      layer_dense(modelCV, units = units3, activation = NULL,
                  use_bias = !norm,
                  kernel_initializer = hidden_init,
                  kernel_regularizer = make_reg(regularizer3_l1, regularizer3_l2))
      add_norm(modelCV)
      layer_activation(modelCV, activation_hidden)
      layer_dropout(modelCV, rate = rate3)
    }
    
    if (is_classif) {
      
      if (length(levels(DV)) < 3) {
        layer_dense(modelCV, units = 1, activation = "sigmoid",
                    kernel_initializer = output_init)
        compile_loss <- "binary_crossentropy"
      } else {
        layer_dense(modelCV, units = length(levels(DV)), activation = "softmax",
                    kernel_initializer = output_init)
        compile_loss <- "sparse_categorical_crossentropy"
      }
      
      train_labels <- as.numeric(DV_train) - 1
      test_labels  <- as.numeric(DV_valid) - 1
      
    } else {
      
      layer_dense(modelCV, units = 1, activation = "linear",
                  kernel_initializer = output_init)
      compile_loss <- "mse"
      
      train_labels <- DV_train
      test_labels  <- DV_valid
    }
    
    opt <- switch(as.character(optimizer_type),
                  "rmsprop" = optimizer_rmsprop(learning_rate = learning_rate),
                  "adam"    = optimizer_adam(learning_rate = learning_rate),
                  "sgd"     = optimizer_sgd(learning_rate = learning_rate),
                  "adamw"   = tf$keras$optimizers$AdamW(learning_rate = learning_rate),
                  stop("Unknown optimizer_type"))
    
    keras3::compile(modelCV, optimizer = opt, loss = compile_loss)
    
    keras3::fit(modelCV, trainK, train_labels,
                epochs = epochs, batch_size = batch_size, verbose = FALSE)
    
    if (externalV) {
      
      if (is_classif) {
        
        if (length(levels(DV)) < 3) {
          preds_prob <- as.numeric(predict(modelCV, validationK))
          pred <- ifelse(preds_prob > 0.5, levels(DV)[2], levels(DV)[1])
        } else {
          preds <- predict(modelCV, validationK)
          pred <- levels(DV)[max.col(preds)]
        }
        
        pred <- factor(pred, levels = levels(DV))
        class_table <- table(Predictions = pred, Actual = DV_valid)
        message(paste0("K-fold / validation completed: ", i))
        print(class_table)
        
        df <- caret::confusionMatrix(class_table, mode = "everything")
        
        if (binary_case) {
          
          dt[i, 1] <- df$overall[1]
          dt[i, 2] <- df$byClass[11]
          
          actual <- rep(seq_len(nrow(class_table)), each = nrow(class_table))
          predicted <- rep(seq_len(nrow(class_table)), times = nrow(class_table))
          values <- as.vector(class_table)
          
          dm <- data.table::data.table(
            actual = rep(actual, values),
            predicted = rep(predicted, values)
          )
         
          # dm$actual <- factor(dm$actual, labels = levels(DV))
          # dm$predicted <- factor(dm$predicted, labels = levels(DV))
          
          dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                              labels = levels(DV))
          
          dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                                 labels = levels(DV))
          
          dt[i, 3] <- if (length(unique(dm$actual)) > 1 &&
                          length(unique(dm$predicted)) > 1)
            as.numeric(yardstick::mcc(dm, actual, predicted)[3]) else NA
          
          f1_1 <- (2 * df$byClass[1] * df$byClass[3]) /
            (df$byClass[1] + df$byClass[3])
          f1_2 <- (2 * df$byClass[2] * df$byClass[4]) /
            (df$byClass[2] + df$byClass[4])
          
          dt[i, 4] <- mean(c(f1_1, f1_2), na.rm = TRUE)
          dt[i, 5] <- f1_1
          dt[i, 6] <- f1_2
          
          colnames(dt)[1:6] <- c(
            "Accuracy", "Balanced Accuracy", "MCC",
            "Avg. F1",
            paste0("F1 ", levels(DV)[1]),
            paste0("F1 ", levels(DV)[2])
          )
          
          result <- dt
          
        } else {
          
        ## ---------- MULTICLASS CASE ----------
	p <- as.data.frame(df$byClass)

	dt[i, 1] <- df$overall["Accuracy"]
	dt[i, 2] <- mean(p[, 6], na.rm = TRUE)

	actual <- rep(seq_len(nrow(class_table)), each = nrow(class_table))
	predicted <- rep(seq_len(nrow(class_table)), times = nrow(class_table))
	values <- as.vector(class_table)

	dm <- data.table::data.table(
	  actual = rep(actual, values),
	  predicted = rep(predicted, values)
	)

	dm$actual <- factor(dm$actual, levels = seq_len(nlevels(DV)),
                    labels = levels(DV))
	dm$predicted <- factor(dm$predicted, levels = seq_len(nlevels(DV)),
                       labels = levels(DV))

	if (length(unique(dm$predicted)) > 1 && length(unique(dm$actual)) > 1) {
	  dt[i, 3] <- as.numeric(yardstick::mcc(dm, actual, predicted)[3])
	} else {
	  dt[i, 3] <- NA
	}

	dt[i, 4] <- mean(p$F1, na.rm = TRUE)

	for (z in seq_len(nlevels(DV))) {
  dt[i, 4 + z] <- p$F1[z]
  colnames(dt)[4 + z] <- paste0("F1 ", levels(DV)[z])
	}

	colnames(dt)[1:4] <- c("Accuracy", "Balanced Accuracy", "MCC", "Avg. F1")
          
          result <- dt
        }
        
      } 
        
         else {
        
        pred <- as.numeric(predict(modelCV, validationK))
        dt[i, 1] <- yardstick::rmse_vec(test_labels, pred)
        dt[i, 2] <- yardstick::mae_vec(test_labels, pred)
        dt[i, 3] <- yardstick::rsq_vec(test_labels, pred)
        message(paste0("K-fold / validation completed: ", i))
        
        colnames(dt)[1:3] <- c("RMSE", "MAE", "R2")
        result <- dt
      }
      
    } else {
      
      y_for_explain <- if (is_classif)
        factor(DV_valid, levels = levels(DV)) else DV_valid
      
      pred_fun <- function(m, newdata) {
        
        if (factor) {
          newdata <- predict(dv_obj, newdata)
          
          if (scaling) {
            newdata <- scale(newdata, center = mean_tr, scale = std_tr)
            newdata[is.na(newdata)] <- 0
          }
        }
        
        p <- predict(m, as.matrix(newdata))
        
        ## ----- REGRESSION -----
        if (!is_classif) {
          return(as.numeric(p))
        }
        
        ## ----- CLASSIFICATION -----
        if (ncol(p) == 1) {
          p <- as.numeric(p)
          out <- cbind(1 - p, p)
          colnames(out) <- levels(DV)
          return(out)
        } else {
          colnames(p) <- levels(DV)
          return(as.matrix(p))
        }
      }
      
      
      
      expl <- DALEX::explain(
        model = modelCV,
        data  = if (factor) dataframe_valid else validationK,
        y     = y_for_explain,
        predict_function = pred_fun,
        residual_function = function(...) NA,
        model_info = DALEX::model_info(
          model = modelCV,
          type  = if (is_classif) "classification" else "regression"
        ),
        verbose = FALSE,
        label = "keras NN"
      )
      
      
      message(paste0("K-fold / validation completed: ", i))
      
      fi_obj <- ingredients::feature_importance(
        expl, B = repetitions, type = type, loss_function = loss
      )
      
      fi_obj <- fi_obj[!fi_obj$variable %in%
                         c("_baseline_", "_full_model_"), ]
      imp <- aggregate(dropout_loss ~ variable, fi_obj, mean)
      imp <- imp[order(imp$variable), ]
      
      if (i == 1) {
        dt2 <- data.frame(matrix(
          nrow = nrow(imp),
          ncol = K_eff
        ))
        rownames(dt2) <- imp$variable
      }
      
      dt2[imp$variable, i] <- imp$dropout_loss
      result <- dt2
    }
  }
  
  result
}
