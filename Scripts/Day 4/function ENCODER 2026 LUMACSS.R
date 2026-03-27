#############################
# FUNCTION to extract dynamic WE
#############################

get_text_embeddings_stride <- function(
    model_name,
    layer,
    texts,
    embedding_type = c("avg", "cls", "tokens", "token_embeddings"),
    max_length = 512L,
    stride = 128L,
    device = NULL,
    return_overflowing_tokens = TRUE,
    remove_special_tokens = TRUE
) {
  
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  
  embedding_type <- match.arg(embedding_type)
  
  if (is.null(device)) {
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  message("🖥️ Using device: ", device)
  
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
  special_tokens <- unique(unlist(tokenizer$all_special_tokens))
  
  config <- transformers$AutoConfig$from_pretrained(model_name)
  config$output_hidden_states <- TRUE
  model <- transformers$AutoModel$from_pretrained(model_name, config = config)
  model$eval()
  if (device == "cuda") model$to(device)
  
  result <- vector("list", length(texts))
  
  for (i in seq_along(texts)) {
    
    enc <- tokenizer(
      texts[[i]],
      truncation = TRUE,
      padding = TRUE,
      max_length = as.integer(max_length),
      stride = as.integer(stride),
      return_overflowing_tokens = return_overflowing_tokens,
      return_offsets_mapping = TRUE,
      return_tensors = "pt"
    )
    
    input_ids <- enc$input_ids
    attention_mask <- enc$attention_mask
    offset_mapping <- enc$offset_mapping
    
    if (device == "cuda") {
      input_ids <- input_ids$to(device)
      attention_mask <- attention_mask$to(device)
    }
    
    with(torch$no_grad(), {
      out <- model(
        input_ids = input_ids,
        attention_mask = attention_mask
      )
    })
    
    hs <- out$hidden_states[[layer + 1L]]   # [chunks, seq, dim]
    
    # --- EXACT SEMANTICS RESTORATION ---
    # Move minimal data to CPU once
    hs_cpu <- hs$cpu()
    ids_cpu <- input_ids$cpu()
    offsets_cpu <- offset_mapping$cpu()
    
    hs_np <- hs_cpu$numpy()
    ids_np <- ids_cpu$numpy()
    offsets_np <- offsets_cpu$numpy()
    
    position_embeddings <- list()
    position_counts <- list()
    position_tokens <- list()
    
    for (chunk in seq_len(dim(ids_np)[1])) {
      tokens <- tokenizer$convert_ids_to_tokens(as.integer(ids_np[chunk, ]))
      offsets <- offsets_np[chunk, , ]
      
      for (j in seq_along(tokens)) {
        token <- tokens[[j]]
        if (remove_special_tokens && token %in% special_tokens) next
        if (offsets[j, 1] == 0 && offsets[j, 2] == 0) next
        
        key <- as.character(offsets[j, 1])
        
        if (!is.null(position_embeddings[[key]])) {
          position_embeddings[[key]] <- position_embeddings[[key]] + hs_np[chunk, j, ]
          position_counts[[key]] <- position_counts[[key]] + 1
        } else {
          position_embeddings[[key]] <- hs_np[chunk, j, ]
          position_counts[[key]] <- 1
          position_tokens[[key]] <- token
        }
      }
    }
    
    if (embedding_type == "avg") {
      all_emb <- do.call(rbind, Map(
        function(e, c) e / c,
        position_embeddings,
        position_counts
      ))
      result[[i]] <- colMeans(all_emb)
    }
    
    if (embedding_type == "cls") {
      cls <- hs_np[, 1, , drop = FALSE]
      result[[i]] <- colMeans(cls)
    }
    
    if (embedding_type == "tokens") {
      result[[i]] <- unname(unlist(position_tokens, use.names = FALSE))
    }
    
    if (embedding_type == "token_embeddings") {
      if (length(position_embeddings) > 0) {
        tok_emb <- Map(
          function(e, c) e / c,
          position_embeddings,
          position_counts
        )
        result[[i]] <- list(
          tokens = unname(unlist(position_tokens, use.names = FALSE)),
          embeddings = do.call(rbind, tok_emb)
        )
      } else {
        result[[i]] <- list(
          tokens = list(),
          embeddings = matrix(nrow = 0, ncol = dim(hs_np)[3])
        )
      }
    }
  }
  
  if (embedding_type %in% c("avg", "cls")) {
    do.call(rbind, result)
  } else {
    result
  }
}

###############################
# FUNCTION to fine-tune an encoder model - GPU
###############################

######################
# for classification tasks
######################

train_transformer_classifier <- function( 
    model_name,
    num_labels,
    texts,
    labels,
    test_size = 0.5,
    num_train_epochs = 3L,
    per_device_train_batch_size = 2L,
    per_device_eval_batch_size = 2L,
    metric_for_best_model = "f1",
    seed = 123L,
    max_length = 512L,
    stride = 128L,
    dir = "./results",
    return_overflowing_tokens = NULL,
    learning_rate = 5e-05,
    weight_decay = 0.0,
    device = NULL,
    deterministic = TRUE     # <-- added
) {
  # --- Imports ---
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  datasets <- import("datasets")
  random <- import("random")
  metrics <- import("sklearn.metrics")
  warnings <- import("warnings")
  
  warnings$filterwarnings("ignore")
  
  # --- Device selection ---
  if (is.null(device)) {
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  message("🖥️ Using device: ", device)
  
if (device == "cuda") {

  if (deterministic) {
    # Reproducible but TRAINER-safe
    torch$manual_seed(seed)
    torch$cuda$manual_seed_all(seed)

    torch$backends$cudnn$deterministic <- TRUE
    torch$backends$cudnn$benchmark <- FALSE

    # IMPORTANT: do NOT enforce deterministic algorithms globally
    torch$use_deterministic_algorithms(FALSE)

    message("⚠️ GPU training: using reproducible (not strictly deterministic) mode")
  } else {
    torch$backends$cudnn$benchmark <- TRUE
  }

} else {
  torch$use_deterministic_algorithms(FALSE) # CPU mode
}
  
  # --- Seeds ---
  random$seed(seed)
  numpy$random$seed(seed)
  torch$manual_seed(seed)
  
  # --- Validation ---
  if (length(texts) != length(labels)) {
    stop("texts and labels must have the same length.")
  }
  
  df <- data.frame(text = texts, label = as.integer(labels), text_id = seq_along(texts))
  transformers$logging$set_verbosity_error()
  
  # --- Load tokenizer and model ---
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
  model <- transformers$AutoModelForSequenceClassification$from_pretrained(
    model_name,
    num_labels = as.integer(num_labels)
  )$to(device)
  
  # --- Detect long texts automatically ---
  encoded_lengths <- sapply(texts, function(t) length(tokenizer$encode(t, add_special_tokens = TRUE)))
  max_text_length <- max(encoded_lengths)
  
  if (is.null(return_overflowing_tokens)) {
    return_overflowing_tokens <- max_text_length > max_length
    message("🧠 Auto return_overflowing_tokens = ", return_overflowing_tokens,
            " (detected max sequence length = ", max_text_length, ")")
  }
  
  # --- Dataset creation ---
  dataset <- datasets$Dataset$from_pandas(df)
  
  tokenize_function <- function(examples) {
    encodings <- tokenizer(
      examples[["text"]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length),
      stride = as.integer(stride),
      return_overflowing_tokens = return_overflowing_tokens,
      return_offsets_mapping = TRUE
    )
    
    if (return_overflowing_tokens) {
      sample_map <- encodings$`overflow_to_sample_mapping`
      labels_list <- sapply(sample_map, function(i) as.integer(examples[["label"]][[i+1]]))
      text_ids <- sapply(sample_map, function(i) as.integer(examples[["text_id"]][[i+1]]))
    } else {
      labels_list <- as.integer(examples[["label"]])
      text_ids <- as.integer(examples[["text_id"]])
    }
    
    list(
      input_ids = encodings$input_ids,
      attention_mask = encodings$attention_mask,
      labels = labels_list,
      text_id = text_ids
    )
  }
  
  # --- Apply tokenization ---
  tokenized_dataset <- dataset$map(tokenize_function, batched = TRUE, remove_columns = c("text", "label"))
  
  # --- Train/Test split ---
  split_datasets <- tokenized_dataset$train_test_split(test_size = test_size, seed = seed)
  train_dataset <- split_datasets$train
  eval_dataset <- split_datasets$test
  
  # --- Training arguments ---
  training_args <- transformers$TrainingArguments(
    output_dir = dir,
    num_train_epochs = as.integer(num_train_epochs),
    per_device_train_batch_size = as.integer(per_device_train_batch_size),
    per_device_eval_batch_size = as.integer(per_device_eval_batch_size),
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = TRUE,
    metric_for_best_model = metric_for_best_model,
    greater_is_better = TRUE,
    logging_dir = file.path(dir, "logs"),
    logging_steps = 1L,
    seed = seed,
    save_total_limit = 1L,
    learning_rate = learning_rate,
    weight_decay = weight_decay
  )
  
  # --- Metrics ---
  if (return_overflowing_tokens) {
    message("📏 Using document-level metrics (chunk aggregation).")
    
    compute_metrics <- function(eval_pred) {
      logits <- eval_pred$predictions
      labels <- eval_pred$label_ids
      text_ids <- numpy$array(eval_dataset[["text_id"]])
      
      logits_df <- as.data.frame(logits)
      colnames(logits_df) <- paste0("logits_", seq_len(ncol(logits_df)) - 1)
      
      df_results <- data.frame(
        text_id = as.integer(text_ids),
        label = as.integer(labels),
        logits_df
      )
      
      logit_cols <- grep("^logits_", colnames(df_results), value = TRUE)
      agg_logits <- aggregate(as.matrix(df_results[, logit_cols]),
                              by = list(text_id = df_results$text_id, label = df_results$label),
                              FUN = mean)
      
      predictions <- apply(agg_logits[, logit_cols], 1, which.max) - 1
      
      accuracy <- metrics$accuracy_score(agg_logits$label, predictions)
      precision <- metrics$precision_score(agg_logits$label, predictions, average = "macro", zero_division = 0)
      recall <- metrics$recall_score(agg_logits$label, predictions, average = "macro", zero_division = 0)
      f1 <- metrics$f1_score(agg_logits$label, predictions, average = "macro", zero_division = 0)
      composite_score <- 0.5*f1 + 0.3*recall + 0.2*accuracy
      
      list(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
           composite_score=composite_score)
    }
    
  } else {
    message("⚡ Using normal evaluation (no chunking).")
    
    compute_metrics <- function(eval_pred) {
      logits <- eval_pred$predictions
      labels <- eval_pred$label_ids
      predictions <- numpy$argmax(logits, axis=-1L)
      
      accuracy <- metrics$accuracy_score(labels, predictions)
      precision <- metrics$precision_score(labels, predictions, average = "macro", zero_division = 0)
      recall <- metrics$recall_score(labels, predictions, average = "macro", zero_division = 0)
      f1 <- metrics$f1_score(labels, predictions, average = "macro", zero_division = 0)
      composite_score <- 0.5*f1 + 0.3*recall + 0.2*accuracy
      
      list(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
           composite_score=composite_score)
    }
  }
  
  # --- Trainer ---
  trainer <- transformers$Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics
  )
  
  trainer$train()
  
  # --- Save best model ---
  best_model_dir <- file.path(dir, "best_model")
  dir.create(best_model_dir, recursive = TRUE, showWarnings = FALSE)
  trainer$model$save_pretrained(best_model_dir)
  tokenizer$save_pretrained(best_model_dir)
  
  # --- Remove checkpoints ---
  unlink(list.files(dir, pattern="checkpoint", full.names=TRUE), recursive=TRUE)
  
  return(trainer)
}

######################
# for regression tasks
######################

train_transformer_regressor <- function( 
    model_name,
    texts,
    labels,
    test_size = 0.5,
    num_train_epochs = 3L,
    per_device_train_batch_size = 2L,
    per_device_eval_batch_size = 2L,
    metric_for_best_model = "rmse", # you can also use mae or mse or r2. rmse is sqrt(MSE)
    seed = 123L,
    max_length = 512L,
    stride = 128L,
    dir = "./results",
    return_overflowing_tokens = NULL,
    learning_rate = 5e-05,
    weight_decay = 0.0,
    device = NULL,
    deterministic = TRUE
) {
  # --- Imports ---
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  datasets <- import("datasets")
  random <- import("random")
  metrics <- import("sklearn.metrics")
  warnings <- import("warnings")
  
  warnings$filterwarnings("ignore")
  
  # --- Device selection ---
  if (is.null(device)) {
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  message("🖥️ Using device: ", device)
  
  if (device == "cuda") {
    if (deterministic) {
      torch$manual_seed(seed)
      torch$cuda$manual_seed_all(seed)
      torch$backends$cudnn$deterministic <- TRUE
      torch$backends$cudnn$benchmark <- FALSE
      torch$use_deterministic_algorithms(FALSE)
      message("⚠️ GPU training: using reproducible (not strictly deterministic) mode")
    } else {
      torch$backends$cudnn$benchmark <- TRUE
    }
  } else {
    torch$use_deterministic_algorithms(FALSE)
  }
  
  # --- Seeds ---
  random$seed(seed)
  numpy$random$seed(seed)
  torch$manual_seed(seed)
  
  # --- Validation ---
  if (length(texts) != length(labels)) {
    stop("texts and labels must have the same length.")
  }
  
  # REGRESSION: labels as numeric (float)
  df <- data.frame(
    text = texts,
    label = as.numeric(labels),
    text_id = seq_along(texts)
  )
  transformers$logging$set_verbosity_error()
  
  # --- Load tokenizer and model ---
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
  
  # REGRESSION: num_labels = 1 and problem_type = "regression"
  model <- transformers$AutoModelForSequenceClassification$from_pretrained(
    model_name,
    num_labels = as.integer(1),
    problem_type = "regression"
  )$to(device)
  
  # --- Detect long texts automatically ---
  encoded_lengths <- sapply(texts, function(t) length(tokenizer$encode(t, add_special_tokens = TRUE)))
  max_text_length <- max(encoded_lengths)
  
  if (is.null(return_overflowing_tokens)) {
    return_overflowing_tokens <- max_text_length > max_length
    message("🧠 Auto return_overflowing_tokens = ", return_overflowing_tokens,
            " (detected max sequence length = ", max_text_length, ")")
  }
  
  # --- Dataset creation ---
  dataset <- datasets$Dataset$from_pandas(df)
  
  tokenize_function <- function(examples) {
    encodings <- tokenizer(
      examples[["text"]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length),
      stride = as.integer(stride),
      return_overflowing_tokens = return_overflowing_tokens,
      return_offsets_mapping = TRUE
    )
    
    if (return_overflowing_tokens) {
      sample_map <- encodings$`overflow_to_sample_mapping`
      # REGRESSION: float labels
      labels_list <- sapply(sample_map, function(i) as.numeric(examples[["label"]][[i+1]]))
      text_ids <- sapply(sample_map, function(i) as.integer(examples[["text_id"]][[i+1]]))
    } else {
      labels_list <- as.numeric(examples[["label"]])
      text_ids <- as.integer(examples[["text_id"]])
    }
    
    list(
      input_ids = encodings$input_ids,
      attention_mask = encodings$attention_mask,
      # Trainer expects "labels" field; for regression keep float
      labels = labels_list,
      text_id = text_ids
    )
  }
  
  # --- Apply tokenization ---
  tokenized_dataset <- dataset$map(tokenize_function, batched = TRUE, remove_columns = c("text", "label"))
  
  # --- Train/Test split ---
  split_datasets <- tokenized_dataset$train_test_split(test_size = test_size, seed = seed)
  train_dataset <- split_datasets$train
  eval_dataset <- split_datasets$test
  
  # --- Training arguments ---
  # REGRESSION: choose direction for best model
  greater_is_better <- TRUE
  if (metric_for_best_model %in% c("rmse", "mae", "mse")) {
    greater_is_better <- FALSE
  }
  
  training_args <- transformers$TrainingArguments(
    output_dir = dir,
    num_train_epochs = as.integer(num_train_epochs),
    per_device_train_batch_size = as.integer(per_device_train_batch_size),
    per_device_eval_batch_size = as.integer(per_device_eval_batch_size),
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = TRUE,
    metric_for_best_model = metric_for_best_model,
    greater_is_better = greater_is_better,
    logging_dir = file.path(dir, "logs"),
    logging_steps = 1L,
    seed = seed,
    save_total_limit = 1L,
    learning_rate = learning_rate,
    weight_decay = weight_decay
  )
  
  # --- Metrics ---
  if (return_overflowing_tokens) {
    message("📏 Using document-level metrics (chunk aggregation).")
    
    compute_metrics <- function(eval_pred) {
      preds <- eval_pred$predictions
      y_true <- eval_pred$label_ids
      
      # preds often shape: (N, 1) for regression
      preds <- numpy$array(preds)
      if (length(dim(preds)) == 2L && dim(preds)[2] == 1L) {
        preds <- preds[, 1]
      }
      
      text_ids <- numpy$array(eval_dataset[["text_id"]])
      
      df_results <- data.frame(
        text_id = as.integer(text_ids),
        y_true = as.numeric(y_true),
        y_pred = as.numeric(preds)
      )
      
      # Aggregate per document (mean over chunks)
      agg <- aggregate(
        cbind(y_pred, y_true) ~ text_id,
        data = df_results,
        FUN = mean
      )
      
      yhat <- agg$y_pred
      y <- agg$y_true
      
      message("Var(y) = ", var(y))
      message("MSE model = ", mean((y - yhat)^2))
      message("Baseline MSE = ", mean((y - mean(y))^2))
      
      mse <- metrics$mean_squared_error(y, yhat)
      rmse <- sqrt(mse)
      mae <- metrics$mean_absolute_error(y, yhat)
      r2 <- metrics$r2_score(y, yhat)
      
      list(mse = mse, rmse = rmse, mae = mae, r2 = r2)
    }
    
  } else {
    message("⚡ Using normal evaluation (no chunking).")
    
    compute_metrics <- function(eval_pred) {
      preds <- eval_pred$predictions
      y_true <- eval_pred$label_ids
      
      preds <- numpy$array(preds)
      if (length(dim(preds)) == 2L && dim(preds)[2] == 1L) {
        preds <- preds[, 1]
      }
      
      yhat <- as.numeric(preds)
      y <- as.numeric(y_true)
      
      message("Var(y) = ", var(y))
      message("MSE model = ", mean((y - yhat)^2))
      message("Baseline MSE = ", mean((y - mean(y))^2))
      
      mse <- metrics$mean_squared_error(y, yhat)
      rmse <- sqrt(mse)
      mae <- metrics$mean_absolute_error(y, yhat)
      r2 <- metrics$r2_score(y, yhat)
      
      list(mse = mse, rmse = rmse, mae = mae, r2 = r2)
    }
  }
  
  # --- Trainer ---
  trainer <- transformers$Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics
  )
  
  trainer$train()
  
  # --- Save best model ---
  best_model_dir <- file.path(dir, "best_model")
  dir.create(best_model_dir, recursive = TRUE, showWarnings = FALSE)
  trainer$model$save_pretrained(best_model_dir)
  tokenizer$save_pretrained(best_model_dir)
  
  # --- Remove checkpoints ---
  unlink(list.files(dir, pattern="checkpoint", full.names=TRUE), recursive=TRUE)
  
  return(trainer)
}

#############################
# FUNCTION to predict text after fine-tuning an encoder
#############################

######################
# for classification tasks
######################

process_encoder_classification <- function(texts, model, tokenizer, 
                                   max_length = 512L, stride = 128L, 
                                   mode = c("predict", "cls"),
                                   device = NULL,
                                   return_overflowing_tokens = TRUE,
                                   return_probs = FALSE) {
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  
  if (is.null(device)) device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  
  mode <- match.arg(mode)
  model$eval()
  model$to(device)
  
  all_outputs <- list()
  all_text_ids <- list()
  
  for (text_id in seq_along(texts)) {
    encodings <- tokenizer(
      texts[[text_id]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length),
      stride = as.integer(stride),
      return_overflowing_tokens = return_overflowing_tokens,
      return_tensors = "pt"
    )
    
    num_chunks <- encodings$input_ids$shape[0]
    
    inputs_list <- list(
      input_ids = encodings$input_ids$to(device),
      attention_mask = encodings$attention_mask$to(device)
    )
    
    with(torch$no_grad(), {
      outputs <- do.call(model, inputs_list)
    })
    
    if (mode == "predict") {
      logits <- outputs$logits$detach()$cpu()$numpy()
      all_outputs[[text_id]] <- logits
      
    } else if (mode == "cls") {
      cls_vectors <- outputs$last_hidden_state[, 0L, ]$detach()$cpu()$numpy()
      all_outputs[[text_id]] <- cls_vectors
    }
    
    all_text_ids[[text_id]] <- rep(text_id, num_chunks)
  }
  
  output_matrix <- do.call(rbind, all_outputs)
  text_id_vector <- unlist(all_text_ids)
  
  df_results <- data.frame(
    text_id = text_id_vector,
    output_matrix
  )
  
  agg_outputs <- aggregate(. ~ text_id, data = df_results, FUN = mean)
  
  if (mode == "predict") {
    logits_matrix <- as.matrix(agg_outputs[, -1])
    
    if (return_probs) {
      # Softmax function for numerical stability
      softmax <- function(x) {
        x <- x - max(x)
        exp_x <- exp(x)
        exp_x / sum(exp_x)
      }
      
      probs <- t(apply(logits_matrix, 1, softmax))
      return(probs)
      
    } else {
      predictions <- apply(logits_matrix, 1, which.max) - 1
      return(list(logits = logits_matrix, predictions = predictions))
    }
    
  } else if (mode == "cls") {
    cls_vectors <- as.matrix(agg_outputs[, -1])
    rownames(cls_vectors) <- NULL
    return(cls_vectors)
  }
}

######################
# for regression tasks
######################

process_encoder_regression <- function(
    texts, model, tokenizer,
    max_length = 512L, stride = 128L,
    mode = c("predict", "cls"),
    device = NULL,
    return_overflowing_tokens = TRUE,
    clip_range = NULL   # e.g. c(1,5) if you want predictions constrained
) {
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  
  if (is.null(device)) device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  
  mode <- match.arg(mode)
  model$eval()
  model$to(device)
  
  all_outputs <- list()
  all_text_ids <- list()
  
  for (text_id in seq_along(texts)) {
    encodings <- tokenizer(
      texts[[text_id]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length),
      stride = as.integer(stride),
      return_overflowing_tokens = return_overflowing_tokens,
      return_tensors = "pt"
    )
    
    num_chunks <- encodings$input_ids$shape[0]
    
    inputs_list <- list(
      input_ids = encodings$input_ids$to(device),
      attention_mask = encodings$attention_mask$to(device)
    )
    
    with(torch$no_grad(), {
      outputs <- do.call(model, inputs_list)
    })
    
    if (mode == "predict") {
      # REGRESSION: logits shape is usually (num_chunks, 1)
      preds <- outputs$logits$detach()$cpu()$numpy()
      all_outputs[[text_id]] <- preds
      
    } else if (mode == "cls") {
      cls_vectors <- outputs$last_hidden_state[, 0L, ]$detach()$cpu()$numpy()
      all_outputs[[text_id]] <- cls_vectors
    }
    
    all_text_ids[[text_id]] <- rep(text_id, num_chunks)
  }
  
  output_matrix <- do.call(rbind, all_outputs)
  text_id_vector <- unlist(all_text_ids)
  
  df_results <- data.frame(
    text_id = text_id_vector,
    output_matrix
  )
  
  agg_outputs <- aggregate(. ~ text_id, data = df_results, FUN = mean)
  
  if (mode == "predict") {
    # For regression: return a numeric vector (one value per text)
    pred_mat <- as.matrix(agg_outputs[, -1, drop = FALSE])
    
    # pred_mat may be Nx1; convert to vector
    yhat <- as.numeric(pred_mat[, 1])
    
    if (!is.null(clip_range)) {
      yhat <- pmin(pmax(yhat, clip_range[1]), clip_range[2])
    }
    
    return(list(predictions = yhat, raw = pred_mat))
    
  } else if (mode == "cls") {
    cls_vectors <- as.matrix(agg_outputs[, -1])
    rownames(cls_vectors) <- NULL
    return(cls_vectors)
  }
}

#############################
# FUNCTION to run a NLI, zero-shot or sentiment
#############################

#############################
# FUNCTION to extract the results of a NLI  and a sentiment classifier
#############################

make_nli_df_auto <- function(results, premises, hypothesis) {
  first_item <- results[[1]]
  if (is.list(first_item[[1]]) && !is.null(first_item[[1]]$label)) {
    df <- lapply(seq_along(results), function(i) {
      tibble(
        premise = rep(premises[i], length(results[[i]])),
        hypothesis = rep(hypothesis[i], length(results[[i]])),
        label = sapply(results[[i]], `[[`, "label"),
        score = sapply(results[[i]], `[[`, "score")
      )
    }) %>%
      bind_rows() %>%
      tidyr::pivot_wider(names_from = label, values_from = score)
  } else {
    df <- tibble(
      premise = premises,
      hypothesis = hypothesis,
      label = sapply(results, `[[`, "label"),
      score = sapply(results, `[[`, "score")
    )
  }
  
  df
}

#############################
# FUNCTION to extract the results of a zero-shot model
#############################

make_zero_shot_df <- function(results) {
  # If only one input
  if (!is.list(results[[1]])) {
    results <- list(results)
  }
  
  df <- lapply(seq_along(results), function(i) {
    tibble(
      text = results[[i]]$sequence,
      label = results[[i]]$labels,
      score = results[[i]]$scores
    )
  }) %>%
    dplyr::bind_rows()
  
  # optional: wide version
  df_wide <- df %>%
    tidyr::pivot_wider(names_from = label, values_from = score)
  
  return(df_wide)
}

#############################
# FUNCTION to fine-tune a NLI model 
#############################

train_nli_classifier <- function(model_name, num_labels, premises, hypotheses, labels, 
                                     test_size = 0.5, 
                                     num_train_epochs = 3L, 
                                     metric_for_best_model = "f1", 
                                     seed = 123L,
                                     max_length = 512L,
                                     per_device_train_batch_size = 2L,
                                     per_device_eval_batch_size = 2L,
                                     dir = "./results_nli",
                                     learning_rate = 5e-05,              
                                     weight_decay = 0.0,    
                                     device = NULL,
                                     deterministic = TRUE) {
  
  # --- Imports ---
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  datasets <- import("datasets")
  random <- import("random")
  metrics <- import("sklearn.metrics")
  warnings <- import("warnings")
  
  # warnings$filterwarnings("ignore")
  
  # --- Device selection ---
  if (is.null(device)) {
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  message("🖥️ Using device: ", device)
  
  if (device == "cuda") {
    
    if (deterministic) {
      # Reproducible but TRAINER-safe
      torch$manual_seed(seed)
      torch$cuda$manual_seed_all(seed)
      
      torch$backends$cudnn$deterministic <- TRUE
      torch$backends$cudnn$benchmark <- FALSE
      
      # IMPORTANT: do NOT enforce deterministic algorithms globally
      torch$use_deterministic_algorithms(FALSE)
      
      message("⚠️ GPU training: using reproducible (not strictly deterministic) mode")
    } else {
      torch$backends$cudnn$benchmark <- TRUE
    }
    
  } else {
    torch$use_deterministic_algorithms(FALSE) # CPU mode
  }
  
  # --- Reproducibility ---
  random$seed(seed)
  numpy$random$seed(seed)
  torch$manual_seed(seed)
  
  # --- Sanity check ---
  if (!(length(premises) == length(hypotheses) && length(premises) == length(labels))) {
    stop("premises, hypotheses, and labels must have the same length.")
  }
  
  # --- DataFrame setup ---
  df <- data.frame(premise = premises, hypothesis = hypotheses, label = as.integer(labels))
  transformers$logging$set_verbosity_error()
  
  # --- Tokenizer ---
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
  
  # --- Model setup with GPU ---
  if (num_labels == 2L) {
    id2label <- dict("0"="NOT_ENTAILMENT", "1"="ENTAILMENT")
    label2id <- dict("NOT_ENTAILMENT"=0L, "ENTAILMENT"=1L)
    
  } else if (num_labels == 3L) {
    
    # Load pretrained model config to get its native label order
    tmp_model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_name)
    id2label_r <- reticulate::py_to_r(tmp_model$config$id2label)
    
    # Build python dicts from the model's mapping
    id2label <- dict()
    label2id <- dict()
    
    # id2label_r should be a named list with names like "0","1","2"
    # (if not, we fall back to 0:(n-1))
    if (is.null(names(id2label_r))) {
      names(id2label_r) <- as.character(seq_along(id2label_r) - 1L)
    }
    
    for (k in names(id2label_r)) {
      v <- as.character(id2label_r[[k]])
      id2label[[as.character(k)]] <- v
      label2id[[v]] <- as.integer(k)
    }
    
    # (optional) free tmp_model reference
    rm(tmp_model)
    
  }
  
  else {
    stop("num_labels must be 2 or 3.")
  }
  
  model <- transformers$AutoModelForSequenceClassification$from_pretrained(
    model_name,
    num_labels = as.integer(num_labels),
    id2label = id2label,
    label2id = label2id,
    ignore_mismatched_sizes = TRUE,
    torch_dtype = torch$float32 # added
  )$to(device)  # <--- GPU
  
  model <- model$to(dtype = torch$float32)  # added

  # --- Dataset ---
  dataset <- datasets$Dataset$from_pandas(df)
  
  tokenize_function <- function(examples) {
    encodings <- tokenizer(
      examples[["premise"]],
      examples[["hypothesis"]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length)
    )
    list(
      input_ids = encodings$input_ids,
      attention_mask = encodings$attention_mask,
      labels = as.integer(examples[["label"]])
    )
  }
  
  tokenized_dataset <- dataset$map(
    tokenize_function, 
    batched = TRUE, 
    remove_columns = c("premise", "hypothesis", "label")
  )
  
  split_datasets <- tokenized_dataset$train_test_split(test_size = test_size, seed = seed)
  train_dataset <- split_datasets$train
  eval_dataset <- split_datasets$test
  
  # --- Training arguments ---
  training_args <- transformers$TrainingArguments(
    output_dir = dir,
    num_train_epochs = as.integer(num_train_epochs),
    per_device_train_batch_size = as.integer(per_device_train_batch_size),
    per_device_eval_batch_size = as.integer(per_device_eval_batch_size),
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = TRUE,
    metric_for_best_model = metric_for_best_model,
    greater_is_better = TRUE,
    logging_dir = file.path(dir, "logs"),
    logging_steps = 1L,
    seed = seed,
    save_total_limit = 1L,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    max_grad_norm = 1.0  # to prevent exploding gradients
  )
  
  # --- Metrics computation ---
  compute_metrics <- function(eval_pred) {
    preds_raw  <- eval_pred$predictions
    labels <- eval_pred$label_ids
    
    
    # ---- STEP 1: extract logits safely ----
    if (is.list(preds_raw)) {
      # encoder–decoder models (BART, T5)
      logits <- preds_raw[[1]]
    } else {
      # encoder-only models (BERT, DeBERTa)
      logits <- preds_raw
    }
    
    # ---- STEP 2: ensure matrix shape ----
    if (is.null(dim(logits))) {
      logits <- matrix(logits, nrow = 1)
    }
    
    cat("logits length:", length(logits), "\n")
    cat("first logit:", logits[[1]], "\n")
    
    if (any(!numpy$isfinite(logits))) {
      cat("!!! Non-finite logits detected\n")
    }
    
    # Argmax over label dimension
    preds <- apply(logits, 1, which.max) - 1
    
    accuracy <- metrics$accuracy_score(labels, preds)
    precision <- metrics$precision_score(labels, preds, average = "macro", zero_division = 0)
    recall <- metrics$recall_score(labels, preds, average = "macro", zero_division = 0)
    f1 <- metrics$f1_score(labels, preds, average = "macro", zero_division = 0)
    composite_score <- 0.5 * f1 + 0.3 * recall + 0.2 * accuracy
    list(
      accuracy = accuracy,
      precision = precision,
      recall = recall,
      f1 = f1,
      composite_score = composite_score
    )
  }
  
  # --- Trainer ---
  trainer <- transformers$Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics
  )
  
  trainer$train()
  
  # --- Save best model ---
  best_model_dir <- file.path(dir, "best_model")
  dir.create(best_model_dir, showWarnings = FALSE, recursive = TRUE)
  trainer$model$save_pretrained(best_model_dir)
  tokenizer$save_pretrained(best_model_dir)
  
  # Cleanup checkpoints
  checkpoint_dirs <- list.files(dir, pattern = "checkpoint", full.names = TRUE)
  unlink(checkpoint_dirs, recursive = TRUE)
  
  return(trainer)
}

#############################
# FUNCTION to fine-tune a sentiment classifier
#############################

train_sentiment_classifier <- function(premises, labels,
                                           model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                                           num_labels = NULL,
                                           test_size = 0.2,
                                           num_train_epochs = 3L,
                                           metric_for_best_model = "f1",
                                           seed = 123L,
                                           max_length = 128L,
                                           per_device_train_batch_size = 8L,
                                           per_device_eval_batch_size = 8L,
                                           dir = "./results_sentiment",
                                           learning_rate = 5e-5,
                                           weight_decay = 0.0,
                                           device = NULL,
                                           deterministic = TRUE) {
  
  # --- Imports ---
  transformers <- import("transformers")
  torch <- import("torch")
  numpy <- import("numpy")
  datasets <- import("datasets")
  random <- import("random")
  metrics <- import("sklearn.metrics")
  warnings <- import("warnings")
  
   warnings$filterwarnings("ignore")
  
  # --- Device selection ---
  if (is.null(device)) {
    device <- if (torch$cuda$is_available()) "cuda" else "cpu"
  }
  message("🖥️ Using device: ", device)
  
if (device == "cuda") {

  if (deterministic) {
    # Reproducible but TRAINER-safe
    torch$manual_seed(seed)
    torch$cuda$manual_seed_all(seed)

    torch$backends$cudnn$deterministic <- TRUE
    torch$backends$cudnn$benchmark <- FALSE

    # IMPORTANT: do NOT enforce deterministic algorithms globally
    torch$use_deterministic_algorithms(FALSE)

    message("⚠️ GPU training: using reproducible (not strictly deterministic) mode")
  } else {
    torch$backends$cudnn$benchmark <- TRUE
  }

} else {
  torch$use_deterministic_algorithms(FALSE) # CPU mode
}
  
  # --- Reproducibility ---
  random$seed(seed)
  numpy$random$seed(seed)
  torch$manual_seed(seed)
  
  # --- Sanity check ---
  if (length(premises) != length(labels)) {
    stop("premises and labels must have the same length.")
  }
  
  # --- DataFrame ---
  df <- data.frame(text = premises, label = as.integer(labels), text_id = seq_along(premises))
  transformers$logging$set_verbosity_error()
  
  # --- Tokenizer ---
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
  
  # --- Detect number of labels if not set ---
  if (is.null(num_labels)) {
    base_model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_name)
    num_labels <- base_model$config$num_labels
    rm(base_model)
  }
  
  cat(sprintf("→ Using model: %s with %d labels\n", model_name, num_labels))
  
  # --- Model ---
  model <- transformers$AutoModelForSequenceClassification$from_pretrained(
    model_name,
    num_labels = as.integer(num_labels),
    ignore_mismatched_sizes = TRUE
  )$to(device)
  
  # --- Dataset ---
  dataset <- datasets$Dataset$from_pandas(df)
  
  tokenize_function <- function(examples) {
    encodings <- tokenizer(
      examples[["text"]],
      padding = "max_length",
      truncation = TRUE,
      max_length = as.integer(max_length)
    )
    list(
      input_ids = encodings$input_ids,
      attention_mask = encodings$attention_mask,
      labels = as.integer(examples[["label"]]),
      text_id = as.integer(examples[["text_id"]])
    )
  }
  
  tokenized_dataset <- dataset$map(
    tokenize_function,
    batched = TRUE,
    remove_columns = c("text", "label")
  )

  split_datasets <- tokenized_dataset$train_test_split(test_size = test_size, seed = seed)
  train_dataset <- split_datasets$train
  eval_dataset <- split_datasets$test
  
  # --- Training Arguments ---
  training_args <- transformers$TrainingArguments(
    output_dir = dir,
    num_train_epochs = as.integer(num_train_epochs),
    per_device_train_batch_size = as.integer(per_device_train_batch_size),
    per_device_eval_batch_size = as.integer(per_device_eval_batch_size),
    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = TRUE,
    metric_for_best_model = metric_for_best_model,
    greater_is_better = TRUE,
    learning_rate = learning_rate,
    weight_decay = weight_decay,
    logging_dir = file.path(dir, "logs"),
    logging_steps = 1L,
    seed = seed,
    save_total_limit = 1L
  )
  
  # --- Metrics ---
  compute_metrics <- function(eval_pred) {
    logits <- eval_pred$predictions
    labels <- eval_pred$label_ids
    predictions <- apply(logits, 1, function(x) which.max(x) - 1)
    
    accuracy <- metrics$accuracy_score(labels, predictions)
    precision <- metrics$precision_score(labels, predictions, average = "macro", zero_division = 0)
    recall <- metrics$recall_score(labels, predictions, average = "macro", zero_division = 0)
    f1 <- metrics$f1_score(labels, predictions, average = "macro", zero_division = 0)
    
    list(accuracy = accuracy, precision = precision, recall = recall, f1 = f1)
  }
  
  # --- Trainer ---
  trainer <- transformers$Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics
  )
  
  cat(sprintf("Starting fine-tuning on %s...\n", device))
  trainer$train()
  cat("Training complete. Saving model...\n")
  
  best_model_dir <- file.path(dir, "best_model")
  dir.create(best_model_dir, showWarnings = FALSE, recursive = TRUE)
  trainer$model$save_pretrained(best_model_dir)
  tokenizer$save_pretrained(best_model_dir)
  cat(sprintf("Model and tokenizer saved to: %s\n", best_model_dir))
  
  # Cleanup checkpoints
  checkpoint_dirs <- list.files(dir, pattern = "checkpoint", full.names = TRUE)
  unlink(checkpoint_dirs, recursive = TRUE)
  
  cat("Done!\n")
  return(trainer)
}
