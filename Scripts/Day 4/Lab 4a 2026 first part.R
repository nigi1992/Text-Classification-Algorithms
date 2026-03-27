#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()

library(reticulate)
reticulate::conda_list()
# open the virtual env
use_condaenv("python_lib", required = TRUE)

library(dplyr)
library(e1071)
library(caret)
library(ggplot2)
library(cvTools)
library(yardstick)
library(ranger)

# let's call the Python transformers package in R
transformers <- import("transformers")

######################
# fine tuning
#####################

#############################
#############################
# TWO-CLASS toy example 
#############################
#############################

# What we did in the previous class was an exercise of feature extraction from BERT, no fine-tuning!
# So let's run a proper fine-tuning exercise! 

torch <- import("torch")
numpy <- import("numpy")
datasets <- import("datasets")
random <- import("random")  

texts <- c("I love this movie!", "Terrible acting.", "Fantastic plot and cast.", "Worst film ever.")
# Let's suppose that we identify the first and third text as positive (=1), while the second
# and the fourth as negative (=0).
# By adding L after a number in R, you explicitly declare it as an integer. 
# Transformers does not want numeric values!
labels <- c(1L, 0L, 1L, 0L)  # as integers
class(labels)
# alternatively:
labels <- as.integer(c(1, 0, 1, 0))  # as integers not as numeric!
class(labels)
# let's generate our dataframe
df <- data.frame(text=texts, label=labels)
str(df)

# Let's load the tokenizer (same one used in our previous class)
tokenizer <- transformers$AutoTokenizer$from_pretrained("bert-base-uncased")
# Let's load the corresponding model. Here we want to use a model for classification, therefore
# let's use the $AutoModelForSequenceClassification function (remember here to specify
# the number of labels that we want to classify!)
model <- transformers$AutoModelForSequenceClassification$from_pretrained(
  "bert-base-uncased",  num_labels=2L)  

# let's convert our dataframe into a format that can be used by the tokenizer function
dataset <- datasets$Dataset$from_pandas(df)
dataset

# let's create a function to tokenize each texts in our corpus
tokenize_function <- function(examples) {
  tokens <- tokenizer(
    examples[["text"]], 
    padding = "max_length",
    truncation = TRUE 
  )
  tokens$labels <- as.integer(examples[["label"]])
  return(tokens)
}

# let's tokenize our dataframe using the function just created
tokenized_dataset <- dataset$map(tokenize_function, batched=TRUE)
# If batched = TRUE, the function is called once per batch of examples. 
# By default, the library uses a batch size of 1000 examples (i.e., it sends 1000 rows at a time 
# to your function).
# If batched=FALSE (default), the function is called once per example (for each text)

# Let's split the dataset into a training and an evaluation set
split_datasets <- tokenized_dataset$train_test_split(test_size = 0.5, seed = 42L)
train_dataset <- split_datasets$train
eval_dataset <- split_datasets$test

# TrainingArguments object holds all the key configuration settings for your training loop — like 
# where to save results, how long to train, batch sizes, evaluation strategy, etc.
# It’s then passed to a Trainer object, which handles the training process.
# If you get an error like "unused argument (eval_strategy = "epoch")" replace this line with 
# "evaluation_strategy = "epoch" below. It depends on the version of the transformers library 
# you are using

training_args <- transformers$TrainingArguments(
  output_dir = "./resultsFakeExample", # Directory where the trained model and checkpoints 
  # would normally be saved after training.
  num_train_epochs = 3L, # Number of complete passes (epochs) through the training dataset. 
  # In this case: 3 epochs. A typical value is between 3 and 5
  per_device_train_batch_size = 2L, # 2 examples per training batch.
  per_device_eval_batch_size = 2L, # 2 examples per evaluation batch
  eval_strategy  = "epoch", # How often to run evaluation on the validation set.
  # Here: once at the end of each epoch.
  save_strategy = "epoch",  # How often to save model checkpoints. 
  load_best_model_at_end = TRUE, # save the best model (the model that will be employed to predict)
  metric_for_best_model="f1",	# Which metric to track for "best"
  greater_is_better = TRUE, # Whether higher or lower is better for that metric
  logging_dir = "./logs", # Directory to store training logs (e.g., loss, metrics per step or epoch).
  logging_steps = 1L, # How often (in number of steps) to log training metrics. 
  # Here: every 1 step — essentially logging after every batch.
  seed = 42L,  # seed for Trainer
  save_total_limit = 1L # Maximum number of checkpoints to keep if saving is enabled 
  # (e.g. keeping only the latest one) unless, as in our case, load_best_model_at_end = TRUE 
  # in this case we save the best model
)

# note that there are also several other arguments that we do not specify above
training_args

# How Model Selection Works during Training:
# 1) At the end of each epoch, the model is evaluated on your validation set
# 2) The evaluation metrics (accuracy, F1, loss, etc.) are recorded
# 3) A checkpoint is saved (if save_strategy="epoch")
# 4) The trainer continuously compares the current epoch's metrics against 
# previous best. It remembers which checkpoint achieved the optimal metric value
# 5) After the final epoch completes, it automatically reloads the weights from the 
# best-performing epoch. This becomes your final model (not necessarily the last epoch's model).
# So by specifying load_best_model_at_end=TRUE we are going to use to predict some new data the
# model that through the different epochs performed better. 
# If you don’t set load_best_model_at_end = TRUE, you’d be using the model state from the final 
# epoch — which might not be the best one

# Remember: if you don't specify an argument in TrainingArguments, it will use the default value 
# for that argument

# compute_metrics function for evaluation of the fine-tuned model 
metrics  <- import("sklearn.metrics")

compute_metrics <- function(eval_pred) {
  logits <- eval_pred$predictions
  labels <- eval_pred$label_ids
  predictions <- numpy$argmax(logits, axis = -1L)
  accuracy <- metrics$accuracy_score(labels, predictions)
  precision <- metrics$precision_score(labels, predictions)
  recall <- metrics$recall_score(labels, predictions)
  f1 <- metrics$f1_score(labels, predictions, average="macro", zero_division=0)
  # Weighted composite score of f1, recall and accuracy
  composite_score <- 0.5*f1 + 0.3*recall + 0.2*accuracy
  # if you want to use this metric instead of f1 as the metric to employ
  # to select the best model just write: metric_for_best_model="composite_score"
  return(list(
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1 = f1,
    composite_score = composite_score
  ))
}

# let's finally fine-tune BERT
# Let's define our function
trainer <- transformers$Trainer(
  model = model, # Bert as defined above
  args = training_args, # our arguments as defined above
  train_dataset = train_dataset, # our training set
  eval_dataset = eval_dataset, # our evaluation set
  compute_metrics = compute_metrics # the metrics to compute during fine-tuning
)

# let's run it (all the results will be save in the folder defined above!)
trainer$train()

# After training completes:
best_epoch <- trainer$state$best_metric  # Gets the best metric value
best_epoch

# We can also save both our fine-tuned model and tokenizer in a new folder if we want 
# to use them later for example:
# trainer$save_model("./my_fine_tuned_model")
# tokenizer$save_pretrained("./my_fine_tuned_model")

############################################
# To do everything with one function
############################################

# let's call the source of all the functions
source("function ENCODER 2026 LUMACSS.R")

# this is the function that we will employ 
head(train_transformer_classifier)

# by deafult in the function we implement:
# load_best_model_at_end = TRUE # save the best model (the model that will be employed to predict)
# greater_is_better = TRUE # Whether higher or lower is better for that metric

# Typical test_size values to employ:
# Large dataset (≥10,000 samples)	0.1–0.2 
# Medium dataset (1,000–10,000 samples)	0.15–0.25	
# Small dataset (<1,000 samples)	0.2–0.3	
# Very small dataset (<300 samples)	0.3–0.4 

# around 20 seconds - here we specify device="cpu" but we could have selected device="cuda" if we had a GPU
system.time(trainer <- train_transformer_classifier(
  model_name = "bert-base-uncased",
  num_labels = 2L, # binary example
  texts = texts,
  labels = labels,
  test_size = 0.5,
  num_train_epochs = 3,
  device="cpu",
  metric_for_best_model = "f1",
  seed = 42L,
  dir = "./resultsALT"
))

# After training completes:
best_epoch <- trainer$state$best_metric  # Gets the best metric value
# same as above
best_epoch

# When fine-tuning BERT, there are several key hyperparameters to focus on for tuning.
# The most important ones:
# 1) learning-rate - typical range: 1e-5 – 5e-5
# Lower =  slow convergence, more stable, higher = faster but risk of forgetting pretraining.
# 2) Batch Size - typical range: 4–32. Larger batches stabilize training but require more memory
# Smaller batches → more noisy updates but better generalization
# 3) Number of Epochs - typical range: 2 - 5. More → risk of overfitting, less → underfitting.
# 4) Weight Decay (L2 regularization) - typical values: 0.01 and 0.1 It prevents overfitting

# for example, let's increase the value for weight_decay (do not run!)
head(train_transformer_classifier)

# system.time(trainer <- train_transformer_classifier_longtextGPU2(
#   model_name = "bert-base-uncased",
#   num_labels = 2L, # binary example
#   texts = texts,
#   labels = labels,
#   test_size = 0.5,
#   num_train_epochs = 3,
#   metric_for_best_model = "f1",
#   seed = 42L,
#   device="cpu",
#   weight_decay=0.1,
#   dir = "./resultsALT2"
# ))

# Note that you can also change the architecture of the NN model on top of your encoder (for
# example adding an hidden layer, changing the dropout rate - default 0.1, sometimes 0.2–0.3 improves
# generalization). However doing that it is a bit me complex. So we avoid it here. But if you
# are interest, send me an email!

# What does it change if we use long texts (i.e., > 512)? 
# In the previous function, metrics are computed per-chunk — not per original text.
# But given we have short texts, each text < 512 and that's fine.
# However if you have a text>512, we could have a problem, given that we will have metrics (such
# as f1) computed at the level of the chunk, not at the level of the text. But you want the latter!
# However if you have a long text, these could be divided in several chunks, and you won't ever
# get the metrics at the level of the chunks. So what to do? If a text is longer than 512 tokens,
# then it will be divided in chunks, each chunk will get a metric, but then an avg. metric for
# the text will be computed

texts <- c(
  "I love this movie!", 
  "Terrible acting.", 
  "Fantastic plot and cast.", 
  "Worst film ever.",
  paste(rep("This is a long boring movie.", 200), collapse = " ")
)

labels <- as.integer(c(1, 0, 1, 0, 0))
str(labels)

# around 52 seconds
system.time(trainer <- train_transformer_classifier(
  model_name = "bert-base-uncased",
  num_labels = 2L,
  texts = texts,
  labels = labels,
  test_size = 0.5,
  num_train_epochs = 3,
  metric_for_best_model = "f1",
  max_length = 512L,
  stride = 128L,
  device="cpu",
  seed = 42L,
  dir = "./results_long"
))

# After training completes:
best_epoch <- trainer$state$best_metric  # Gets the best metric value
best_epoch

#############################
#############################
# fine tuning with a real dataset
#############################
#############################

# let's focus on a new dataset related to social disaster (from here: https://www.kaggle.com/datasets/vstepanenko/disaster-tweets)
# and let's generate out of this dataset a training-set, a validation-set and a data-set to be 
# employed to fine-tune our encoder model (here: BERT)

set_500 <- readRDS("Input Data/Day 4/Lab4a2026/set_500.RDS") # training-set
set_300 <- readRDS("Input Data/Day 4/Lab4a2026/set_300.RDS") # fine-tuning set
set_200 <- readRDS("Input Data/Day 4/Lab4a2026/set_200.RDS") # validation-set

# let's now fine-tune BERT to understand if we can improve on the previous results
set_300 <- readRDS("Input Data/Day 4/Lab4a2026/set_300.RDS") # fine-tuning set
str(set_300)
class(set_300$target) # our target variable is already an integer. Good!
labels <- set_300$target
str(labels)

# around 40 minutes - do not run here. Let's save the results in a folder to download on the home-page of 
# the course
# system.time(trainer <- train_transformer_classifier(
#     model_name = "bert-base-uncased",
#     num_labels = 2L, # binary example
#     texts = set_300$text,
#     labels = labels,
#     per_device_train_batch_size = 10L, # 10 examples per training batch.
#     per_device_eval_batch_size = 10L, # 10 examples per evaluation batch
#     test_size = 0.3,
#     num_train_epochs = 3,
#     metric_for_best_model = "f1",
#     seed = 42L,
#     device="cpu",
#     dir = "./results"
#   ))

# let's save the most important info of the fitted model
#   trainer_info <- list(
#     best_metric = trainer$state$best_metric,
#     best_model_checkpoint = trainer$state$best_model_checkpoint,
#     epoch = trainer$state$epoch,
#     log_history = trainer$state$log_history
# )

#  saveRDS(trainer_info, "Input Data/Day 4/Lab4a2026/trainer_exFT.RDS")
 
trainer_info <- readRDS("Input Data/Day 4/Lab4a2026/trainer_exFT.RDS")
best_epoch <- trainer_info$best_metric # Gets the best metric value
best_checkpoint <- trainer_info$best_model_checkpoint # Gets the path to best model
print(paste("Best F1 score:", best_epoch))
print(paste("Best model path:", best_checkpoint)) # checkpoint-42 means “after 42 training steps”

# we have 300 samples (set_300)
# test_size = 0.3 → 70% train = 210 samples
# Batch size = 10 → 210 / 10 = 21 steps per epoch
# Then: 3 epochs × 21 steps/epoch = 63 total steps
# Therefore, our “best model path” being checkpoint-42 means the best model was saved at the end of 
# the 2nd epoch.
trainer_info$epoch

log_df <- bind_rows(trainer_info$log_history)
head(log_df)

# let's extract the performance results over the different epochs
log_df2 <- log_df[!is.na(log_df$eval_f1), ]
str(log_df2)

ggplot(log_df2, aes(x = epoch, y = eval_f1)) +
  geom_point() +
  geom_line() +
  theme_minimal()

############################################
# Let's use the fine tuned model to predict new texts
############################################

# Let's load the tokenizer and the model for the best model computed via fine-tuning 
model_dir <- "./results/results/best_model"
model_classifier <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_dir)

# Get predictions for the validation-set
str(set_200)

# What is happening right now: You are no longer training the model.
# The model you pass in contains:
# 1) The fine-tuned BERT encoder weights
# 2) The fine-tuned classification head weights (the linear layer you trained)
# Both sets of weights are now frozen (in inference mode).
# So during prediction:
# a) The tokenizer converts your new texts into token IDs and attention masks.
# b) The model passes them through the fine-tuned BERT encoder.
# c) The output from the [CLS] token embedding goes into the trained linear layer.
# d) The linear layer produces logits → probabilities → predicted labels.
# No gradient updates, no training — it’s a forward pass only using the saved fine-tuned weights.

# this is the function we will use. If you want to get just the predicted probabilities,
# you can specify return_probs=TRUE
head(process_encoder_classification )

# around 100 seconds
# system.time(predictions <- process_encoder_classification(set_200$text, model_classifier, tokenizer, 
#                                               mode = "predict", device="cpu"))
# saveRDS(predictions, "Input Data/Day 4/Lab4a2026/predictionsBIS.RDS")
predictions <- readRDS("Input Data/Day 4/Lab4a2026/predictions.RDS")
str(predictions)
table(predictions$predictions)

# if you want the probabilities
softmax <- function(x) exp(x) / rowSums(exp(x))
probs <- softmax(predictions$logits)
head(probs)

# let's check our predictions vs. true values for set_200$target
mt_ft <- table(as.factor(predictions$predictions), as.factor(set_200$target))
mt_ft
# 84% with balanced accuracy with fine tuning. Improving on a model with dynamic embeddings but 
# w/o fine-tuning (you can check for that by yourself!)
confusionMatrix(mt_ft)

# Of course we are always called to internally validate our predictions. We can follow
# the approach we adopted to enter into the "black box" of an encoder model in the previous lab and 
# using such approach also here (I leave that to you)

# The previous End-to-End Fine-Tuning works quite good for most NLP tasks. However it:
# a) forces you to use the linear layer for prediction; b) it is rather complex to use
# to run a CV exercise. Indeed, fine-tuning an entire BERT model inside a k-fold CV loop is
# not only computationally expensive (since BERT is big, and you'd retrain it k times)
# but also bit awkward because you typically have to fine-tune and save/restore multiple large models 
# for each fold

# As an alternative, we can:
# a) feed new texts (training and test-set) through the fine-tuned BERT model, up to and including 
# the transformer encoder stack, but NOT the classification head.
# b) Extract the vector corresponding to the [CLS] token from the last hidden state.
# c) Use that [CLS] vector as a fixed-length feature representation of the entire input text.
# When you fine-tune the model on your task, the [CLS] vector’s meaning subtly adapts to the new task 
# — so extracting the [CLS] after fine-tuning and using it in a separate downstream model 
# makes it more task-relevant than using a pretrained-only BERT.
# d) Feed that feature vector into a completely different classifier (SVM, RF, etc.)

# So let's get the CLS vectors after fine-tuning! Not that here I don't use as above
# transformers$AutoModelForSequenceClassification$from_pretrained cause I am not considering
# anymore the classification head!

# Use the one below if you use the folder with the zipped file (as in my case):
model_cls <- transformers$AutoModel$from_pretrained("./results/results/best_model/")
# Get the dynamic WE of the [CLS] token for the training-set after fine-tuning.
# Note that below now mode = "cls"
# You would need around 4 minutes:
# system.time(cls_vectorsFT_train <- process_encoder_classification(set_500$text, model_cls, tokenizer, 
#                                                      mode = "cls", device="cpu"))
# saveRDS(cls_vectorsFT_train, "Input Data/Day 4/Lab4a2026/cls_vectorsFT_trainBIS.RDS")
cls_vectorsFT_train <- readRDS("Input Data/Day 4/Lab4a2026/cls_vectorsFT_trainBIS.RDS")

# As an alternative, you could also use our old friend the get_text_embeddings_stride function.
# The advantage of get_text_embeddings_stride is that you can choose the layers and also keep
# the "avg" value (not only extracting the "cls")
# In this case remember to pass to the function a path string (not the model object)
# use the one below if you use the folder with the zipped file
# system.time(cls_embeddings <- get_text_embeddings_stride ("./results/results/best_model/", 12, 
#                                                               set_500$text, "cls", device="cpu"))

# around 90 seconds
# get the dynamic WE for the validation-set after fine-tuning using our fine-tuned BERT
# system.time(cls_vectorsFT <- process_encoder_classification(set_200$text, model_cls, tokenizer, mode = "cls",
#                                                    device="cpu"))
# saveRDS(cls_vectorsFT, "Input Data/Day 4/Lab4a2026/cls_vectorsFTBIS.RDS")
cls_vectorsFT <- readRDS("Input Data/Day 4/Lab4a2026/cls_vectorsFTBIS.RDS")

set.seed(123)
system.time(RF3 <- ranger(y=as.factor(set_500$target), x=cls_vectorsFT_train))
set.seed(123)
predict_testFT <- predict(RF3, cls_vectorsFT)
str(predict_testFT)
mt_cls_ft <- table(predict_testFT$predictions, as.factor(set_200$target))
mt_cls_ft
# balanced accuracy: 83.5%
confusionMatrix(mt_cls_ft)

# of course, we can now run a CV exercise with the cls vectors from a BERT with fine tuning
source("Function 2026 LUMACSS.R")
head(Function_RF)

rf_cls <- Function_RF(input=cls_vectorsFT_train, k=5, DV=as.factor(set_500$target), ML=ranger)
colMeans(rf_cls[ , c(1, 2, 3, 4)]) 

#############################
#############################
# MULTI-CLASS toy example - nothing substantially change!
#############################
#############################

# Multi-class example: values 0, 1, 2 (0=negative; 1=neutral; 2=positive)
texts <- c("I love this movie!", "Terrible acting.", "Fantastic plot and cast.", 
"I have watched that movie.", "Which is the last movie you
           watched?", "Stallone sucks")
labels <- as.integer(c(2, 0, 2, 1, 1, 0))  
length(texts)
length(labels)

# around 30 seconds
system.time(trainer2 <- train_transformer_classifier(
  model_name = "bert-base-uncased",
  num_labels = 3L, # here you specify 3L cause you have 3-labels!
  texts = texts,
  labels = labels,
  test_size = 0.5,
  num_train_epochs = 3,
  metric_for_best_model = "f1",
  seed = 42L,
  device="cpu",
  dir = "./multi_results"
))

# After training completes:
best_epoch <- trainer2$state$best_metric  # let's get the best metric value
print(paste("Best F1 score:", best_epoch)) # very low f1 - but nothing surprisingly with just 6 texts

# Example texts to predict
new_texts <- c("Awesome movie!", "Terrible!", "The new movie is at the cinema")

# Load tokenizer and model
tokenizer <- transformers$AutoTokenizer$from_pretrained("./multi_results/best_model/")
model_classifier <- transformers$AutoModelForSequenceClassification$from_pretrained("./multi_results/best_model/")

# Get predictions
predictions <- process_encoder_classification(new_texts, model_classifier, tokenizer, mode = "predict",
                                              device="cpu")
# quite horrible! But not surprisingly given our fine-tuning dataset...
print(predictions)
probs <- softmax(predictions$logits)
head(probs)