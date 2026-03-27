rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(reticulate)
conda_list()
use_condaenv("python_lib", required = TRUE)

library(dplyr)
transformers <- import("transformers")

#################################################
#################################################
# let's fine-tune a NLI 
#################################################
#################################################

#################################################
# let's keep the usual three NLI labels
#################################################

# Always check what labels the model uses (in case you're unsure of label mappings):
model <- transformers$AutoModelForSequenceClassification$from_pretrained("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
print(model$config$id2label)

# 0 → entailment  
# 1 → neutral  
# 2 → contradiction

# If you do the same with deberta (without the multilingual - just English):
# model <- transformers$AutoModelForSequenceClassification$from_pretrained("cross-encoder/nli-deberta-v3-base")
# print(model$config$id2label)
# You would get:
# $`0`
# [1] "contradiction"
# $`1`
# [1] "entailment"
# $`2`
# [1] "neutral"
# You see that it reverses the order of labels (e.g. 0=contradiction), compared to the previous NLI model
# 0 → contradiction  
# 1 → entailment  
# 2 → neutral

# If you do the same with BART:
# modelBart <- transformers$AutoModelForSequenceClassification$from_pretrained("facebook/bart-large-mnli")
# print(modelBart$config$id2label)
# You would get:
# $`0`
# [1] "contradiction"
# $`1`
# [1] "neutral"
# $`2`
# [1] "entailment"
# 0 → contradiction  
# 1 → neutral  
# 2 → entailment

# let's duplicate the obs to have a couple of more examples to run our fine-tune exercise
premises <- rep(c("The weather is sunny today.", 
                  "A man is playing a guitar.", 
                  "I don't know what to do tonight.",
                  "The cat is sleeping on the couch.",
                  "The restaurant received good reviews."), 2)

hypotheses <- rep(c("It's raining.", 
                    "What a lovely electric sound!", 
                    "I like going to cinema.",
                    "I love sleeping in my bed.",
                    "The food was great."), 2)

# if you use mDeBERTa, you should follow this indexing:
# 0 → entailment  
# 1 → neutral  
# 2 → contradiction

labels_mDeBERTa <- rep(c(2L, 0L, 2L, 1L, 0L), 2)  # 10 examples
labels_mDeBERTa

# if you use BART
# 0 → contradiction  
# 1 → neutral  
# 2 → entailment

labels_bart <- ifelse(labels_mDeBERTa == 0, 2L,
                      ifelse(labels_mDeBERTa == 1, 1L, 0L))
labels_bart

# if you use deberta
# 0 → contradiction  
# 1 → entailment  
# 2 → neutral

labels_deberta <- ifelse(labels_mDeBERTa == 0, 1L,
                         ifelse(labels_mDeBERTa == 2, 0L, 1L))
labels_deberta

# let's call the source of all the functions
source("function ENCODER 2026 LUMACSS.R")
# this is the functiont that we will employ
head(train_nli_classifier)

# Since we are reinitializing the classifier layer, it’s sometimes better to use a smaller 
# learning rate (e.g., 2e-5: 0.00002 vs. 0.00005).

# let's use mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
# around 30 seconds
system.time(trainer <- train_nli_classifier(
  model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
  # model_name = "facebook/bart-large-mnli",
  # model_name="cross-encoder/nli-deberta-v3-base",
  num_labels = 3L, # you have 3 class labels in NLI 
  premises = premises,
  hypotheses = hypotheses,
  labels = labels_mDeBERTa,
  # labels = labels_bart,
  # labels=labels_deberta,
  num_train_epochs = 1L, # Just 1 epoch to make things faster! Never use it!
  test_size = 0.4,
  learning_rate = 1e-05, 
  dir = "./results_nli3",
  device="cpu"
))

# Load fine-tuned model and tokenizer
model_dir <- "./results_nli3/best_model"
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
# our usual three labels
print(model$config$id2label)

nli_classifierFT <- transformers$pipeline(
  "text-classification",
  model = model_dir,
   top_k = NULL
)

# Predict on new premise-hypothesis pair
premise <- c("Our armed forces keep us safe.")
hypothesis <- c("Military is good.")
inputs_pair <- reticulate::dict(text = premise, text_pair = hypothesis)
inputs_pair
resultFT <- nli_classifierFT(inputs_pair)
make_nli_df_auto(resultFT, premises=premise, hypothesis)

# different results compared to when you use the same NLI w/o any fine-tuning
nli_classifierStandard <- transformers$pipeline("text-classification", 
                 model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                 top_k = NULL)
result <- nli_classifierStandard(inputs_pair)
make_nli_df_auto(resultFT, premises=premise, hypothesis)
make_nli_df_auto(result, premises=premise, hypothesis)

#################################################
# let's fine-tune the NLI just with 2 labels
#################################################

library(dplyr)
transformers <- import("transformers")

# If you fine-tune a NLI model with num_labels = 2L, you are not using the original NLI 3-class head anymore. 
# HuggingFace will create a new classification head of size 2.
# More in details, HuggingFace detects:
# pretrained head = 3 outputs
# requested head = 2 outputs
# So it:
# ✔ Keeps the encoder weights
# ✔ Discards the 3-class classifier head (contrary to when you have 3L)
# ✔ Creates a NEW randomly initialized 2-class head
# So now:
# Logit 0 = whatever you define as class 0 0 = NOT_ENTAILMENT
# Logit 1 = whatever you define as class 1 1 = ENTAILMENT

# let's once again duplicate our obs 
premises <- rep(c("The weather is sunny today.", 
                  "A man is playing a guitar.", 
                  "I don't know what to do tonight.",
                  "The cat is sleeping on the couch.",
                  "The restaurant received good reviews."), 2)

hypotheses <- rep(c("It's raining.", 
                    "What a lovely electric sound!", 
                    "I like going to cinema.",
                    "I love sleeping in my bed.",
                    "The food was great."), 2)

labels <- rep(c(0L, 1L, 0L, 0L, 1L), 2)  # where 0=not entailment and 1=entailment - it must be always
# like that for the function train_nli_classifier that we use!
labels

# around 60 seconds - let's avoid to run the model here and let's open the folder with the saved results
# system.time(trainer <- train_nli_classifier(
#    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
#    num_labels = 2L, # you have 2 class labels in NLI now!
#    premises = premises,
#    hypotheses = hypotheses,
#    labels = labels,
#    num_train_epochs = 1L, # very few! Just to make things faster!
#    test_size = 0.4,
#    learning_rate = 2e-5,
#    dir = "./results_nli2Alter",
#    device="cpu"
#  ))

# Load fine-tuned model and tokenizer
model_dir <- "./results_nli2Alter/results_nli2Alter/best_model"
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
# just two labels now!
print(model$config$id2label)
# here we need also to call the tokenizer with two class-labels
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_dir)

nli_classifier2labels <- transformers$pipeline(
  "text-classification",
  model = model,
  tokenizer=tokenizer, # you need to specify the tokenizer as well with 2 class-labels
  top_k = NULL
)

# And then you can use it as always: predict on new premise-hypothesis pair
premise <- c("Our armed forces keep us safe.")
hypothesis <- c("Military is good.")
inputs_pair <- reticulate::dict(text = premise, text_pair = hypothesis)
inputs_pair
result <- nli_classifier2labels(inputs_pair)
print(result)
make_nli_df_auto(result, premises=premise, hypothesis)

#################################################
#################################################
# let's fine-tune a Zero Shot Model 
#################################################
#################################################

# Define toy NLI dataset - while always duplicate the examples
premises <- rep(c(
  "A man is playing guitar on stage.",
  "A woman is reading a book in the park.",
  "A child is sleeping on the bed."
),2)

hypotheses <- rep(c(
  "A musician is performing.",
  "Someone is swimming in a pool.",
  "A kid is awake and playing."
),2)

# Labels: 0=entailment, 1=neutral, 2=contradiction with mDeBERTa
# Suppose we know the ground-truth labels
labels <- rep(c(0L, 2L, 2L),2)
labels

# Run fine-tuning 
# around 30 seconds - let's avoid to run the model here and let's open the folder with the saved results
# system.time(trainer <- train_nli_classifier(
#   model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
#   num_labels = 3L, # three categories
#   premises = premises,
#   hypotheses = hypotheses,
#   labels = labels,
#   num_train_epochs = 1L, # very few! Just to make things faster!
#   per_device_train_batch_size = 2L,
#   per_device_eval_batch_size = 2L,
#   learning_rate = 2e-5,
#   test_size = 0.4,
#   dir = "./results_zero",
#   device="cpu"
# ))

# Load fine-tuned model and tokenizer
model_dir <- "./results_zero/results_zero/best_model"
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
# our usual three labels
print(model$config$id2label)

zero_shot_classifierFT <- transformers$pipeline("zero-shot-classification", 
                                              model = model_dir)

text <- "The restaurant had amazing food but terrible service."
labels <- c("food", "service", "sport")
zero_shotFT<-zero_shot_classifierFT(text, candidate_labels = labels)
make_zero_shot_df(zero_shotFT)

# Different probabilities compared to the one you get w/o fine-tuning
zero_shot_classifier2 <- transformers$pipeline("zero-shot-classification", 
                  model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
zero_shot<-zero_shot_classifier2(text, candidate_labels = labels)
make_zero_shot_df(zero_shotFT)
make_zero_shot_df(zero_shot)

#################################################
#################################################
# let's fine-tune a Sentiment Model 
#################################################
#################################################

# this functions works with any # of labels of classifiers (not just 2)
head(train_sentiment_classifier)

# always check the labels!
model <- transformers$AutoModelForSequenceClassification$from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
print(model$config$id2label)

# let's duplicate the obs
texts <- rep(c(
  "I love this movie!",
  "This was awful.",
  "The food was great.",
  "I’ll never go back."
),2)
labels <- rep(c(1, 0, 1, 0),2)  # 1 = positive, 0 = negative
labels

# around 10 seconds
system.time(trainer <- train_sentiment_classifier(
  premises = texts,
  labels = labels,
  model_name = "distilbert-base-uncased-finetuned-sst-2-english",
  num_labels = 2L,
  num_train_epochs = 1L,
  learning_rate = 2e-5,
  dir = "./results_sst2",
  device="cpu"
))

# Load fine-tuned model and tokenizer
model_dir <- "./results_sst2/best_model"

model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
# our usual three labels
print(model$config$id2label)

sentiment_pipelineFT <- transformers$pipeline("sentiment-analysis", 
                                            model = model_dir)

# Example text input
texts <- c("I love this product!", "This is the worst movie I've ever seen.")

# Apply sentiment analysis
sentiment_resultsFT <- sentiment_pipelineFT(texts) 
make_nli_df_auto(sentiment_resultsFT, texts, hypothesis = NA)

# Multi-class sentiment (5 labels)
# always check the labels!
model <- transformers$AutoModelForSequenceClassification$from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
print(model$config$id2label)

texts <- rep(c(
  "Terrible product!",
  "Not great.",
  "Okay, I guess.",
  "Good job.",
  "Absolutely wonderful!"
),2)
labels <- rep(c(0, 1, 2, 3, 4),2)  # 0 = very negative ... 4 = very positive
labels

# around 10 seconds - let's avoid to run the model here and let's open the folder with the saved results
# system.time(trainer <- train_sentiment_classifier(
#   premises = texts,
#   labels = labels,
#   model_name = "nlptown/bert-base-multilingual-uncased-sentiment",
#   num_labels = 5L,
#   num_train_epochs = 1L,
#   learning_rate = 2e-5,
#   dir = "./results_multilingual",
#   device="cpu"
# ))

# Load fine-tuned model and tokenizer
model_dir <- "./results_multilingual/results_multilingual/best_model"
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
# 5 labels
print(model$config$id2label)

sentiment_pipelineFTmulti <- transformers$pipeline("sentiment-analysis", 
                                              model = model_dir)

# Example text input
texts <- c("I love this product!", "This is the worst movie I've ever seen.")

# Apply sentiment analysis
sentiment_resultsFT2 <- sentiment_pipelineFTmulti(texts) 
make_nli_df_auto(sentiment_resultsFT2, texts, hypothesis = NA)
