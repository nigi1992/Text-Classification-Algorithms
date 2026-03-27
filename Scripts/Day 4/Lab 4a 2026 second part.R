rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(dplyr)
library(reticulate)
reticulate::conda_list()
use_condaenv("python_lib", required = TRUE)

library(dplyr)
library(tidyr)

transformers <- import("transformers")

# The transformers pipeline via reticulate is very user-friendly — 
# just specify the task ("sentiment-analysis", zero-shot-classification" or "text-classification")
# and the corresponding model. The functions moreover are deterministic at inference time

##################################
##################################
# NLI
##################################
##################################

transformers$logging$set_verbosity_error() # to avoid to report harmless warnings 

# An alternative model that we could use to run a NLI: "facebook/bart-large-mnli" - see below

# Note that we have specified below "text-classification" in transformers$pipeline.
# By default transformers$pipeline uses CPU even when a cuda/GPU virtual env is available
# To use the latter one, you should specify: device=as.integer(0) or device=0L; 
# device=-1L corresponds to CPU

nli_classifier <- transformers$pipeline("text-classification", 
                  model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

# if you want to use a GPU (assuming you have it)
# nli_classifierGPU <- transformers$pipeline("text-classification", 
# model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", device=as.integer(0))

premise <- "The weather is sunny today."
hypothesis <- "It’s raining."

# Concatenate premise and hypothesis into a format expected by the NLI model
inputs_pair <- reticulate::dict(text = premise, text_pair = hypothesis)
inputs_pair

# Output is related to three labels:
# ENTAILMENT
# NEUTRAL
# CONTRADICTION
nli_classifier(inputs_pair)

# to get the probabilities for the 3 labels you need to specify "top_k = NULL"
nli_classifier2 <- transformers$pipeline("text-classification", 
                                         model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                                         return_all_scores = TRUE, top_k = NULL)
nli_classifier2(inputs_pair)
# let's convert the results into a data frame
results <- nli_classifier2(inputs_pair)
# let's use the function below to extract the results of a NLI and store them in a dataframe
source("Scripts/Day 4/function ENCODER 2026 LUMACSS.R")
head(make_nli_df_auto)
df_wide <- make_nli_df_auto(results, premises=premise, hypothesis)
df_wide

# If you only care about entailment vs non-entailment,we can treat both neutral and contradiction
# as a single "non-entailment" class. We can do this by examining probabilities 
# after inference and combining accordingly.
# Alternatively we can fine-tune a NLI model where a classification head has 2 outputs instead 
# of the usual 3. Advantage: you’ll get a model explicitly optimized for your 2-category task that
# can achieve better performance and clearer decision boundaries
# See later for an implementation!

# multiple premises 
premises <- c("Our armed forces keep us safe.", "I love peace not war!", 
              "The general met the President.")
hypothesis <- c("Military is good.")

#let's pass a list-of-dicts (one example per element) so the pipeline handles each example individually:
inputs_list <- lapply(seq_along(premises), function(i) {
  reticulate::dict(text = premises[i], text_pair = hypothesis)
})
inputs_list

nli_classifier(inputs_list)

# let's use once again the below function to extract the results of a NLI and store them in a dataframe
results_multi <- nli_classifier(inputs_list)
head(make_nli_df_auto)
df_wide <- make_nli_df_auto(results_multi, premises, hypothesis)
df_wide

# multiple premise-hypothesis pairs
premises <- c("The weather is sunny today.", "I just bought a new phone.", "I went to the pool today.")
hypothesis <- c("It’s raining.", "I have a new gadget.", "I do sport.")

inputs_list2 <- lapply(seq_along(premises), function(i) {
  reticulate::dict(
    text = premises[i],
    text_pair = hypothesis[i]
  )
})
inputs_list2

results_multi2 <- nli_classifier(inputs_list2)
df_wide2 <- make_nli_df_auto(results_multi2, premises, hypothesis)
df_wide2

# the nice thing about "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" is that 
# it supports 100+ languages
premises2 <- c("The weather is sunny today.", "Ho appena comprato un nuovo cellulare.", "
今日はプールに行きました.")
hypothesis2 <- c("It’s raining.", "I have a new gadget.", "I do sport.")

inputs_list3 <- lapply(seq_along(premises2), function(i) {
  reticulate::dict(
    text = premises2[i],
    text_pair = hypothesis2[i]
  )
})
inputs_list3
nli_classifier2(inputs_list3)

results_multiHyp <- nli_classifier2(inputs_list3)
df_wideMultiple <- make_nli_df_auto(results_multiHyp, premises2, hypothesis2)
df_wideMultiple

##################################
##################################
# sentiment analysis via the function textClassify
##################################
##################################

# In this case we are applying a Transformer (BERT) that have been already fine tuned with the aim 
# to provide a sentiment classification for English texts, i.e., 
# model="distilbert-base-uncased-finetuned-sst-2-english"

# Note sentiment-analysis in transformers$pipeline.
sentiment_pipeline <- transformers$pipeline("sentiment-analysis", 
             model = "distilbert-base-uncased-finetuned-sst-2-english")

# Example text input
texts <- c("I love this product!", "This is the worst movie I've ever seen.")

# Apply sentiment analysis
sentiment_pipeline(texts) 
sentiment_results <- sentiment_pipeline(texts) 
df_sentiment <- make_nli_df_auto(sentiment_results, texts, hypothesis = NA) # here there are not hypothesis!
df_sentiment

# if you add top_k = NULL you would get the probability for both the negative and the positive label rather
# than the highest of the two
sentiment_pipeline2 <- transformers$pipeline("sentiment-analysis", 
                                            model = "distilbert-base-uncased-finetuned-sst-2-english",
                                            top_k = NULL)

# with all the existing dictionaries, you would get a positive review (give the large number of positive
# tokens...)
testText <- "This movie has good premises. Looks like it has a nice plot, and exceptional cast, 
first class actors and Stallone gives his best. But it sucks"

sentiment_results <- sentiment_pipeline2(testText) 
df_sentiment <- make_nli_df_auto(sentiment_results, testText, hypothesis = NA)
df_sentiment

# multi-lingual sentiment dictionary:
# nlptown/bert-base-multilingual-uncased-sentiment
# → Supports English, French, Spanish, German, Italian, Dutch.
# The nlptown/bert-base-multilingual-uncased-sentiment outputs 5-point star labels (1-5), 
# not just positive/negative.

# Another example:
# cardiffnlp/twitter-xlm-roberta-base-sentiment
# → Based on XLM-Roberta, trained on multilingual Twitter data.

##################################
##################################
# zero-shot classification
##################################
##################################

# You cannot compute a zero-shot classification without a decoder-only model, i.e.,
# you need a generative next-token likelihood model guided by instruction or prompt.
# But you can still rely on formal NLI premise-hypothesis entailment scoring as discussed.
# If you want to run a zero-shot classification using a decoder only model, you can explore 
# the rollama package that implements LLAMA

# Let's use the encoder-decoder model "facebook/bart-large-mnli"

zero_shot_classifier <- transformers$pipeline("zero-shot-classification", 
                                              model = "facebook/bart-large-mnli")

text <- "The restaurant had amazing food but terrible service."
labels <- c("food", "service", "sport")
zero_shot_classifier(text, candidate_labels = labels)

# of course you can also use the previous NLI model in this regard
zero_shot_classifier2 <- transformers$pipeline("zero-shot-classification", 
                        model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
text <- "The restaurant had amazing food but terrible service."
labels <- c("food", "service", "sport")
zero_shot_classifier2(text, candidate_labels = labels)

# if you specify multi_label = TRUE you allow multi-label 
# (e.g. detect both “food” and “service” at once)

zero_shot_classifier22 <- transformers$pipeline("zero-shot-classification", 
                            model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                            multi_label = TRUE)
zero_shot_classifier22(text, candidate_labels = labels)

# now the three probabilities do not add to 1 anymore! Moreover you see that your text
# is classified as discussing both food and service

# let's use this function to extract the results of a zero-shot and store them in a data frame 
head(make_zero_shot_df)

result_zero <- zero_shot_classifier2(text, candidate_labels = labels)
df_zero_shot <- make_zero_shot_df(result_zero)
df_zero_shot

result_zeroMultiLabel <- zero_shot_classifier22(text, candidate_labels = labels)
df_zero_shotMulti <- make_zero_shot_df(result_zeroMultiLabel)
df_zero_shotMulti
