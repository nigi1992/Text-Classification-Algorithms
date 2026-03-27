rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(reticulate)
reticulate::conda_list()
# open the virtual env
use_condaenv("python_lib", required = TRUE)
options(warn = 0) # I delete the warning

library(Rtsne)
library(ggplot2)
library(lsa)  
library(cvTools)
library(ranger)
library(caret)
library(readr)
library(yardstick)
library(naivebayes)
library(quanteda.textstats)
library(quanteda.textplots )
library(quanteda)
library(cowplot)
library(dplyr)
library(tidytext)

# let's call the Python transformers library in R
transformers <- import("transformers")

#############################
#############################
# Step 1: calling a TOKENIZER
#############################
#############################

# Load a tokenizer - for example the "bert-base-uncased"
# Uncased models convert all words to lowercase, whereas cased models (i.e., "bert-base-cased") 
# keep the casing of the text
tokenizer <- transformers$AutoTokenizer$from_pretrained("bert-base-uncased")
# size of the vocabulary
tokenizer$vocab_size

# You can look at the list of possible tokenizers and Transformers that you can call here: 
# https://huggingface.co/models
# Another popular tokenizer and Transformer is for example "roberta-base". RoBERTa is basically an optimized, 
# improved BERT. RoBERTa uses dynamic masking vs. the static masking of BERT. 
# What do you mean by that? When pretraining BERT, each input sentence is masked once and saved. 
# Every time that sentence is used during training, the same tokens are masked in the same positions. 
# That implies that model always sees the same mask pattern for that example.
# Instead of pre-saving masked sentences, RoBERTa applies random masking on the fly during each 
# training epoch. So each time a sentence appears during training, a different set of tokens gets
# masked. Dynamic masking exposes the model to more varied learning scenarios, making it better 
# at generalizing and less prone to overfitting on specific mask positions.
# You can call "roberta-base" by writing:
# tokenizerR <- transformers$AutoTokenizer$from_pretrained("roberta-base")
# Among other possible Transformer algorithms that you can decide to explore:
# Electra (via "google/electra-base-discriminator");
# BERTweet (the first public large-scale language model pre-trained for English Tweets);
# "bert-base-multilingual-cased" covers 104 top languages at Wikipedia;
# "xlm-roberta-large" covers 100 language

# let's apply the BERT tokenizer to a text.
# The key rule: BERT always inserts [CLS] per sequence, not per sentence.
# What counts as a sequence depends on how you call the tokenizer.

# Case 1 — One single text
text <- c( "Hello, how are you? I'm fine, thank you.")

# here BERT treats the two sentences as one sequence
# One input → one BERT sequence
# BERT format: [CLS] sequence [SEP]
  
encoded1 <- tokenizer(
  text
)

# Special tokens:
# [CLS] token at start (id 101)
# [SEP] token at end (id 102)
# Take also a look at the attention_mask equals to 1 for all the tokens in the sequence (i.e., each token
# is giving attention to each other token)

encoded1$input_ids
# let's extract such inputs
input_ids <- encoded1$input_ids
decoded <- tokenizer$decode(as.integer(input_ids))
decoded
# print w/o special tokens
decodedBis <- tokenizer$decode(as.integer(input_ids), skip_special_tokens=TRUE)
print(decodedBis)

# Case 2 — Sentence pair 
# Here we consider two sentences separated one from the other but still in one sequence.
# This is explicit sentence-pair mode
text2 <- c( "Hello, how are you?", "I'm fine, thank you.")

encoded2 <- tokenizer(
  text2[1],
  text2[2]
)

# internally tokenizer also creates:
# token_type_ids = 0 → sentence A
# token_type_ids = 1 → sentence B
# This is by design for example for NLI (as we will see)

# here one 101 and two 102 tokens
print(encoded2)
encoded2$input_ids
input_ids <-encoded2$input_ids

decoded <- tokenizer$decode(as.integer(input_ids))
decoded

# Case 3 — Batch of strings
# Now we are doing batch encoding. This means:
# “These are two independent inputs, not a sentence pair” i.e., two separated sequences.
# Why adding "padding=TRUE" and "truncation=TRUE"? More on this in one minute

encoded3 <- tokenizer(
  text2,
  truncation = TRUE,
  padding = TRUE
)

# Let's extract the tokens: the 0 token stands for [PAD]
input_ids <- encoded3$input_ids
input_ids

# Decode each Batch
decoded2 <- lapply(input_ids, function(x) {
  tokenizer$decode(as.integer(x))
})
decoded2

# from where the [PAD] tokens come from?
# This is related to "padding=TRUE".
# When you tokenize a text for a transformer model like BERT, the model expects:
# 1) all input sequences in a batch to have the same length
# 2) and that the length should not exceed the model’s maximum input length (512 tokens for BERT)
# But in real life, your text sentences have all different lengths.
# That’s where padding and truncation come in:
# padding=TRUE ensures that all sequences in the batch are padded to the length of the longest 
# sequence in that batch.  
# If you set padding="max_length", it will pad to the model’s maximum input length (usually 512) — 
# padding=TRUE auto-pads to the longest sentence in the batch.
# Summing up: padding ensures BERT gets inputs of equal shape.
# In our example, you have two [PAD] tokens (i.e., two 0s) in the first batch to have the same length
# of the second batch!

# in the attention_mask you see also two final 0s for the first sequence. Why? 
print(encoded3)
# cause you have two [PAD] that are not tokens to give attention to!
# That is, for the [PAD] tokens, the attention masks is 0, meaning that those tokens are not
# affected (and their presence does not affect) the other tokens of the sentences (as it should be!
# We have just added such [PAD] tokens for convenience)

# We saw the [PAD] token in the previous example, but let's show it even more explicitly by defining
# padding="max_length"
encoded3bis <- tokenizer(
  text2,
  truncation = TRUE,
  padding="max_length"
)
print(encoded3bis)

input_ids <- encoded3bis$input_ids
input_ids

# Decode each Batch
decoded2 <- lapply(input_ids, function(x) {
  tokenizer$decode(as.integer(x))
})

decoded2

# truncation=TRUE on the other side ensures that if any sequence exceeds the model’s maximum allowed
# length (512), it will be truncated to fit. That is: the tokenizer will cut it to 512 tokens 
# and discard the rest

long_text <- paste(rep("word", 600), collapse=" ")
long_text <- paste(long_text, "pippo")
long_text
input <- tokenizer(long_text, padding="max_length" , truncation = TRUE)
print(input)
input_ids <- input$input_ids
input_ids
decoded <- tokenizer$decode(as.integer(input_ids))
decoded

# if you want to keep the feature "pippo"
inputs2 <- tokenizer(
  long_text,
  padding = "max_length", 
  truncation = TRUE,
  stride=256L, 
  return_overflowing_tokens=TRUE
)

# When tokenizing a long text that gets split into multiple chunks, stride controls how much overlap
# there is between consecutive chunks.
# Let’s say: our model has a max_length of 512 tokens, our long text is 1200 tokens and we set 
# stride=256. What happens:
# First chunk: tokens 1–512
# Second chunk: tokens 256–768 (i.e., 512-256=256)
# Third chunk: tokens 512–1024
# Fourth chunk: tokens 768–1200
# Each chunk overlaps the previous one by 256 tokens.

# With 601 tokens (as in our case):
# max_length = 512
# stride = 256
# How many chunks?
# Chunk 1: tokens 1–512 
# Chunk 2: tokens 601-512 = 89 remaining, but we stride forward by 256 → next chunk starts at 256
# Chunk 2: tokens 256–601 (601–256 = 345 tokens — since it's under 512, it takes all the remaining)
# But wait — 601-256 = 345 tokens is < 512, so final chunk just has 345 tokens. The remaining are Pad

input_ids <- inputs2$input_ids
input_ids

decoded2 <- lapply(input_ids, function(x) {
  tokenizer$decode(as.integer(x))
})

decoded2 # there is the token "pippo" in the second batch

# Note that you can also decide to use Long-sequence Transformers (rather that applying the previous
# strategy) such as: Longformer (up to 4096 tokens), BigBird (up to 8192+ tokens) or DeBERTa-v3-large

# for a multi-lingual BERT let's call the "bert-base-multilingual-uncased" tokenizer
tokenizerJP <- transformers$AutoTokenizer$from_pretrained("bert-base-multilingual-uncased")
tokenizerJP$vocab_size
text <- c("名前は何ですか")

encodedJP <- tokenizerJP(
  text,
  truncation=TRUE,
  padding=TRUE 
)

print(encodedJP)
input_ids <- encodedJP$input_ids
input_ids

decoded <- tokenizerJP$decode(as.integer(input_ids))
decoded

#############################
#############################
# Step 2: calling a Model
#############################
#############################

# Load BERT model (if you want for example Roberta just write: "roberta-base" instead of
# "bert-base-uncased")
model <- transformers$AutoModel$from_pretrained("bert-base-uncased")

# BERT-base NN architecture comprises 12 layers where each layer captures numeric values 
# for each contextualized embedding dimension, which for BERT is 768 dimensions. 
# = i.e., 768 dimensions per each layer. 
print(model)

# As an alternative we can also use the $AutoModelForSequenceClassification function (assuming
# our task is a classification one)
# Let's select the model
model_name <- "bert-base-uncased"  # or "roberta-base", etc.
# Let's specify the # of labels in our classification exercise. Why adding num_labels?
# When using the $AutoModelForSequenceClassification function, you need to tell to the model 
# how many output logits (classification categories) you’ll need at the head.
# Binary classification → num_labels = 2
# Multi-class classification (e.g., 5 classes) → num_labels = 5
num_labels <- 2L  # for binary classification
model2 <- transformers$AutoModelForSequenceClassification$from_pretrained(
  model_name, num_labels = as.integer(num_labels)
)
print(model2)

# Set model to evaluation mode (no training behaviors, i.e., no fine-tuning). 
# model$eval() moreover ensures deterministic behavior
model$eval()

texts <- c("This is great.", "Terrible movie.")

# Let's tokenize the texts. By specifying return_tensors = "pt" the command produces as an output a tensor. 
# A tensor is the fundamental data structure used in deep learning frameworks in Python. 
# It is a generalization of scalars, vectors, and matrices to any number of dimensions
# For simplicity here let's also define padding =  "max_length".

inputs <- tokenizer(
  texts,
  padding =  "max_length",
  truncation = TRUE,
  return_tensors = "pt"
)

print(inputs)

# Convert the tensor into an R list
inputs_list <- list(
  input_ids = inputs$input_ids,
  attention_mask = inputs$attention_mask
)

class(inputs_list)
names(inputs_list)

# let's pass the tokenized object to our model to extract their dynamic embeddings
outputs <- do.call(model, inputs_list)
# we have a tensor with 2 obs, each with 512 tokens (remember padding =  "max_length") and 
# with a vector of lenght=768 for each token (the total number of dimensions in BERT base)
print(outputs$last_hidden_state$shape)

# Get number of sentences in batch (2 in our case)
batch_size <- length(texts)
batch_size

# let's extract in a list the tokens and their corresponding embeddings of the two sentences
# Get input_ids
input_ids <- inputs$input_ids
input_ids
# Get batch size and tokenizer method
convert_ids_to_tokens <- tokenizer$convert_ids_to_tokens
# Initialize list to hold everything
token_embeddings_list <- list()

# Loop over each sequence to extratc a list with the tokens of the two sentences and their corresponding
# dynamic embeddings
for (i in 0:(batch_size-1)) {
  # Get token IDs for sentence i
  token_ids <- input_ids[i, ]$tolist()
  # Convert IDs to tokens
  tokens <- convert_ids_to_tokens(token_ids)
  # Get embeddings for sentence i
  sentence_emb <- outputs$last_hidden_state[i, , ]
  sentence_emb_matrix <- reticulate::py_to_r(sentence_emb$detach()$numpy())
  # Save both tokens and embeddings
  token_embeddings_list[[i+1]] <- list(
    tokens = tokens,
    embeddings = sentence_emb_matrix
  )
}

# Check result
str(token_embeddings_list)

token_embeddings_list[[1]]$tokens
token_embeddings_list[[2]]$tokens

# The model still produces a 768-dim vector for [PAD] — but it’s essentially not influenced by 
# other tokens’ contexts due to the attention mask as discussed above.

# So what we can do with this dynamic embeddings? 
# We can take the average of the contexualized embeddings of the tokens in each document 
# to compute document vectors and then using them as an input of a ML model
# (exactly as we did with the static WEs).

# For doing it, we can also decide to remove all special tokens before computing avg. for each text

special_tokens <- c("[PAD]", "[CLS]", "[SEP]")
clean_token_embeddings_list <- list()

for (i in seq_along(token_embeddings_list)) {
  tokens <- token_embeddings_list[[i]]$tokens
  embeddings <- token_embeddings_list[[i]]$embeddings
  # Get indices of non-special tokens
  keep_idx <- which(!tokens %in% special_tokens)
  # Keep only non-special tokens and embeddings
  tokens_clean <- tokens[keep_idx]
  embeddings_clean <- embeddings[keep_idx, , drop=FALSE]
  # Save cleaned version to new list
  clean_token_embeddings_list[[i]] <- list(
    tokens = tokens_clean,
    embeddings = embeddings_clean
  )
}

# Check structure
str(clean_token_embeddings_list)
clean_token_embeddings_list[[1]]$tokens
clean_token_embeddings_list[[2]]$tokens

# We can then now compute the avg. for each text

# Initialize list to hold average embeddings
average_embeddings <- list()
for (i in seq_along(clean_token_embeddings_list)) {
  embeddings <- clean_token_embeddings_list[[i]]$embeddings
  # Compute mean of each column (dimension 1 to 768)
  avg_embedding <- colMeans(embeddings)
  # Save to list
  average_embeddings[[i]] <- avg_embedding
}

# Check structure
str(average_embeddings)
# let's convert the list into a matrix
average_embeddings_matrix <- do.call(rbind, average_embeddings)
str(average_embeddings_matrix)

# Alternatively, we can just keep the embedding of the [CLS] token from the last layer (12) 
# as a summary of the entire document.

# Remember: R is 1-based indexed, while Python is 0-based indexed [the first element in
# a vector in Python starts at 0 not 1!].
# In this sense, outputs$last_hidden_state below is a PyTorch tensor coming from Python (via reticulate) 
# and Python uses 0-based indexing for arrays and tensors as already said.
# So when you loop through outputs$last_hidden_state in R using reticulate, you need to use 
# Python-style 0-based indices to access the batch elements correctly.
# In the model output: outputs$last_hidden_state has shape (batch_size, sequence_length, hidden_size)
# and the [CLS] token is always at position 0 [=1; in Python it starts at 0!] along the sequence length
# of the vector.
# cls_vectors[[i+1]]: → In R, lists are 1-based, so it stores the vector into position i+1 of 
# cls_vectors.

cls_vectors <- list()
for (i in 0:(batch_size-1)) {
  cls_vectors[[i+1]] <- reticulate::py_to_r(outputs$last_hidden_state[i, 0L, ]$detach()$numpy())
}

str(cls_vectors)

# At the moment we are computing the documents' avg or extracting the [CLS] from the last layer 
# (the 12th). This is fine for the [CLS] vector. It is not necessarily so (as discussed) for 
# the former option (i.e., computing the avg. position of a text in the WE space).
# So, let's suppose we want to compute the documents' avg from layer 11. How to do that?

# Load config first and modify it so that BERT will return a list of hidden states, 
# one for each layer plus the embedding output (before any Transformer layers), i.e., 
# layer=0 in Python
config <- transformers$AutoConfig$from_pretrained("bert-base-uncased")
config$output_hidden_states
config$output_hidden_states <- TRUE
config$output_hidden_states

# Then load the model with this config
model2 <- transformers$AutoModel$from_pretrained(
  "bert-base-uncased",
  config = config
)

# let's add to the model the inputs_list including input ids and their attention mask
outputs2 <- do.call(model2, inputs_list)
str(outputs2)
layer_11_hidden_states <- outputs2$hidden_states[[12]] 

# Why [[12]] in R means “layer 11” in Python? Remember: R indexing is 1-based; Python indexing is 0-based
# So:
#  Conceptual layer	 Python index	 R index
#  Embeddings	output      0	        1
#  Transformer Layer 1	  1	        2
#  Layer 1	              2	        3
#  …	…	…
#  Layer 11	             11	        12
#  Layer 12	             12	        13

# Then everything is going to be the same as above: from batch_size to below.

############################################
# To do everything with one function
############################################

# let's call the source of all the functions we will employ today
source("function ENCODER 2026 LUMACSS.R")
# this function can run the encoder also with a "gpu" as long as you have a "gpu" available
# on your laptop. The same on Google Colab.
# To use a "cpu", just specify below device="cpu". To use a "gpu" specify device="cuda".
# Why writing "cuda" if you want to use a "gpu"?
# CUDA (Compute Unified Device Architecture) is the interface that allows PyTorch to use your NVIDIA GPU.
# If you want PyTorch to run on the GPU, you must therefore tell it to use "cuda".
head(get_text_embeddings_stride)

texts <- c("This is great.", "Terrible movie.")

# Let's compute the average position of a text from layer 12. 
# Note that by default remove_special_tokens = TRUE
transformers$logging$set_verbosity_error() # to avoid to report harmless warnings 
avg_embeddings <- get_text_embeddings_stride("bert-base-uncased", 12, texts, "avg", device="cpu")
str(avg_embeddings)

# CLS token embedding from layer 12
cls_embeddings <- get_text_embeddings_stride("bert-base-uncased", 12, texts, "cls", device="cpu")
str(cls_embeddings)

# getting just the tokens (w/o special tokens)
tokens_cleaned <- get_text_embeddings_stride("bert-base-uncased", 12, texts, "tokens", device="cpu")
str(tokens_cleaned)

# getting the tokens (w/o special tokens) as well as their dynamic embeddings
tokens_emb <- get_text_embeddings_stride("bert-base-uncased", 12, texts, "token_embeddings", device="cpu")
str(tokens_emb)

# the function also work for long texts (i.e., with more than 512 tokens) 
texts_long <- c("This is great.", paste(rep("This is a long sentence.", 200), collapse = " "))
texts_long

# But what happens if you have a token that appears in two chunks cause the second chunk, for example,
# overlaps the previous one by 128 tokens? 
# In this case we take the avg. between the two embeddings
# This way, overlapping tokens’ embeddings at the same absolute text position are averaged over 
# the number of times they appeared in overlapping chunks.

# Average of non-special tokens from layer 12 
avg_embeddings_long <- get_text_embeddings_stride(model="bert-base-uncased", stride = 128L,
                                             layer=12, texts_long, "avg", device="cpu")
str(avg_embeddings_long)

# CLS token embedding from layer 12
cls_embeddings_long <- get_text_embeddings_stride("bert-base-uncased", 12,  stride = 128L, 
                                                  texts_long, "cls", device="cpu")
str(cls_embeddings_long)

# Get the embeddings of the tokens without special tokens
tokens_cleaned_long <- get_text_embeddings_stride("bert-base-uncased", 12, stride = 128L,
                                             texts_long, "token_embeddings", device="cpu")
str(tokens_cleaned_long)

# let's extract the list of tokens for the two sentences
clean_token_embeddings <- function(embedding_list) {
  lapply(embedding_list, function(item) {
    item$tokens <- unlist(item$tokens)
    return(item)
  })
}

tokens_cleaned_long2 <- clean_token_embeddings(tokens_cleaned_long)
str(tokens_cleaned_long2)

# note that if you specify "return_overflowing_tokens = FALSE" in get_text_embeddings_stride, 
# you would apply a truncation at 512, i.e., you only embed the first 512 model tokens, 
# including special tokens. For BERT single-sentence inputs, this corresponds to 510 text tokens
# after removing [CLS] and [SEP] if you keep the argument remove_special_tokens = TRUE

# example of dynamic embeddings with the token "bank"
texts <- c("The man was accused of robbing a bank.", 
           "The man went fishing by the bank of the river. 
               He went there with his kid. He was happy",
           "I brought my savings to the bank next home")

# here, let's extract the embeddings from the 11th layer
tokens_cleaned <- get_text_embeddings_stride("bert-base-uncased", 11, texts, "token_embeddings", device="cpu")
str(tokens_cleaned)
tokens_cleaned[[1]]$tokens
tokens_cleaned[[2]]$tokens
tokens_cleaned[[3]]$tokens

# Let's compute cosine-similarity  for the token "bank".
# let's first select only the vectors for the token "bank"
token <- c("bank")
token_bank <- list()

for (i in seq_along(tokens_cleaned)) {
  tokens <- tokens_cleaned[[i]]$tokens
  embeddings <- tokens_cleaned[[i]]$embeddings
  # Get indices of non-special tokens
  keep_idx <- which(tokens %in% token)
  # Keep only bank and embeddings
  tokens_clean <- tokens[keep_idx]
  embeddings_clean <- embeddings[keep_idx, , drop=FALSE]
  # Save cleaned version to new list
  token_bank[[i]] <- list(
    tokens = tokens_clean,
    embeddings = embeddings_clean
  )
}

str(token_bank)

embeddings_list <- lapply(token_bank, function(item) as.vector(item$embeddings))
str(embeddings_list)
class(embeddings_list)

# Step 2: let's now compute pairwise cosine similarities
n <- length(embeddings_list)
sim_matrix <- matrix(0, n, n)

for (i in 1:n) {
  for (j in 1:n) {
    sim_matrix[i, j] <- cosine(embeddings_list[[i]], embeddings_list[[j]])
  }
}

# Step 3: Label the matrix 
rownames(sim_matrix) <- colnames(sim_matrix) <- paste0("Sentence_", 1:n)

# Step 4: Let's print the results of the cosine-similarity matrix - we can see that the
# embeddings for the token "bank" in the first and third sentence presents a higher cosine
# similarity compared to the token "bank" out of the second sentence
print(sim_matrix)

# Let' see a graphical representation with the t-SNE algorithm 
clean_token_embeddings <- function(embedding_list) {
  lapply(embedding_list, function(item) {
    item$tokens <- unlist(item$tokens)
    return(item)
  })
}

tokens_cleaned2 <- clean_token_embeddings(tokens_cleaned)
str(tokens_cleaned2)

# Step 1: Convert list to data frame
embedding_df <- do.call(rbind, lapply(tokens_cleaned2, function(item) {
  df <- as.data.frame(item$embeddings)
  colnames(df) <- paste0("Dim", 1:ncol(df))
  data.frame(tokens = item$tokens, df)
}))

# Step 2: View structure
str(embedding_df)

labels <- as.data.frame(embedding_df[, "tokens"]) # let's extract the tokens' labels
str(labels)

set.seed(123)
system.time(tsne <-  Rtsne(embedding_df, perplexity=10))
str(tsne)
tsne_plot <- tsne$Y
tsne_plot  <- as.data.frame(tsne_plot)
str(tsne_plot)

# as you can see, two "bank" tokens are next to "savings", the other "bank" is next to "river"
ggplot(tsne_plot, aes(x = V1, y = V2)) +
  geom_point(colour = 'blue') +
  geom_text(label=labels[,1], hjust=0, vjust=0 )

#############################
#############################
# Let's use a real dataset
#############################
#############################

x_tot <- read.csv("Input Data/Day 4/disaster_tot.csv")
str(x_tot)

# let's extract the texts
texts <- x_tot$text 
length(texts)

# let's check if you have any empty texts: no! 
any(texts == "")
sum(texts == "")

# if you have any, we can remove them as below:
# texts <- texts[texts != ""]

##############
# extracting the dynamic embeddings 
#############

# Average of non-special tokens from layer 11
# you need around 2 minutes
# system.time(avg_embeddings <- get_text_embeddings_stride("bert-base-uncased", 11, texts, "avg", 
#                                                              device="cpu"))
# saveRDS(avg_embeddings, "Input Data/Day 4/Lab4m2026/avg_embeddings.RDS")
avg_embeddings <- readRDS("Input Data/Day 4/Lab4m2026/avg_embeddings.RDS")
str(avg_embeddings)
# we have a matrix
class(avg_embeddings)

# CLS token embedding from layer 12
# you need around 2 minutes
# system.time(cls_embeddings <- get_text_embeddings_stride("bert-base-uncased", 12, texts, "cls", device="cpu"))
# saveRDS(cls_embeddings, "Input Data/Day 4/Lab4m2026/cls_embeddings.RDS")
cls_embeddings <- readRDS("Input Data/Day 4/Lab4m2026/cls_embeddings.RDS")
str(cls_embeddings)
# we have a matrix
class(cls_embeddings)

# let's run some cross-validation with the just extracted data using just a RF to make things faster
source("Input Data/Day 2/Function 2026 LUMACSS.R")

y <- factor(x_tot $choose_one,  levels=c("NoDisaster", "Disaster"), 
                           labels=c("NoDisaster", "SocialDisaster"))
str(y)

# let's use a RF model
Ranger_avg <- Function_RF(input=avg_embeddings, k=5, DV=y, ML=ranger)
Ranger_cls <- Function_RF(input=cls_embeddings, k=5, DV=y, ML=ranger)
colMeans(Ranger_cls[ , c(1, 2, 3, 4)]) # balanced accuracy: .773
colMeans(Ranger_avg[ , c(1, 2, 3, 4)]) # balanced accuracy: .777

#############################
#############################
# Exploring global intepretation
#############################
#############################

### WITH 2-categories example

x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)
y <- factor(x$choose_one,  levels=c("0", "1"), labels=c("NoDisaster", "SocialDisaster"))
class(y)
table(y)

# let's extract the avg. embeddings from layer 11 and then let's compute a ML algorithm
# system.time(avg <- get_text_embeddings_stride(x$text,  model_name = "bert-base-uncased",
# embedding_type = "avg", layer = 11, device="cpu"))
# saveRDS(avg, "Input Data/Day 4/Lab4m2026/avg_train.RDS")
avg <- readRDS("Input Data/Day 4/Lab4m2026/avg_train.RDS")
str(avg)

# let's fit a RF model. But before doing that, remember to add fictional colnames to the
# extracted matrix! This is done automatically using the Function_RF function, but not here!
colnames(avg)
colnames(avg) <- paste0("x",1:ncol(avg))
colnames(avg)

set.seed(123)
system.time(RF <- ranger(y= y,  x=avg)) 

# How "entering" the black box of the ML in this case with dynamic WE?
# Let's follow Park, Ju Yeon, and Jacob M. Montgomery. "Toward a framework for creating trustworthy
# measures with supervised machine learning for text." Political Science Research and Methods (2025):
# 1-17 (in particular pp.11-12)
xtest <- read.csv("Input Data/Day 1/test_disaster.csv", stringsAsFactors=FALSE)
str(xtest)

# let's extract the dynamic WE for the test set 
# system.time(avg_test <- get_text_embeddings_stride(xtest$text,  model_name = "bert-base-uncased",
#                                               embedding_type = "avg", layer = 11, device="cpu"))
# saveRDS(avg_test, "Input Data/Day 4/Lab4m2026/avg_Test.RDS")
avg_test <- readRDS("Input Data/Day 4/Lab4m2026/avg_Test.RDS")

# once again, let's add fictional colnames to the extracted matrix!
colnames(avg_test)
colnames(avg_test) <- paste0("x",1:ncol(avg_test))
colnames(avg_test)

# no need to do any matching between training and test-set here cause they are include the same exact
# features (i.e., the BERT dimensions from the dynamic WE space)
setequal(colnames(avg), colnames(avg_test)) 

set.seed(123)
prediction <- predict(RF, avg_test)
str(prediction)
table(prediction$predictions)

# let's add back the predicted labels to our test-set dataframe
xtest$label <- prediction$predictions
str(xtest)

# let's create a dfm out of our test-set
myCorpus2 <- corpus(xtest)
tok2  <- tokens(myCorpus2, remove_punct = TRUE, remove_numbers=TRUE, 
                remove_symbols = TRUE, 
                split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, c("0*"))
tok2<- tokens_wordstem (tok2, language =("english"))
tok2 <- tokens_remove(tok2 , stopwords("english"))

myDfm2 <- dfm(tok2)
myDfm2 <- dfm_trim(myDfm2 , min_docfreq = 2, verbose=TRUE)
myDfm2  <- dfm_remove(myDfm2 , min_nchar = 2)

str(myDfm2@docvars)
# Let's group the documents according to the label
myDfm  <- dfm_group(myDfm2, groups = label)
# our target label is "SocialDisaster"
result_keyness <- textstat_keyness(myDfm , target = "SocialDisaster")
str(result_keyness)
# Let's consider as salient only terms with a significant overrepresentation at a p<0.001 level 
result_keyness2 <- result_keyness[ which(result_keyness$p<=0.001), ]
textplot_keyness(result_keyness2,  n = 30L, show_reference = FALSE)

######################################
######################################
### WITH 3 or more categories to classify
######################################
######################################

airlines <- read.csv("Input Data/Day 1/train_airlines.csv")

# your dv
class(airlines$airline_sentiment)
table(airlines$airline_sentiment)

# Let'c convert the "choose_one" variable into a factor variable
y <- factor(airlines$airline_sentiment)
class(y)
table(y)

# let's extract the dynamic WE for the training set 
# system.time(cls3 <- get_text_embeddings_stride(airlines$text,  model_name = "bert-base-uncased",
#                     embedding_type = "cls", layer = 12, device="cpu"))
# saveRDS(cls3, "Input Data/Day 4/Lab4m2026/cls_train3.RDS")
cls3 <- readRDS("Input Data/Day 4/Lab4m2026/cls_train3.RDS")

Ranger_avg <- Function_RF(input=cls3, k=5, DV=y, ML=ranger)
colMeans(Ranger_avg[ , c(1, 2, 3, 4)]) 

colnames(cls3) <- paste0("x",1:ncol(cls3))
colnames(cls3)

set.seed(123)
system.time(RF3 <- ranger(y= y, x=cls3))

airlines_test <- read.csv("Input Data/Day 1/test_airlines.csv")
str(airlines_test)

# let's extract the dynamic WE for the test set 
# system.time(cls3_test <- get_text_embeddings_stride(airlines_test$text,  model_name = "bert-base-uncased",
#                                                embedding_type = "cls", layer = 12, device="cpu"))
# saveRDS(cls3_test, "Input Data/Day 4/Lab4m2026/cls3_test.RDS")
cls3_test <- readRDS("Input Data/Day 4/Lab4m2026/cls3_test.RDS")
dim(cls3_test)

colnames(cls3_test) <- paste0("x",1:ncol(cls3_test))
set.seed(123)
prediction <- predict(RF3, cls3_test)
str(prediction)

airlines_test$label <- prediction$predictions
str(airlines_test)
myCorpus2 <- corpus(airlines_test)
tok2  <- tokens(myCorpus2, remove_punct = TRUE, remove_numbers=TRUE, 
                remove_symbols = TRUE, 
                split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, c("0*"))
tok2<- tokens_wordstem (tok2, language =("english"))
tok2 <- tokens_remove(tok2 , stopwords("english"))

myDfm2 <- dfm(tok2)
myDfm2 <- dfm_trim(myDfm2 , min_docfreq = 2, verbose=TRUE)
myDfm2  <- dfm_remove(myDfm2 , min_nchar = 2)

str(myDfm2@ docvars)
# Calculate keyness for each label against the others
myDfm  <- dfm_group(myDfm2, groups = label)
myDfm

# Create separate keyness plots
tstat_positive <- textstat_keyness(myDfm, target = "positive")
tstat_negative <- textstat_keyness(myDfm, target = "negative")
tstat_neutral <- textstat_keyness(myDfm, target = "neutral")

# Let's consider as salient only terms with a significant overrepresentation at a p<0.001 level 
tstat_positive <- tstat_positive[ which(tstat_positive$p<=0.001), ]
tstat_negative <- tstat_negative[ which(tstat_negative$p<=0.001), ]
tstat_neutral <- tstat_neutral[ which(tstat_neutral$p<=0.001), ]

# Create a multi-panel plot
p1 <- textplot_keyness(tstat_positive, show_reference = FALSE) + ggtitle("Positive")
p2 <- textplot_keyness(tstat_negative, show_reference = FALSE) + ggtitle("Negative") 
p3 <- textplot_keyness(tstat_neutral, show_reference = FALSE) + ggtitle("Neutral")

plot_grid(p1, p2, p3, nrow = 1)

# As an alternative, let's create a combined data frame for comparison via ggplot

# Get top terms for each label
top_positive <- head(tstat_positive, 15)
top_negative <- head(tstat_negative, 15)
top_neutral <- head(tstat_neutral, 15)

comparison_df <- bind_rows(
  top_positive <- mutate(top_positive, label = "positive"),
  top_negative <- mutate(top_negative, label = "negative"),
  top_neutral <- mutate(top_neutral, label = "neutral")
)

ggplot(comparison_df, aes(x = chi2, y = reorder_within(feature, chi2, label), fill = label)) +
  geom_col() +
  facet_wrap(~label, scales = "free_y") +
  scale_y_reordered() +
  labs(x = "Keyness Score", y = "Terms") +
  theme_minimal()

