rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(reticulate)
use_condaenv("python_lib", required = TRUE)
options(warn = 0) # I delete the warning

install.packages(RcppParallel, repos="http://cran.us.r-project.org")

library(quanteda.textstats)
library(quanteda)
library(text2vec)
library(quanteda.textplots)
library(Rtsne)
library(ggplot2)
library(lsa)
library(dplyr)
library(ranger)
library(caret)
library(cvTools)
library(readr)
library(tidyverse)
library(glue)
library(naivebayes)
library(yardstick)
library(keras3)

# Let's focus on a dataset containing 1,500 MOVIE reviews
tot <- read.csv("Input Data/Day 3/movie_tot.csv")
# if you have problems to open this .csv file, plz use the below .rds file
# tot <- readRDS("Input Data/Day 3/movie_tot.rds")
str(tot)

# Note: here we don't compute the COM on the original text (i.e., tot$text), but on the tokens we get
# after pre-processing the texts and creating the corresponding DfM. Why? Cause we can get rid of those 
# words appearing just few times in the texts (to avoid to build a very sparse COM). On top of that,
# this allows us to embed a given document into an embedding space more easily. However,
# you can also decide to compute a GloVe model directly on the original text (see below)

# let's replace the apostrophes with an empty space (Quanteda struggles with that)
tot$text <- gsub("'"," ",tot$text)

# Let's create our corpus
myCorpus <- corpus(tot)
ndoc(myCorpus)

# why identifying as our unit of analysis the "sentences"? Cause the window size in GlOVe does not usually 
# cross sentence boundaries, i.e., the window is applied only within each sentence! In other words,
# unless you separate sentences in a document, both GloVe and Word2Vec will compute 
# windows across sentences, i.e. with a window size of 2, in the sentence "I like sushi. What about you?"
# the word "What" will have in its window also "." and "sushi". This could still make sense if you have short
# texts or if sentences in a document refers to the same topic. However, I would always avoid it.

myCorpus <- corpus_reshape(myCorpus, to = "sentences")
ndoc(myCorpus)

# Let's extract the tokens from each of the text included in the corpus.
# We also do some minimal preprocessing. 
# REMEMBER: everytime you employ a pre-trained WE, it is always advisable check if it has 
# been computed using the stemming option or otherwise. Typically, it does not use this 
# option! Therefore: be careful with employing the stemming option on your features! 
# Here we avoid it cause later on we will also used a pre-trained WE!

tok2 <- tokens(myCorpus , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, 
               remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
Dfm_sentences <- dfm(tok2 )
# we find the "s" extracted above via  gsub("'"," ",tot$text) out of the genitive saxon 
topfeatures(Dfm_sentences )
# Let's remove the words with just 1 character
Dfm_sentences <- dfm_remove(Dfm_sentences, min_nchar=2)
topfeatures(Dfm_sentences )

# Moreover, we follow standard practice which is to include all words with a minimum count 
# above a given threshold, between 5-10 (we choose 5)
Dfm_sentences <- dfm_trim(Dfm_sentences,  min_termfreq = 5, verbose=TRUE)

#############################################
#############################################
# Applying the GloVe algorithm via Quanteda 
#############################################
#############################################

# Let's first extract the vocabulary from our Dfm
Dfm_vocab <- featnames(Dfm_sentences )
head(Dfm_vocab)

# Then let's select the tokens that are present in our previously defined corpus
mov_tokens <- tokens(myCorpus) 

# Finally, let's match the two vocabularies (i.e., let's keep only those features that are 
# both present in the tokenized corpus as well as in the Dfm - after that we pre-processed
# and trimmed the texts!)

# Note the following: the command "padding=TRUE" below leaves an empty string where 
# the removed tokens previously existed. This is useful if a window of adjacency needs 
# to be computed. This prevents that non-adjacent words (in the original text) becomes 
# adjacent (after pre-processing). 
# W/o "padding=TRUE" the word "eat" and "sushi" in the following sentence: 
# "I like to eat 2 sushi sets" and windows=1 would be included in the same window 
# (after you remove the numbers as we did from our corpus). With "padding=TRUE" this 
# wouldn't happen. Summing up: removal of tokens changes the lengths of documents, 
# but they remain the same if you set "padding = TRUE"

mov_tokens2 <- tokens_select(mov_tokens, Dfm_vocab, padding = TRUE)
head(mov_tokens2 )

# Create a term co-occurance matrix (default window is 5; you can change it by using 
# the "window" argument in the fcm function)
fcmat_news <- fcm(mov_tokens2, context = "window")
fcmat_news
dim(fcmat_news)

# Alternatively we can weight out FCM. Why using the weights option? 
# In this case, we want to weight more tokens that are closer compared to those more
# far away to the target word. This is one way to account for the fact that very distant 
# word pairs are expected to contain less relevant information about the words’ relationship
# to one another

# For example in the sentence: "I like sushi a lot, it is so fresh and delicious!", if you consider
# a window=5 and the target-word "is", then (on the right side of the window), "so" will have a weight
# of 1, "fresh" of 1/2, "and" of 1/3, etc. 
fcmat_news_weights <- fcm(mov_tokens2, context = "window", count = "weighted", weights = 1/(1:5))
fcmat_news_weights

# Note that fcm creates a sparse feature co-occurrence matrix. This makes the matrix 
# more easy to treat (and less computational heavy). This is a good thing when dealing 
# with text analysis!

# Compare the two following examples
testText <- c("Kitty likes milk", "Cat likes milk")
testText 

# A dense co-occurance matrix with window=2 would be as the following one:
dense <- matrix(
  c(0,1,1,0,1,0,2,1,1,2,0,1,0,1,1,0), nrow = 4, ncol = 4, byrow = TRUE)
rownames(dense ) = c("kitty", "likes", "milk", "cat")
colnames(dense ) = c("kitty", "likes", "milk", "cat")
dense 

# However, if we apply the command fcm to these two texts we obtain a sparse feature 
# co-occurrence matrix
testCorpus <- corpus(testText)
tok3 <- tokens(testCorpus)
sparse <- fcm(tok3, context = "window", window=2)
sparse 
dense 

# We can also visualize a semantic network analysis starting from the fcm
set.seed(123)
textplot_network(sparse)

#############################################
#############################################
# Let's estimate WE via Glove
#############################################
#############################################

# First let's define the loss function for GloVe
# Which are the main parameters to look at?
# 1) rank=number of dimensions (100 dimensions is the default)
# 2) x_max=maximum number of co-occurrences to use in the weighting function. 
# Here we select 10 (in the paper that introduced GloVe: x_max=100). 
# This parameter affects the weighting function in the loss function that GloVe minimizes 
# (remember our discussion!)

# Note that if you use a weighting procedure to generate a COM matrix as in the example above,
# a word that ALWAYS appears 5 words away from the target word will have a weight of 1/5. 
# So for not being downweighted by the GloVe function (given x_max=10), this word
# should appear at least 50 times in the COM together with the target word. In other words,
# everytime you apply a weighting scheme to produce a COM, you can also consider to reduce
# the value of x_max

# Then, let's train the model! How does GloVe works?
# 1) It initializes vectors of each pair of words randomly
# 2) Sample a batch of (i,j) word pairs
# 3) Compute the loss J for these pairs
# 4) Update vectors via gradient descent (iterative training). In particular, each iteration (or epoch)
# is as usual a pass over the training data (in this case, the co-occurrence matrix), 
# and the model updates its parameters — the word vectors — to better fit the data
# 5) Stop when vectors stabilize (loss plateaus)

# Note: if you want to exactly reproduce your results, you have to specify:
# n_threads = RcppParallel::setThreadOptions(1) (i.e., no parallelization!).
# You will need more computational time, but at least replication is assured. 
# The differences if we replicate the analysis with a larger number of threads 
# will be however minimal.

# You decide the number of iterations via n_iter (20 below is pretty low. A good value is between 
# 50 and 100; but you would need more time...). You also decide the convergence_tol. 
# convergence_tol defines early stopping strategy. GloVe stops fitting when one 
# of the following two conditions is satisfied: 
# (a) GloVe has used all the specified iterations 
# (b) (cost_previous_iter / cost_current_iter - 1) < convergence_tol

# around 50 seconds
set.seed(123)
glove <- GlobalVectors$new(rank=100, x_max=10)
system.time(glove_main <- glove$fit_transform(fcmat_news, n_iter = 20, 
n_threads = RcppParallel::setThreadOptions(1), convergence_tol = 0.01 )) 
# As discussed, you can use the final loss-function value to eventually compare 
# across different GloVe specifications 

str(glove_main)
# As discussed, GloVe actually learns 2 matrices of word-vectors:
# 1) main (the one we employed above) and 2) context.
# It could be a good idea to average or take a sum of main and context vector. 
# Let's follow the former procedure here

wv_context <- glove$components # the context WE
str(t(wv_context)) # we need to transpose the matrix to directly compare it to the original glove_main matrix
dim(wv_context)
glove_main <- (glove_main + t(wv_context))/2 # let's take the avg. of the two vectors
# alternatively we could take the sum:
# glove_main <- glove_main + t(wv_context)

# Let's create a dataframe out of the Glove results
glove_dataframe <- as.data.frame(glove_main)
nrow(glove_dataframe)
# the same # of words as in our co-occurance matrix of course!
nrow(fcmat_news)

colnames(glove_dataframe )
# let's add to glove_dataframe a specific column called "word" with the list of features
glove_dataframe$word <- row.names(glove_dataframe )
colnames(glove_dataframe )

#############################################
#############################################
# Estimating WE via Glove on the original texts
#############################################
#############################################

# As discussed, you can compute a local WE directly on the original text (without the need of 
# doing any pre-processing on your data - also because pre-processing is less relevant
# when dealing with WE). In this case you could write as below for a GloVe model:

tokenGlove <- tokens(corpus(tot))
fcmat_newsGlove <- fcm(tokenGlove, context = "window")
fcmat_newsGlove
# of course now the dimensionality of your COM is going to be larger than when dealing
# only with the tokens included in your DfM after pre-processing
nrow(fcmat_newsGlove)
nrow(fcmat_news)
# then you can apply GloVE to fcmat_newsGlove as usual

#############################################
#############################################
############# Checking for word-similarity 
#############################################
#############################################

# Let's see what is similar to "girl", "boy" and "movie" in terms of cosine-similarity 
# using the 100-dimensional embedding space we computed above.
# As always, the higher the value of cosine-similarity , the higher is the semantic 
# similarity between two words. 
# NOTE: This function requires a matrix (not a dataframe), i.e., glove_main NOT glove_dataframe 

similarity <- function(data, word){
  word<- data[word, , drop = F]
  cos_sim <- sim2(x = data, y = word, method = "cosine")
  return(head(sort(cos_sim[,1], decreasing = T), 10))
}

similarity(glove_main, "girl")
similarity(glove_main, "boy")
similarity(glove_main, "movie")

# Let's plot the embeddings! But how to do that in a 100-dimensional world?
# The t-SNE algorithm can be used to visualize the embeddings.
# The goal of this algorithm is to take a set of points in a high-dimensional space 
# and find a faithful representation of those points in a lower-dimensional space, 
# typically the 2D plan.
# The algorithm is non-linear and adapts to the underlying data, performing different 
# transformations on different regions. 
# For example, a feature of t-SNE is a tuneable parameter, perplexity, which controls
# how many neighbors each point “pays attention to” when balancing local vs global structure
# of your data in the low dimensional embedding. So when you set: perplexity = 30
# you’re telling t-SNE: “For each point, try to preserve distances to roughly its 30 nearest neighbors.”
# Larger values put more emphasis on global over local structure preservation, and viceversa. 
# Intuition specific to word embeddings:
# Low perplexity → focuses on very tight synonym-like neighborhoods (“king” ↔ “queen” ↔ “prince”)
# Higher perplexity → preserves broader topical structure (royalty vs animals vs verbs)
# Perplexity should be much smaller than N. A common range: 5–50
# Good heuristic: perplexity ≈ N / 10 (in our case, N=number of tokens)
# Because of time-constraints we will only use it with the first 500 words of our corpus
# and focusing just on Glove results

set.seed(123)
system.time(tsne <-  Rtsne(glove_main[1:500,], dims = 2, perplexity = 50))
str(tsne)
tsne_plot <- tsne$Y
tsne_plot  <- as.data.frame(tsne_plot)
str(tsne_plot)
tsne_plot$word  <- row.names(glove_main)[1:500]
str(tsne_plot)

ggplot(tsne_plot) +
  geom_point(aes(x = V1, y = V2), colour = 'blue', size = 0.05) + 
  labs(title = "Word embedding in 2D using  t-SNE")

ggplot(tsne_plot, aes(x = V1, y = V2, label = word)) + geom_text(size = 3)

# This plot may look like a mess, but let's zoom for the word "trailer" for example (a word included
# in the first 500 words of our COM)

tsne_plot2 <- mutate(tsne_plot, highlight = ifelse(word %in% c("trailer"), "yes", "no"))

ggplot(tsne_plot2, aes(x = V1, y = V2, label = word)) +
  geom_text(aes(color = highlight), size = 3, show.legend = FALSE) +
  scale_color_manual(values = c("yes" = "red", "no" = "gray")) +
  theme_minimal()

# A good (and in some cases, better) alternative to t-SNE is the package "umap". 
# Check about it by yourself!

# We can also compute the similarity between any pair of words
# NOTE: This function requires a dataframe (not a matrix), i.e., glove_dataframe NOT glove_main 

similarity_2 <- function(data, word1, word2){
  cosine(
    x=as.numeric(data[data$word==word1,1:100]),
    y=as.numeric(data[data$word==word2,1:100]))
}

similarity_2(glove_dataframe, "father", "mother")
similarity_2(glove_dataframe,"tarantino", "fiction")

# Note the difference between "woman" and "man" here:
similarity_2(glove_dataframe,"woman", "wife")
similarity_2(glove_dataframe,"woman", "mother")

similarity_2(glove_dataframe,"man", "husband")
similarity_2(glove_dataframe,"man", "father")

#############################################
#############################################
# Doing Machine Learning classification with WE (using GLOVE results)
#############################################
#############################################

# At the moment glove_dataframe is a matrix of 12,246 rows (one for each feature) 
# and 101 columns (1 column for word and the other 100 for the 100 dimensions of WE)
dim(glove_dataframe)
# but in the original dataset I had 1,500 documents
dim(tot)

# So how to use the tokens' WEs to compute the position of a text including such tokens in the same 
# WE multidimensional space?
# First, let's the change the unit of analysis of our DfM: from the sentences of each text to their 
# original text
ndoc(Dfm_sentences) # 5250 sentences
# let's group them according to the meta-level variable "X.1"
head(docvars(myCorpus))
Dfm <- dfm_group(Dfm_sentences, groups =docvars(myCorpus, "X.1"))
ndoc(Dfm) # back to 1500 documents!

# For each text in the Dfm let's compute a value equals to the average of 
# its words in each of the 100 dimensions of WE. As a result we can generate a new matrix 
# with 1,500 rows (one for each document) and 100 columns (with the average position of 
# each document in each of the 100 dimensions of WE), wherein each text will be 
# represented by a mean vector. 
# Note that other possibilities are available.
# For example the doc2vec approach combine directly word and document embeddings: 
# see: Le, Q., & Mikolov, T. (2014). Distributed representations of sentences and documents. 
# International Conference on Machine Learning

# Let's define a function to this aim and let's use it.
# You have three arguments in such function: a) the Dfm you are using; b) the WE dataset; c) the name of
# the column in the WE dataset reporting the name of the tokens. In our case: 
head(glove_dataframe$word)
# Note that here either you report the name (in our case: "word") or the position in our WE dataset (in our
# case: 101)
colnames(glove_dataframe)
# d) if you want to compute an average or a weighted average (see below).

doc_embeddings <- function(Dfm, we_dataframe,
                              word_col = 1,
                              method = c("average", "weighted")) {
  
  method <- match.arg(method)
  # resolve word column (name or index)
  if (is.character(word_col)) {
    word_col_name <- word_col
  } else {
    word_col_name <- colnames(we_dataframe)[word_col]
  }
  # embedding columns are all except the word column
  embed_cols <- setdiff(seq_len(ncol(we_dataframe)), 
                        which(colnames(we_dataframe) == word_col_name))
  n_docs <- ndoc(Dfm)
  n_dims <- length(embed_cols)
  # initialize output
  embed <- matrix(NA, nrow = n_docs, ncol = n_dims)
  rownames(embed) <- docnames(Dfm)
  colnames(embed) <- colnames(we_dataframe)[embed_cols]
  # for faster matching, extract word vector once
  embedding_words <- we_dataframe[[word_col_name]]
  for (i in 1:n_docs) {
    if (i %% 100 == 0) message(i, "/", n_docs)
    vec <- as.numeric(Dfm[i, ])
    names(vec) <- featnames(Dfm)
    doc_words <- names(vec)[vec > 0]
    matched_words <- intersect(doc_words, embedding_words)
    if (length(matched_words) == 0) {
      embed[i, ] <- 0
      next
    }
    # extract rows for matched words
    rows <- match(matched_words, embedding_words)
    embed_vec <- as.matrix(we_dataframe[rows, embed_cols])
    if (method == "average") {
      embed[i, ] <- colMeans(embed_vec, na.rm = TRUE)
    } else if (method == "weighted") {
      word_freqs <- vec[matched_words]
      weighted_sum <- t(embed_vec) %*% word_freqs
      denom <- sum(word_freqs)
      embed[i, ] <- as.numeric(weighted_sum / denom)
    }
  }
  return(embed)
}

doc_embeddingsA <- doc_embeddings(Dfm, glove_dataframe, word_col= "word", method = "average")
str(doc_embeddingsA)

# A possible problem with the former approach is that it does not consider if a given feature appears once,
# twice or n-times in a given text. It will always count it as 1. This could be problematic!
# Therefore, to compute for each document a weighted average of the words in each of the 100 dimensions 
# of WE, we can use the below code:

doc_embeddingsW <- doc_embeddings(Dfm, glove_dataframe, word_col= "word", method = "weighted")
str(doc_embeddingsW)

# Our baseline value given a "naive" algorithm is 0.502. We want that the accuracy of our ML 
# algorithm outperforms such value
str(tot$Sentiment)
prop.table(table(tot$Sentiment))

# To compute the CV for ranger let's employ the RF function
source("Function 2026 LUMACSS.R") 
head(Function_RF)

# name of the DV (it must be a factor!)
class(tot$Sentiment)
y <- as.factor(tot$Sentiment)
class(y)

# Let's check if the documents in our original DfM present the same indexing as in the original 
# data frame "tot", so that the DV and the rows of the matrix including the documents' embeddings are
# correctly aligned. For doing that, let's take advantage of the X.1 variable
head(tot$X.1)
head(Dfm@Dimnames$docs)
# Yes, we are fine!
identical(as.character(tot$X.1),Dfm@Dimnames$docs)
# Otherwise sort both of the objects!
# Dfm <- Dfm[order(docvars(Dfm)$X.1), ]
# tot <- tot[order(tot$X.1), ]

# with doc_embeddingsA
Ranger_res <- Function_RF(input=doc_embeddingsA, k=5, DV=y, ML=ranger)
Ranger_res 
# better than the simple learner!
colMeans(Ranger_res[ , c(1, 2, 3, 4)])

# with doc_embeddingsW
Ranger_resW <- Function_RF(input=doc_embeddingsW, k=5, DV=y, ML=ranger)
Ranger_resW 
# very similar results (slightly better with doc_embeddingsW)
colMeans(Ranger_res[ , c(1, 2, 3, 4)])
colMeans(Ranger_resW[ , c(1, 2, 3, 4)])

# You could have employed also a NN or a Naive Bayes model.
# Let's begin with the Naive Bayes model.
# Note that in this latter case you have to fit a naive_bayes rather than a multinomial_naive_bayes given
# that you have continuous IVs (not counts!)! See the extra slides for an explanation
NB_resW <- Function_NB(input=doc_embeddingsW, k=5, DV=y, ML=naive_bayes)

colMeans(Ranger_resW[ , c(1, 2, 3)])
colMeans(NB_resW[ , c(1, 2, 3)])

# Let's also fit a NN with a simple architecture
max_features_toUSE <- ncol(doc_embeddingsW)
max_features_toUSE

head(Function_NN)

NN_resW <- Function_NN(input=doc_embeddingsW, k=5, DV=y, ML=keras_model_sequential, 
                      max_features=max_features_toUSE, 
                      scaling=FALSE, norm=FALSE, units1=16,  
                      epochs=20, batch_size=10)

colMeans(NN_resW[ , c(1:4)])
colMeans(Ranger_resW[ , c(1:4)])

###########################################################################
###########################################################################
# Let's now employ a pre-trained word embeddings dataset computed on a sample of Google news 
# via Word2Vec 
###########################################################################
###########################################################################

# extracting the word embeddings on 100 dimensions 
pre_trained <- read_delim("vector.txt", 
                  skip=1, delim=" ", quote="",
                  col_names=c("word", paste0("V", 1:100)))

# 100 dimensions + 1 column for features
colnames(pre_trained)
nrow(pre_trained) # almost 72K features included; much more than the number of features included in our corpus!
nrow(glove_dataframe)

# let's play a bit with this new pre-trained WE

# let's convert the object from a tibble to a data frame and then to a matrix
class(pre_trained)
pre_trained2 <- as.data.frame(pre_trained)
class(pre_trained2)
# let's transform the column with words into row names
row.names(pre_trained2) <- pre_trained2$word
# and let's remove from the data frame the first columns (i.e., the one including the feature words)
str(pre_trained2)
pre_trained2 <- pre_trained2[-c(1)]
pre_trainedMatrix <- as.matrix(pre_trained2)
str(pre_trainedMatrix)
str(glove_main)

# Let's see an analogy
# Which is the spatially closest word vector to "(vec)king - (vec)male + (vec)female" = ?

analogy <- function(data, word1, word2, word3){
  ex <- data[word1, , drop = FALSE] -
    data[word2, , drop = FALSE] +
    data[word3, , drop = FALSE]
  cos_sim_test <- sim2(x = data, y = ex , method = "cosine")
  x <- head(sort(cos_sim_test[,1], decreasing = T), 5)
  return(x[-c(1)])
}

analogy(pre_trainedMatrix, "king", "male", "female")

# Subtracting the "male" vector from the "king" vector and adding "female", the most 
# similar word to this is "queen"

# Let's see a second analogy
# Which is the spatially closest word vector to "(vec)paris - (vec)france + (vec)uk" = ?

analogy(pre_trainedMatrix, "paris", "france", "uk")

# and if we replace uk with japan?, i.e., "(vec)paris - (vec)france + (vec)japan" = ?

analogy(pre_trainedMatrix, "paris", "france", "japan")

# music analogy!

analogy(pre_trainedMatrix, "madonna", "pop", "rock")

# We can also use element-wise addition of vector elements to ask questions such as 
# ‘japanese + island’ and by looking at the closest tokens to the composite vector

addition <- function(data, word1, word2){
  ex <- data[word1, , drop = FALSE] +
    data[word2, , drop = FALSE] 
  cos_sim_test <- sim2(x = data, y = ex , method = "cosine")
  x <- head(sort(cos_sim_test[,1], decreasing = T), 5)
  return(x[-c(1:2)])
}

addition(pre_trainedMatrix, "japanese", "island")
addition(pre_trainedMatrix, "italian", "sea")

# Working good on these analogies implies that our pre-trained WE is of good quality!

# Let's search for some semantic similarities

similarity(pre_trainedMatrix, "bush")
similarity(pre_trainedMatrix, "google")
similarity(pre_trainedMatrix, "woman")

# We can also compute the similarity between any pair of words. Remember! 
# We need a data-frame here!

pre_trained2 <- as.data.frame(pre_trained)
colnames(pre_trained2)
pre_trained2 <- relocate(pre_trained2, word, .after=V100) # let's move the "word" column to the end
colnames(pre_trained2)

similarity_2(pre_trained2, "woman", "wife")
similarity_2(pre_trained2, "woman", "mother")

similarity_2(pre_trained2, "man", "husband")
similarity_2(pre_trained2, "man", "father")

# Let's NOW match the words included in the pre-trained WE object (pre_trained) 
# with the words included in the Dfm of our corpus
pre_trained<- pre_trained[pre_trained$word %in% featnames(Dfm),] 
nrow(pre_trained) # around 1,000 features less than in our previous GloVe data-frame.
# Why? Cause some words included in the DfM were not included in pre_trained
nrow(glove_dataframe)
pre_trained[1:20, 1:11]

# Once again, let's estimate for each document in the Dfm a value equals to the 
# weighted average of its words in each of the 100 dimensions of the pre-trained WE

doc_embeddingsW2 <- doc_embeddings(Dfm, pre_trained, word_col= "word", method = "weighted")
str(doc_embeddingsW2) # 1500 documents, with 100 columns (1 for each dimension)

# Let's compute a new RF model (we omit here to recompute the naive bayes and the NN model)
Ranger_resW2 <- Function_RF(input=doc_embeddingsW2, k=5, DV=y, ML=ranger)
# A similar result to what we got via a locally trained WE, at least given our corpus of texts
colMeans(Ranger_resW[ , c(1, 2, 3, 4)]) # local WE 
colMeans(Ranger_resW2[ , c(1, 2, 3, 4)]) # pre-trained WE

#############################################
# Global interpretation via WE
#############################################

# let's refit the last model while also computing feature importance (let's use the ranger package for this
# just to make things faster)

# Let's train our model with doc_embeddingsW2 (computed using the pre-trained WE dataset)
set.seed(123)
system.time(RFI <- ranger(y= y, x=doc_embeddingsW2, importance="permutation", 
                          scale.permutation.importance = TRUE))
# 10 most important features:
# here we have problems, given that as features we have meaningless WE dimensions!
head(sort(RFI$variable.importance , decreasing=TRUE), 10)
# Let's then follow what discussed in the Lab and let's use this small test-set:
movie_test <- readRDS("Input Data/Day 3/movie_test.RDS")
str(movie_test)
movie_test$text <- gsub("'"," ",movie_test$text)
test_Corpus <- corpus(movie_test) # here no need to reshape the corpus to sentences given we are NOT training
# anew a WE model on the test-set, but simply using an already trained WE dataset! In our case a pre-trained
# one (see below), but it could have been also a locally trained WE model
summary(test_Corpus)
tok2 <- tokens(test_Corpus , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, 
               remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
Dfm_test <- dfm(tok2 )
# we find the "s" extracted above via  gsub("'"," ",tot$text) out of the genitive saxon 
Dfm_test <- dfm_remove(Dfm_test, min_nchar=2)
Dfm_test <- dfm_trim(Dfm_test,  min_termfreq = 5, verbose=TRUE)
# let's apply to the DfM computed out of the test-set the pre-trained WE dataset
doc_embeddingsWtest <- doc_embeddings(Dfm_test, pre_trained, word_col= "word", method = "weighted")
# let's now predict the test-set

# no need to do any matching between training and test-set here cause they are include the same exact
# features (i.e., the 100 dimensions from the WE space)
setequal(colnames(doc_embeddingsWtest), colnames(doc_embeddingsW2)) 

set.seed(123)
prediction <- predict(RFI, doc_embeddingsWtest)
table(prediction$predictions)

# let's add back the predicted labels to our DfM
str(Dfm_test@docvars)
Dfm_test@docvars$label <- prediction$predictions
str(Dfm_test@docvars)

# Let's group the documents according to the label
Dfm_test_group  <- dfm_group(Dfm_test, groups = label)
# now you have only two rows in the Dfm! One for each label ("pos" and "neg")
dim(Dfm_test_group)
Dfm_test_group
# our target label is "neg"
result_keyness <- textstat_keyness(Dfm_test_group , target = "neg")
str(result_keyness)
# Words with higher keyness scores are considered distinguishing linguistic features for each group
textplot_keyness(result_keyness) 
textplot_keyness(result_keyness,  n = 30L, show_reference = FALSE)
# Let's consider as salient only terms with a significant overrepresentation at a p<0.001 level 
result_keyness2 <- result_keyness[ which(result_keyness$p<=0.001), ]
textplot_keyness(result_keyness2,  n = 30L, show_reference = FALSE)

################################################
# same approach using our locally trained WE
################################################

# Let's train our model with doc_embeddingsW
set.seed(123)
system.time(RFLocal <- ranger(y= y, x=doc_embeddingsW))
# let's apply to the test-set DfM the locally-trained WE dataset
doc_embeddingsWtestLocal <- doc_embeddings(Dfm_test, glove_dataframe, word_col= "word", method = "weighted")
# let's now predict the test-set
set.seed(123)
prediction <- predict(RFLocal, doc_embeddingsWtestLocal)
table(prediction$predictions)
# let's add back the predicted labels to our DfM
str(Dfm_test@docvars)
Dfm_test@docvars$labelLocal <- prediction$predictions
str(Dfm_test@docvars)
# Let's group the documents according to the label
Dfm_test_group  <- dfm_group(Dfm_test, groups = labelLocal)
# now you have only two rows in the Dfm! One for each label ("pos" and "neg")
dim(Dfm_test_group)
Dfm_test_group
# our target label is "neg"
result_keyness <- textstat_keyness(Dfm_test_group , target = "neg")
str(result_keyness)
# Words with higher keyness scores are considered distinguishing linguistic features for each group
textplot_keyness(result_keyness,  n = 30L, show_reference = FALSE)
# Let's consider as salient only terms with a significant overrepresentation at a p<0.001 level 
result_keyness2 <- result_keyness[ which(result_keyness$p<=0.001), ]
textplot_keyness(result_keyness2,  n = 30L, show_reference = FALSE)

#############################################
#############################################
# Expending an existing dictionary via WE
#############################################
#############################################

# Another way of using WE, as discussed in the class, is taking advantage of a pre-trained WE
# to expand a dictionary,say of uncivil words. By looking for other words with semantic 
# similarity to each of these terms, we can identify words that we may not have thought 
# of in the first place, either  because they’re slang, new words or just misspellings 
# of existing words.
# Here we will use a different set of pre-trained word embeddings, which were computed 
# on a large corpus of public Facebook posts on the pages of US Members of Congress.
# Once again it's a WE with 100 dimensions

pre_trained <- read_delim("FBvector.txt", 
                                 skip=1, delim=" ", quote="",
                                 col_names=c("word", paste0("V", 1:100)))

# convert the object from a tibble to a data frame and then to a matrix
class(pre_trained)
pre_trained2 <- as.data.frame(pre_trained)
class(pre_trained2)
str(pre_trained2)
# there is one missing word
table(is.na(pre_trained2$word))
# let's keep only the rows with a word!
pre_trained2 <- pre_trained2[complete.cases(pre_trained2$word), ]
sum(is.na(pre_trained2$word))
# let's transform the column with words into row names
row.names(pre_trained2) <- pre_trained2$word
# and let's now remove the original column with words from the data frame
pre_trained2 <- pre_trained2[-c(1)]
pre_trainedMatrix <- as.matrix(pre_trained2)
str(pre_trainedMatrix)

similarity(pre_trainedMatrix, "idiot")
similarity(pre_trainedMatrix, "crooked")
similarity(pre_trainedMatrix, "bad")
similarity(pre_trainedMatrix, "good")

#############################################
#############################################
# Problems with static (or de-contextualized) WEs
#############################################
#############################################

# You have just one embedding for each given words, regardless of the context in which that
# word appears!

# Let's see another an with 3 sentences involving the word "bank"
textsBank <- c("The man was accused of robbing a bank.", 
               "The man went fishing by the bank of the river",
               "I brought my money to the bank next home")

myCorpus <- corpus(textsBank, to = "sentences")
summary(myCorpus)
tok2 <- tokens(myCorpus , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, 
               remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
DfmEx <- dfm(tok2 )

# let's match the words with the pre-trained WE dataset
pre_trainedEX<- pre_trained[pre_trained$word %in% featnames(DfmEx),] 
print(pre_trainedEX$word)

# we fit a model with a very low number for perplexity given that we have just few words
set.seed(123)
system.time(tsne <-  Rtsne(pre_trainedEX, dims = 2, perplexity = 2))
tsne_plot <- tsne$Y
tsne_plot  <- as.data.frame(tsne_plot)
tsne_plot$word  <- pre_trainedEX$word

# Just one single embedding for the word "bank". And this is misleading from a semantic
# point of view!
ggplot(tsne_plot, aes(x = V1, y = V2, label = word)) + geom_text(size = 3)
