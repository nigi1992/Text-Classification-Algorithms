rm(list=ls(all=TRUE))
setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(quanteda)
library(naivebayes)
library(ggplot2)
library(ranger)
library(vip)
library(dplyr)
library(cowplot)

# The data we will be using are some English social media disaster tweets discussed in 
# this article: https://arxiv.org/pdf/1705.02009.pdf 
# It consists of a number of tweets regarding accidents mixed in with a selection control
# tweets (not about accidents)

#####################################################
# FIRST STEP: let's create the DfM for the training-set
#####################################################

x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)
str(x)

# if you have problems to open this csv file, plz use the below rds file
# x <- readRDS("train_disaster.rds")

# Let'c convert the "choose_one" variable into a factor variable. Why? Cause the naivebayes package
# requires a factor variable as its DV to understand you are dealing with a classification task
x$choose_one <- factor(x$choose_one,  levels=c("0", "1"), 
                       labels=c("NoDisaster", "SocialDisaster"))
str(x)

table(x$choose_one)
prop.table(table(x$choose_one))
nrow(x)

myCorpusTwitterTrain <- corpus(x)
tok2 <- tokens(myCorpusTwitterTrain , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
# let's also remove the unicode symbols
tok2 <- tokens_remove(tok2, c("0*"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)

# Let's trim the dfm in order to keep only tokens that appear in 2 or more tweets 
# (tweets are short texts!) and let's keep only features with at least 2 characters
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
Dfm_train  <- dfm_remove(Dfm_train , min_nchar = 2)
topfeatures(Dfm_train , 20)  # 20 top words

#####################################################
# SECOND STEP: let's create the DfM for the test-set
#####################################################

x10 <- read.csv("Input Data/Day 1/test_disaster.csv", stringsAsFactors=FALSE)
str(x10)

# if you have problems to open this csv file, plz use the below rds file
# x10 <- readRDS("test_disaster.rds")

nrow(x10)
myCorpusTwitterTest <- corpus(x10)
tok <- tokens(myCorpusTwitterTest , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
              split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok <- tokens_remove(tok, stopwords("en"))
tok <- tokens_remove(tok, c("0*"))
tok <- tokens_wordstem (tok)
Dfm_test <- dfm(tok)
Dfm_test<- dfm_trim(Dfm_test, min_docfreq = 2, verbose=TRUE)
Dfm_test<- dfm_remove(Dfm_test, min_nchar = 2)
topfeatures(Dfm_test , 20)  # 20 top words

#####################################################
# THIRD STEP: Let's make the features identical between train and test-set by passing 
# Dfm_train to dfm_match()  as a pattern. After this step, we can "predict" the test-set 
# by employing only the features included in the training-set.
# If you avoid using dfm_match would generate a problem for example with the RF predictions
# on the test-set. That is, w/o dfm_match you would get the following error: 
# "Error: One or more independent variables not found in data"
#####################################################

setequal(featnames(Dfm_train), featnames(Dfm_test)) 
nfeat(Dfm_test)
nfeat(Dfm_train)
test_dfm  <- dfm_match(Dfm_test, features = featnames(Dfm_train))
nfeat(test_dfm)
setequal(featnames(Dfm_train), featnames(test_dfm ))

# Of course if you have a unique dataset including both the training (i.e., the documents 
# human-codified) and the test-set you could avoid this THIRD STEP, given that you would 
# generate a unique Dfm with the same number of columns for all the documents/rows 
# irrespective of their status (i.e., being part of the training or the test-set). 
# See the EXTRA script on the home-page of the course for an example in this regard.

#####################################################
# FOURTH STEP: Let's convert the two DfMs into matrices for the ML algorithms to work
#####################################################

trainDM <- as.matrix(Dfm_train) # dense matrix
str(trainDM )
class(trainDM)
object.size(trainDM )

# Note that we can save the dfm as a compressed sparse matrix to save storing space and 
# speed up the algorithms! So let's do that!
# What's the meaning of that? Take a look at the EXTRA slides on the home-page!

train <- as(Dfm_train, "dgCMatrix") # compressed matrix
object.size(train)
object.size(trainDM)/object.size(train )

str(trainDM)
str(train)

head(trainDM)
head(train)

# The difference between a dense and a compressed matrix is going to make a HUGE difference when you have 
# a very large Dfm! 
test <- as(test_dfm, "dgCMatrix")

# Finally: we could have also converted the dfm into a dataframe via:
# train_DF <-convert(Dfm_train, to="data.frame")
# and then using the dataframe as an input for our ML algorithm. We avoid that here, cause some 
# of the ML algorithms we will employ in the next classes require a matrix not a dataframe

#####################################################
# FIFHT STEP: let's estimate a ML Model
#####################################################

#####################################################
#####################################################
# Let's see a Naive Bayes Model
#####################################################
#####################################################

# We will use the naivebayes package. Another possible package you can consider is the 
# fastNaiveBayes package.

# Which Naive Bayes model to use? You use the function "multinomial_naive_bayes" 
# if you have counts as IVs (as it happens when you are dealing with a DfM), 
# the function "bernoulli_naive_bayes" if you have binary variables as IVs, 
# or the function "naive_bayes" if you have continuous variables as IVs.
# In each case, you are always dealing with a classification problem

# Given our training-set, we have to use a Multinomial rather given that our features 
# are count variables (i.e., frequencies of word presence in a text)
table(Dfm_train@x )

# Which is our DV?
str(Dfm_train@docvars$choose_one) # it is a factor variable (the typical format for a 
# classification task). Good!

# Let's not specify any Laplace smoother in the formula below. By default its value is 0.5 
# in the package. Remember, however, this is a tuning parameter! More on this later on 

# Of course instead of y=Dfm_train@docvars$choose_one we can also write: 
# y$x$choose_one as far as all the training-texts included in our original dataframe are 
# also included in our dfm

system.time(NB <- multinomial_naive_bayes(x=train, y=Dfm_train@docvars$choose_one))
NB
prop.table(table(Dfm_train@docvars$choose_one)) # our priors

# Let's see the association between words and probabilities (i.e., the matrix with the 
# class conditional parameter estimates - i.e., the likelihood!).
# Take a look at "water" and "cream". The likelihood for the former is higher for a tweet 
# discussing about a disaster (Dfm_train@docvars$choose_one="SocialDisaster") compared 
# to irrelevant tweets (Dfm_train@docvars$choose_one="NoDisaster"), 
# i.e., p(water|SocialDisaster)> p(water|NoDisaster); the opposite happens for the word 
# cream. In other words, by focusing on this type of comparison across likelihoods, we can have an
# idea of which features are more relevant for our model in correctly predicting the observations
# in the training-set for each class-label! 
# Note that we get normalized probabilities, so they sum to 1 for each feature-class combination.

head(NB$params, 10)

# Let's investigate about this issue a bit more (we will go back on this issue a lot in 
# the next classes...). Let's first identify for each feature which is the highest
# value between the SocialDisaster and the NoDisaster label .

NB_prob <- as.data.frame(NB$params)
NB_prob$Feature <- row.names(NB_prob)
str(NB_prob)
NB_prob$winner <- apply(NB_prob[c(1:2)], 1, FUN=max)
str(NB_prob)
NB_prob$class <- ifelse(NB_prob$winner == NB_prob$NoDisaster, 
                        "NoDisaster", "SocialDisaster")
str(NB_prob)

# Let's then save the "SocialDisaster" and "NoDisaster" features in two different
# dataframes
nodisaster <- NB_prob[ which(NB_prob$class =="NoDisaster"), ]
disaster <- NB_prob[ which(NB_prob$class =="SocialDisaster"), ]

# Let's finally check the words that present the highest likelihood in the two cases
# (if they make sense to us!)
print(head(disaster [order(disaster $winner , decreasing=TRUE),], 15)) # features with the highest likelihood for "SocialDisaster"
print(head(nodisaster [order(nodisaster $winner , decreasing=TRUE),], 15)) # features with the highest likelihood for "NoDisaster"

# We can also plot them 
str(NB_prob)
# Let's define a function that allows to do that
plot_top_features <- function(df, class_col, class_name, fill_color = "lightblue") {
  df %>%
    filter(class == class_name) %>%
    arrange(desc(.data[[class_col]])) %>%
    slice_head(n = 15) %>%
    ggplot(aes(x = reorder(Feature, .data[[class_col]]), y = .data[[class_col]])) +
    geom_bar(stat = "identity", fill = fill_color) +
    coord_flip() +
    theme_minimal() +
    labs(
      title = paste("Top 15 Features for", class_name),
      x = NULL,
      y = class_col
    )
}

# Plot for both classes
plot_top_features(NB_prob, "SocialDisaster", "SocialDisaster")
plot_top_features(NB_prob, "NoDisaster", "NoDisaster")

# plotting the two plots together
p1 <- plot_top_features(NB_prob, "SocialDisaster", "SocialDisaster",
  fill_color = "steelblue"
)

p2 <- plot_top_features(NB_prob, "NoDisaster", "NoDisaster",
  fill_color = "tomato"
)

plot_grid(p1, p2, ncol = 2)

#####################################################
# SIXTH STEP: let's predict the test-set
#####################################################

# Let's now predict the test-set 
predicted_nb <- predict(NB ,test )
table(predicted_nb )
prop.table(table(predicted_nb ))

# You can also decide to predict the probabilities, not the class. In this case, a text 
# with a probability>.5 for class X, will be classified in that class. 
# Note that you can also decide to modify this assignment-rule (i.e., the probability) 

predicted_nb_prob <- predict(NB ,test, type="prob"  )
head(predicted_nb_prob)

predicted_class <- ifelse(predicted_nb_prob[,"SocialDisaster"] > 0.5, 
                          "SocialDisaster", 
                          "NoDisaster")

# of course same results as above!
table(predicted_class)
table(predicted_nb )

head(predict(NB ,test))
head(predict(NB ,test, type="prob" ))

# So basically to train and predict you need just 2 lines of command!
# NB <- multinomial_naive_bayes(x=train, y=Dfm_train@docvars$choose_one)
# predicted_nb <- predict(NB ,test )

#####################################################
#####################################################
# Let's now run a Random Forest
#####################################################
#####################################################

# We will use the ranger package rather than the randomForest one. It is faster and it also allow
# you to employ compressed matrices.
# The ranger package also works with a data frame, not only with a matrix.

# Note that I haven't selected any specific values for the hyperpameters 
# (or tuning-parameters).
# By default, for example, the number of trees employed is 500 (a value that you can change). 
# A RF has also several other hyperparameters. More on this later on!
# Here I specify keep.inbag=TRUE (the default is FALSE) cause I want to save and report 
# how often observations (i.e., texts) are in-bag in each tree

library(ranger)
set.seed(123)  # (define a set.seed for being able to replicate the results!)
system.time(RF <- ranger(y= Dfm_train@docvars$choose_one, x=train, keep.inbag=TRUE))
RF

# By default, the bootstrapped samples (with replacement) have the same size of the 
# original training-set. Let's check the size for example of the first two bootstrapped samples
sum(unlist(RF$inbag.counts[1]))
sum(unlist(RF$inbag.counts[2]))

# Let's see how often observations/texts are in-bag in each fitted tree. 
# Let's focus on the first (of 500) fitted tree:
RF$inbag.counts[1]
# how many unique obs in the first boostrap sample? very close to the 63.2% discussed in class
sum(RF$inbag.counts[[1]] != 0)/400
# so for example, text 1 appears once, text 4 twice in the bootstrap sample
# on which we fitted our first decision tree model, while text 3 is in the OOB
RF$inbag.counts[2]
# how many unique obs in the second boostrap sample? once again, very close to 63.2%
sum(RF$inbag.counts[[2]] != 0)/400

# To reduce the total number of unique obs., you have to reduce the size of the boostrap sample.
# For example, if you specify sample.fraction=0.8, you we will select 0.8*n 
# (where n=400 obs in our training-set) with replacement - the default value when replace=TRUE 
# is sample.fraction=1
set.seed(123)  
system.time(RF_lower <- ranger(y= Dfm_train@docvars$choose_one, x=train, keep.inbag=TRUE,
                               sample.fraction=0.8))
sum(unlist(RF_lower$inbag.counts[1]))
sum(unlist(RF_lower$inbag.counts[2]))
# here the total number of unique obs. have decreased to 55.7%
sum(RF_lower$inbag.counts[[1]] != 0)/400
# given that Expected unique fraction: 1-exp(-m/n) where n = dataset size; m = sample.fraction,
# to target a given % of unique obs (such as .55) just type:
-log(1-.55) # here sample.fraction must be around .8 as above

# Note also that if we specify replace=FALSE bootstrapping no longer happens — 
# instead, ranger performs subsampling (sampling without replacement). Of course in this case
# sample.fraction must be lower than 1 (otherwise you simply replicate the original training-set).
# By default in this latter case sample.fraction=0.632
set.seed(123)  
system.time(RF_noB <- ranger(y= Dfm_train@docvars$choose_one, x=train, keep.inbag=TRUE,
                             replace=FALSE ))
sum(unlist(RF_noB$inbag.counts[1]))
sum(unlist(RF_noB$inbag.counts[2]))
# the total number of unique obs = 63%
sum(RF_noB$inbag.counts[[1]] != 0)/400

# A natural benefit of the bootstrap resampling (subsampling) process is that random forest
# has an out-of-bag (OOB) sample that gives us a reasonable approximation of the test error. 
# This provides a built-in validation set without any extra work on your part. 
# However, REMEMBER: this is less efficient (especially with medium to large training-set) 
# compared with doing a k-fold cross-validation as we will see later on

RF
RF$prediction.error
# Accuracy: 1-error
1-RF$prediction.error
# but which of the two class-labels our model predict better (or worse)?
RF$confusion.matrix
# note that the OOB confusion matrix accounts for all training observations.
# That is, every observation in the training-set contributes exactly one OOB prediction. 
# ranger takes the majority vote (for classification) or the average (for regression) over the 
# OOB trees for each observation to compute such prediction (reported in the confusion matrix).
# As discussed, the probability that an obs is always in-bag (i.e., never OOB) is
# 0.632^T (where T=number of trees). So with hundreds or thousands of trees, the probability 
# that any observation is never OOB is astronomically small. From here our previous conclusion
# (i.e., every observation in the training-set contributes exactly one OOB prediction)

# that's the overall accuracy
(204+83)/(204+83+34+79)
# accuracy for NoDisaster
204/(204+34)
# accuracy for SocialDisaster
83/(83+79)
# We do better for the former class-label!

# The ranger package has it own way of quantifying the importance of each feature for the
# model via "permutation" (i.e.,changing randomly the value of a feature in our data). 
# We will go back to "permutation" at length later on. This measure gives an estimate 
# of the importance of a particular variable in the prediction exercise. It is computed like this:
# For each tree t, we take the OOB samples for that tree, we permute one variable, we then
# recompute OOB loss for that tree with the permuted variable. So each variable is permuted once 
# per tree. If you have 500 trees, each variable is permuted 500 times (once per tree), with a fresh 
# permutation each time. 
# We then take the difference between the prediction error (error rate for classification) 
# on the OOB portion of the data when permuting the variable and w/o permuting it. 
# This difference between the two is then averaged over all trees. If you also
# specify  scale.permutation.importance = TRUE the difference is normalized (i.e., divided) 
# by the standard deviation of the permutation error differences across the trees for that 
# specific feature.  Doing like this helps to stabilize importance values (features with large variance
# in terms of their importance across trees will be penalized more - on the contrary,variables 
# that matter consistently across many trees get boosted.), to make magnitudes 
# more interpretable and to reduce dependence on the number of trees.

# For example, if the feature "fire" is important in reducing the prediction error on the OOB data
# for the class-label "SocialDisaster", then if we randomly permute its value in a text that already
# includes such feature, increasing it from 1 to 2 or to 3, the prediction error is not going to change: 
# we're already classifying (correctly) that observation as "SocialDisaster". If however we permute that
# value to 0, then the prediction error could increase a lot ceteris paribus. Similarly, if a text does
# not include the word "fire", then there is a higher likelihood that the model will (correctly) 
# classify such text as "NoDisaster". In this case, if we would permute the original 0 value for "fire"
# to 1 or 2 or 3, there is a higher probability to increase the error of the model (given that now we
# would classify such text as "SocialDisaster" ceteris paribus).

# Note the difference with respect to the likelihoods of the Naive Bayes model: the likelihood tells us 
# which are the most relevant features in predicting correctly the observations WITHIN the training-set. 
# Permuting the OOB data tells us which which are the most relevant features in predicting correctly the
#  observations on which the RF has NOT BEEN trained (i.e., observations outside the training-set for that 
# specific bunch of bootstrapped trees). This is a crucial difference on which we will return later on.

set.seed(123)
system.time(RFI <- ranger(y= Dfm_train@docvars$choose_one, x=train, importance="permutation", 
                          scale.permutation.importance = TRUE))
head(RFI $variable.importance)

# 10 most important words
head(sort(RFI$variable.importance , decreasing=TRUE), 10)

# let's graph the result
vip(RFI) + ggtitle("Random forest")

# From the above results we can conclude that a randomly permuted "fire" variable increases
# the avg. error rate in our prediction by around 19 percent relative to when we use the
# actual variable value of that feature. Thus, we would conclude that including "fire" 
# is important if we are trying to predict our DV in the OOB data.

# Let's predict the test-set using our original RF model. 
# Always specify the same seed you employed to train a RF. In this way, you get:
# same bootstrap samples, same random feature subsets, same split selection order,
# same trees, same predictions. Because all randomness in RF is controlled by the seed.
# If you change the seed → the entire forest changes.

set.seed(123)
system.time(predicted_rf <- predict(RF, test))
str(predicted_rf )
table(predicted_rf$predictions )
prop.table(table(predicted_rf$predictions ))

set.seed(1)
system.time(predicted_rf2 <- predict(RF, test))
table(predicted_rf2$predictions )

# Finally, note that if you add: "predict.all=TRUE", then the returned object is a list of 2 
# components: aggregate, which is the vector of predicted values by the forest (as above), 
# and individual, which is a matrix where each column contains prediction by a tree in 
# the forest.

set.seed(123)
system.time(predicted_rfALL <- predict(RF, test, predict.all=TRUE))
str(predicted_rfALL )
# let's see the prediction of the 500 trees for the first text in the test-set
predicted_rfALL$predictions[1,]
# this text is classified as 2 according to a majority rule (i.e., SocialDisaster - remember that a 
# higher value corresponds to the higher value for the factor choose_one, i.e., "SocialDisaster")
table(predicted_rfALL$predictions[1,])
# let's see the prediction of the 500 trees for the second text in the test-set
predicted_rfALL$predictions[2,]
# this text is classified as 1 (i.e., "NoDisaster")
table(predicted_rfALL$predictions[2,])
# and indeed:
head(predicted_rf$predictions )

# Note that we can also keep the results of the 500 trees prediction to compute
# pseudo-probabilities for each obs (something on which we will return later on), by
# considering the fraction of votes for each class label across the bootstrapped trees:

votes <- predicted_rfALL$predictions
str(votes)
y <- Dfm_train@docvars$choose_one
class_levels <- levels(y)
class_codes  <- seq_along(class_levels)
pseudo_prob <- sapply(class_codes, function(i) rowMeans(votes == i))
colnames(pseudo_prob) <- class_levels
head(pseudo_prob)
table(predicted_rfALL$predictions[1,])
417/500

# We can also decide to run a probability RF model as discussed in the class. 
# For doing that, we need to specify probability=TRUE directly in the ML function.

set.seed(123)
system.time(RFprob <- ranger(y= Dfm_train@docvars$choose_one, x=train,  probability=TRUE))
# note that now for the OOB prediction error it reports: "Brier score".
# The lower the Brier score, the better
RFprob
# Note also that the Splitrule: gini shown here is misleading: it is not telling you that Gini impurity is 
# being optimized in probability forests. It uses the multiclass Brier score discussed in the class

set.seed(123)
system.time(predicted_rf_prob <- predict(RFprob, test))
str(predicted_rf_prob )
head(predicted_rf_prob$predictions )
# let's convert now the probabilities into class-labels
predicted_rf_prob_class <- ifelse(predicted_rf_prob$predictions[,"SocialDisaster"] > 0.5, "SocialDisaster", 
                                  "NoDisaster")

# The predictions can be slightly different! You are indeed fitting a different model 
# (RFprob is different than RF - it has the probability=TRUE argument in its fitting). 
table(predicted_rf$predictions )
table(predicted_rf_prob_class )

# Finally, also note that when you compute importance via permutation, the loss function used
# by ranger changes if probability=TRUE. When probability=FALSE ranger uses misclassification error
# as discussed above (classification error rate). When probability=FALSE range uses the Brier score,
# so importance is the difference between the permuted Brier score and the original Brier score
set.seed(123)
system.time(RFIprob <- ranger(y= Dfm_train@docvars$choose_one, x=train, importance="permutation", 
                              scale.permutation.importance = TRUE, probability=TRUE))
head(RFIprob $variable.importance)
head(RFI $variable.importance)

# 10 most important words
head(sort(RFIprob$variable.importance , decreasing=TRUE), 10)
head(sort(RFI$variable.importance , decreasing=TRUE), 10)

#####################################################
#####################################################
# And if you have three or more categories as your DV? Nothing change!
#####################################################
#####################################################

# Let's analyze how travelers in February 2015 expressed their feelings on Twitter about
# US Airline Sentiment. Here a sample of 1,300 tweets

airlines <- read.csv("Input Data/Day 1/train_airlines.csv")
str(airlines)

# if you have problems to open this csv file, plz use the below rds file
# airlines <- readRDS("train_airlines.rds")

table(airlines$airline_sentiment)

myCorpusTwitter <- corpus(airlines)
tok2 <- tokens(myCorpusTwitter , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
# Let's trim the dfm as usual
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
topfeatures(Dfm_train , 20)  # 20 top words

#####################################################
# SECOND STEP: let's create the DfM for the test-set
#####################################################

test_airlines<- read.csv("Input Data/Day 1/test_airlines.csv", stringsAsFactors=FALSE)
str(test_airlines)

# if you have problems to open this csv file, plz use the below rds file
# test_airlines <- readRDS("test_airlines.rds")

myCorpusTwitterTest <- corpus(test_airlines)
tok <- tokens(myCorpusTwitterTest , remove_punct = TRUE, remove_numbers=TRUE, 
              remove_symbols = TRUE, split_hyphens = TRUE, remove_separators = TRUE)
tok <- tokens_remove(tok, stopwords("en"))
tok <- tokens_wordstem (tok)
Dfm_test <- dfm(tok)
Dfm_test<- dfm_trim(Dfm_test, min_docfreq = 2, verbose=TRUE)

#####################################################
# THIRD STEP: Let's make the features identical between train and test-set by passing 
# Dfm_train to dfm_match() as a pattern.
#####################################################

setequal(featnames(Dfm_train), featnames(Dfm_test)) 
test_dfm  <- dfm_match(Dfm_test, features = featnames(Dfm_train))
setequal(featnames(Dfm_train), featnames(test_dfm ))

#####################################################
# FOURTH STEP: Let's convert the two DfMs into matrices for the ML algorithms to work
#####################################################

train <- as(Dfm_train, "dgCMatrix") # compressed matrix
test <- as(test_dfm, "dgCMatrix")

#####################################################
# FIFHT STEP: let's estimate a ML Model
#####################################################

#####################################################
#####################################################
# Let's see a Naive Bayes Model
#####################################################
#####################################################

# As usual we have to use the multinomial_naive_bayes function given that we have counts as our IVs

# Note that our DV is a character! So we should specify it below as a factor when computing
# the ML algorithm below!
class(Dfm_train@docvars$airline_sentiment)

# let's run the model
system.time(NB <- multinomial_naive_bayes(x=train, 
                                          y=as.factor(Dfm_train@docvars$airline_sentiment)))
NB
prop.table(table((Dfm_train@docvars$airline_sentiment)))

# Let's see the association between words and probabilities (i.e., the likelihoods!).
# Take a look at "disappoint". The likelihood for the it is higher for a negative review 
# compared to a neutral or a positive review
# i.e., p(disappoint|Negative) > p(disappoint|Neutral) OR > p(disappoint|Positive). 

head(NB$params, 10)

# Let's once again investigate about this issue a bit more and
# As above, let's first identify the maximum value between the likelihood for the negative, 
# neutral and positive label for each feature

NB_prob <- as.data.frame(NB$params)
NB_prob$Feature <- row.names(NB_prob)
str(NB_prob)

NB_prob$winner <- apply(NB_prob[c(1:3)], 1, FUN=max)
str(NB_prob)
NB_prob$class <- ifelse(NB_prob$winner == NB_prob$negative, "negative",
                      ifelse(NB_prob$winner == NB_prob$positive, "positive", "neutral"))
str(NB_prob)

# let's save the "positive", "negative" and "neutral" words in separated dataframes
negatives <- NB_prob[ which(NB_prob$class =="negative"), ]
positives <- NB_prob[ which(NB_prob$class =="positive"), ]
neutrals <- NB_prob[ which(NB_prob$class =="neutral"), ]

# let's check such words (if they make sense to us!)
print(head(negatives [order(negatives $winner , decreasing=TRUE),], 15)) # features with the highest likelihood for "negative"
print(head(positives [order(positives $winner , decreasing=TRUE),], 15)) # features with the highest likelihood for "positive"
print(head(neutrals [order(neutrals $winner , decreasing=TRUE),], 15)) # features with the highest likelihood for "neutral"

# We can also plot them 
plot_top_features <- function(df, class_col, class_name, fill_color = "lightblue") {
  df %>%
    filter(class == class_name) %>%
    arrange(desc(.data[[class_col]])) %>%
    slice_head(n = 15) %>%
    ggplot(aes(x = reorder(Feature, .data[[class_col]]), y = .data[[class_col]])) +
    geom_bar(stat = "identity", fill = fill_color) +
    coord_flip() +
    theme_minimal() +
    labs(
      title = paste("Top 15 Features for", class_name),
      x = NULL,
      y = class_col
    )
}

# Plot for all classes
str(NB_prob)
plot_top_features(NB_prob, "positive", "positive")
plot_top_features(NB_prob, "negative", "negative")
plot_top_features(NB_prob, "neutral", "neutral")

# plotting the 3 graphs together
p1 <- plot_top_features(NB_prob, "positive", "positive",
                        fill_color = "steelblue")


p2 <- plot_top_features(NB_prob, "negative", "negative",
                        fill_color = "tomato")

p3 <- plot_top_features(NB_prob, "neutral", "neutral",
                        fill_color = "green")

plot_grid(p1, p2, p3, ncol = 2)

#####################################################
# SIXTH STEP: let's predict the test-set
#####################################################

# let's predict the test-set
system.time(predicted_nb <- predict(NB, test))
prop.table(table(predicted_nb ))

#####################################################
#####################################################
# Let's now run a Random Forest
#####################################################
#####################################################

# here, let's compute the RF while also computing the importance for each feature only
# when probability=FALSE (the default)
set.seed(123)
system.time(RFI <- ranger(y= as.factor(Dfm_train@docvars$airline_sentiment), x=train, 
                          importance="permutation", 
                          scale.permutation.importance = TRUE)) # around 30 seconds

# prediction error on the OOB sample
RFI$prediction.error
# Accuracy of the model
1-RFI$prediction.error

# but which of the three class-labels our model predict better (or worse)?
RFI$confusion.matrix
# that's the overall accuracy
(454+229+162)/(454+229+162+124+22+133+38+56+82)
# accuracy for negative
454/(454+124+22)
# accuracy for neutral
229/(229+133+38)
# accuracy for positive
162/(162+56+82)
# We do better for the negative class-label!

# let's check the feature importance
head(RFI $variable.importance)
# 10 most important words
head(sort(RFI$variable.importance , decreasing=TRUE), 10)
# let's graph the result
vip(RFI) + ggtitle("Random forest")

# let's predict the test-set
set.seed(123)
system.time(predicted_rf <- predict(RFI, test))
# but the predictions on the test-set are different between the RF and the naive bayes model.
# So which of the two to use? More on this later on
prop.table(table(predicted_rf$predictions ))
prop.table(table(predicted_nb ))


# Bonus -------------------------------------------------------------------

library(rpart)
install.packages("rattle")
library(rattle)
nation <- read.csv("Input Data/Day 1/Nationality.csv", stringsAsFactors=FALSE)
fit <- rpart(Nationality ~ Sex + Weight, method="class", data=nation,
             minsplit=2, minbucket=1)
fancyRpartPlot(fit, palettes = c("Greens", "Blues"), sub = "")
