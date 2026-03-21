rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()

# Use my virtual environment with keras and tensorflow 
library(reticulate)
use_condaenv("python_lib")
library(dplyr)
library(quanteda)
library(naivebayes)
library(ranger)
library(ggplot2)
library(cowplot)
library(PerformanceAnalytics)
library(cvTools)
library(caret)
library(reshape2)
library(keras3)
library(yardstick)
library(data.table)
library(gridExtra)
library(tidyr) 

#####################################################
# let's prepare the training-set (2 class-labels example)
#####################################################

# As usual, let's employ the social-disaster corpus example
# Training-set
x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)

# if you have problems to open this csv file, plz use the below rds file
# x <- readRDS("train_disaster.rds")

str(x)
x$choose_one <- factor(x$choose_one,  levels=c("0", "1"), labels=c("NoDisaster", 
                                                                   "SocialDisaster"))
myCorpusTwitterTrain <- corpus(x)
tok2 <- tokens(myCorpusTwitterTrain , remove_punct = TRUE, remove_numbers=TRUE, 
               remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_remove(tok2, c("0*"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
Dfm_train  <- dfm_remove(Dfm_train , min_nchar = 2)
topfeatures(Dfm_train , 20)  

# our classes
table(Dfm_train@docvars$choose_one)
# our baseline value for accuracy - the expected result of our simple learner, i.e., 
# a model that always predicts the most frequent class in the training-set (here pos) is .595.
# We must get a higher accuracy via CV for our ML algorithms then!
prop.table(table(Dfm_train@docvars$choose_one))

######################################################
######################################################
# Let's run a CROSS-VALIDATION 
######################################################
######################################################

######################################################
# NB case
######################################################

# our X:
trainData <- as(Dfm_train, "dgCMatrix")
# our Y:
y <- Dfm_train@docvars$choose_one
class(y) # a factor

# An example of cross-validation
# let's split our training-set in 2 folds 
set.seed(123) # set the see for replicability
k <- 2 # the number of folds; it does not matter the number of folds you decide here; the below procedure always will work!
folds <- cvFolds(NROW(trainData ), K=k)
str(folds)

# let's train the model on fold 1 to predict fold 2
train1 <- multinomial_naive_bayes(y=y[folds$subsets[folds$which == 1] ], 
                                  x=trainData[folds$subsets[folds$which == 1], ])
predict1 <- predict(train1, trainData[folds$subsets[folds$which != 1], ])
class_table1 <- table("Predictions"= predict1, "Actual"=y[folds$subsets[folds$which != 1]])
class_table1

# let's train the model on fold 2 to predict fold 1
train2 <- multinomial_naive_bayes(y=y[folds$subsets[folds$which != 1] ], 
                                  x=trainData[folds$subsets[folds$which != 1], ])
predict2 <- predict(train2, trainData[folds$subsets[folds$which == 1], ])
class_table2 <- table("Predictions"= predict2, "Actual"=y[folds$subsets[folds$which == 1]])
class_table2

# we can then use class_table1 and class_table2 to compute the avg. value of accuracy, balanced accuracy, etc.

# Let's generalize on it!
# Let's call the source of all the functions we will employ today. 
source("Scripts/Day 2/Function 2026 LUMACSS.R") 

# this is the function that we will employ to compute the CV for NB
head(Function_NB)

# The function for CV has several arguments. The first 7 arguments are the same for each 
# ML function below. The other arguments refer to the hyperparameters of the specific ML algorithm
# under investigation and to the global interpretation exercise (as we will discuss)

# 1) input: name of the training-set (it must be in matrix form; a compressed matrix is ok!) 
trainData <- as(Dfm_train, "dgCMatrix")

# 2) k: the number of folds to use in the cross-validation exercise

# 3) DV: name of the DV in the training-set (it must be a factor, exactly as in our training-set)
class(Dfm_train@docvars$choose_one)
y <- Dfm_train@docvars$choose_one

# 4) validationInput: name of the validation-set if available (it must be in matrix form; a compressed matrix is ok!)  

# 5) validationDV: name of the DV in the validation-set (it must be a factor)

# 6) ML: the name of the ML algorithm (in this case, multinomial_naive_bayes - if you have counts as IVs -, 
# bernoulli_naive_bayes - if you have binary variables as IVs, or naive_bayes if you have continuous
# variables as IVs)

# 7) seed: the seed number to use

# Since the eight argument, we have the hyperparameters for each ML function that we can fine-tune
# For example, in the case of the naive bayes algorithm, the eight argument in the function refers 
# to the  value of the Laplace smoother. 
# The default value is 0.5. If you specify another value, say laplace=3, this value 
# will overwrite the default Laplace value (i.e., you will compute a model with laplace=3)

# a 5-folds exercise. If you do not add the seed value, you keep the default one (i.e., 123)
Function_NB(input=trainData, k=5, DV=y, ML=multinomial_naive_bayes)
# instead of k=5 you can write k=10
Function_NB(input=trainData, k=10, DV=y, ML=multinomial_naive_bayes)
# you can also specify the Laplace's value. In this case of course you get different results compared to above!
Function_NB(input=trainData, k=5, DV=y, ML=multinomial_naive_bayes, laplace=3)

# let's save our results
NBmulti_res <- Function_NB(input=trainData, k=5, DV=y, ML=multinomial_naive_bayes)
# much higher accuracy than our simple learner! That's good!
colMeans(NBmulti_res[ , c(1:4)])
prop.table(table(Dfm_train@docvars$choose_one))

# Always check if you have some large difference between Accuracy and Balanced Accuracy 
# and/or avg. F1
NBmulti_res
# here we are (reasonably) fine
colMeans(NBmulti_res)

# Let's investigate if are results are stable across folds
str(NBmulti_res)
# convert to long format
cv_long <- as.data.frame(NBmulti_res) 
cv_long <-mutate(cv_long, fold = row_number()) # Creates a new column fold
str(cv_long)
cv_long <-select(cv_long, fold, Accuracy, `Balanced Accuracy`, MCC, `Avg. F1`) # Keep only what you need
str(cv_long)
cv_long <-   pivot_longer(cv_long,
                          cols = c(Accuracy, `Balanced Accuracy`, MCC, `Avg. F1`),
                          names_to = "metric",
                          values_to = "value") # Convert from wide to long format; now each row = one measurement
str(cv_long)

# compute summaries
perf_summary <- group_by(cv_long, metric) # Group by metric: each group = one metric across folds
perf_summary <-  summarise(perf_summary,
                           mean = mean(value),
                           sd   = sd(value),
                           se   = sd / sqrt(n()))
str(perf_summary)

ggplot(perf_summary, aes(x = metric, y = mean)) +
  geom_point( color = "steelblue",size = 3) +
  geom_errorbar(aes( ymin = mean - 1.96 * se,ymax = mean + 1.96 * se),
                width = 0.15,color = "steelblue") +
  labs(x = "", y = "Performance (CV mean ± 95% CI)",
       title = "Model performance across folds") +
  coord_flip()

# But suppose now that you have available a training-set as well as a validation-set. 
# In our case, let's split the original training-set in two, while also
# considering the corresponding y

set.seed(123)  # for reproducibility
idx <- sample(seq_len(nrow(trainData)), 300)
train <- trainData[idx, ]
dim(train)
yTrain <- y[idx]
length(yTrain)
valData <-trainData[-idx,]
dim(valData)
yVal <- y[-idx]
length(yVal)

# We can take advantage of the previous function to do something similar to what the validation_split
# argument does with NN. Here we specify k=1 (or we could also decide of not specifying any value for k) 
# given that we are not doing cross-validation, but simply fitting a trained-model on a validation-set

NBmulti_val <- Function_NB(input=train, DV=yTrain, ML=multinomial_naive_bayes, k=1,
                                       validationInput = valData, validationDV = yVal)
NBmulti_val

##############
# CV with Random Forest 
##############

# If you run the function below you keep the hyperparameters to their default values 
# (i.e., num.trees=500, min.node.size=1, etc.). Moreover note the argument probability=FALSE (the default). 
# If you change it to probability=TRUE, you fit a probability RF that predicts the probabilities rather than
# the class-labels, and the results can be slightly different
head(Function_RF)

Ranger_res <- Function_RF(input=trainData, k=5, DV=y, ML=ranger)
Ranger_res

Ranger_resProb <- Function_RF(input=trainData, k=5, DV=y, ML=ranger, probability=TRUE)
Ranger_resProb
Ranger_res

##############
# CV with NN 
##############

# Once again, the first 7 arguments remains the same! 
head(Function_NN)

# By default, the function computes a NN model with 1 hidden layer. But it can use up to
# 3 hidden layers (more than enough for most text analysis). 
# For a NN, you need always to specify (if you stick to just 1 hidden layer): 
# a) units1 (i.e. the # of nodes in the first hidden-layer), 
# b) number of epochs
# c) batch_size 
# d) max_features. In our case:
max_features_toUSE <- ncol(trainData)
max_features_toUSE
# Further, you need to specify if you don't want to normalize the input data 
# (by writing "scaling=FALSE" - the default value is to normalize the input data) 
# and if you don't want batch normalization (by writing "norm=FALSE" - the default value
# is to use batch normalization); if you want to use layer normalization, you have to specify
# norm=TRUE and norm_type = "layer"

# Here we run a NN model with 1 hidden layer,
# without any normalization of the data, without any batch normalization, with 20 epochs and batch_size=10
NN_res <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                            max_features=max_features_toUSE, 
                          scaling=FALSE, norm=FALSE, units1=16,  
                           epochs=20, batch_size=10)

colMeans(NN_res[ , c(1:4)])

# for using the adamw optimizer you must also have tensorflow installed!
library(tensorflow)

NN_resAW <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                         max_features=max_features_toUSE, 
                         scaling=FALSE, norm=FALSE, units1=16,  
                         epochs=20, batch_size=10, optimizer_type = "adamw")
# here RMSProp better
colMeans(NN_res[ , c(1:4)])
colMeans(NN_resAW[ , c(1:4)])

# if you want to use kernel_initializer you must specify use_init = TRUE
NN_resK <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                        max_features=max_features_toUSE, 
                        scaling=FALSE, norm=FALSE, units1=16,  
                        epochs=20, batch_size=10, use_init = TRUE)

# here better to avoid kernel_initializer 
colMeans(NN_res[ , c(1:4)])
colMeans(NN_resK[ , c(1:4)])

# with scaling and batch normalization - we could have also avoided to add scaling=TRUE, norm=TRUE
# given that by default they are TRUE

NN_Bnorm <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                      max_features=max_features_toUSE, 
                      scaling=TRUE, norm=TRUE, units1=16,  
                      epochs=20, batch_size=10)

colMeans(NN_Bnorm[ , c(1:4)])
colMeans(NN_res[ , c(1:4)])

# with scaling and layer normalization (i.e., norm_type = "layer")

NN_Lnorm <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                       max_features=max_features_toUSE, 
                       scaling=FALSE, norm=TRUE, units1=16,  
                       epochs=20, batch_size=10, norm_type = "layer")

colMeans(NN_Lnorm[ , c(1:4)])
colMeans(NN_res[ , c(1:4)])

# Note that we can also employ this function directly on the texts included in our original 
# data-frame, i.e., x$text, once you have tokenized these texts via keras3, and after your have created 
# a matrix of your new input data
str(x)
text <- x$text
str(text)
max_features_toUSE2 <- 1000

# Create the layer
vectorizer <- layer_text_vectorization(
  max_tokens = max_features_toUSE2,
  output_mode = "count"
)

# Adapt the layer to your text data
adapt(vectorizer, text)

# The vocabulary is stored in the layer:
x_trainALT2 <- vectorizer(text)
class(x_trainALT2)

# transform the tensor into a matrix
ttt <- as.array(x_trainALT2)
class(ttt)
ttt[1:5, 1:10]  # inspect first 5 texts, first 10 tokens

NN_resAlt <-Function_NN(input=ttt, k=5, DV=y, 
                               ML=keras_model_sequential, 
                  max_features=max_features_toUSE2, 
                  scaling=FALSE, norm=FALSE, units1=16,  
                  epochs=20, batch_size=10)

colMeans(NN_res[ , c(1:4)])
colMeans(NN_resAlt[ , c(1:4)])

# And if you want to employ two hidden-layers? 
# Just add hidden2=TRUE and specify the number of units for the second hidden layer with units2
NN_res_2units <- Function_NN(input=ttt, k=5, DV=y, ML=keras_model_sequential, 
                            max_features=max_features_toUSE2, scaling=FALSE, norm=FALSE, 
                            hidden2=TRUE, units1=16, units2=16,
                           epochs=20, batch_size=10)

colMeans(NN_resAlt[ , c(1:4)])
colMeans(NN_res_2units[ , c(1:4)])

# And if you want to employ three hidden-layers? 
# Just add hidden2=TRUE and hidden3=TRUE and specify the number of units for both the second and the third
# hidden layer with units2 and units3
NN_res_3units <- Function_NN(input=ttt, k=5, DV=y, ML=keras_model_sequential, 
                             max_features=max_features_toUSE2, scaling=FALSE, norm=FALSE, 
                             hidden2=TRUE, hidden3=TRUE, units1=16, units2=16, units3=8,
                             epochs=20, batch_size=10)

colMeans(NN_resAlt[ , c(1:4)])
colMeans(NN_res_2units[ , c(1:4)])
colMeans(NN_res_3units[ , c(1:4)])

#########################
# to summarize the results
#########################

result <- as.data.frame(colMeans(NBmulti_res[ , c(1:4)]))
str(result)
result <- cbind(result, as.data.frame(colMeans(Ranger_res [ , c(1:4)])))
result <- cbind(result, as.data.frame(colMeans(Ranger_resProb [ , c(1:4)])))
result <- cbind(result, as.data.frame(colMeans(NN_res [ , c(1:4)])))

str(result)
resultT <- as.data.frame(t(result))
str(resultT)
row.names(resultT)[1] = "NBmulti"
row.names(resultT)[2] = "Ranger"
row.names(resultT)[3] = "Ranger prob"
row.names(resultT)[4] = "NN"

str(resultT)
resultT$algorithm <- row.names(resultT)
str(resultT)

df.long <- reshape2::melt(resultT)
str(df.long)

ggplot(df.long,aes(algorithm,value,fill=variable, color = variable))+ geom_boxplot() + coord_flip() +
theme_bw()  + labs(title = "Cross-validation with k=5") + 
  ylab(label="Values of Performance Statistics") 

# Here the best model appears to be the NB one.
# These are however the results we got by keeping the default values for the hyperparameters 
# of each ML. But what if we change such values? In few words, let's do some grid-search!

###################################################
###################################################
# Grid search
###################################################
###################################################

# We will use always the  function "run_grid_search"
head(run_grid_search)
# Nine arguments: 1) which function to use from the discussed above;
# 2) the hyper_grid we created; 3) and then our usual 7 arguments

###################################################
# (let's tune!) the hyperparameters for the NB
###################################################

# The main hyperparameter is the value of Laplace. So let's explore for different values 
# between 0.5 and 8 (by 0.5)

hyper_gridNB <- expand.grid(
  laplace       = seq(0.5, 8, by = 0.5) # hyperparameter values
)

# if you want to explore for just a selected number of Laplace values, you can also write: 
# "Laplace=c(1, 3, 10)". In this case you would explore just 3 values (1, 3 and 10).

hyper_gridNB
nrow(hyper_gridNB) # 16 possibilities  

dt_nb <- run_grid_search(model_function=Function_NB, hyper_grid=hyper_gridNB, 
                         input=trainData, k = 5, DV = y, ML = multinomial_naive_bayes)

dt_nb
# We are able to slightly increase Accuracy compared to the default value of the Laplace 
# smooth (i.e., 0.5). In terms of F1, among such values, slightly better with Laplace=2
head(arrange(dt_nb, -CV_Accuracy, -CV_Avg_F1 ))

# just to double-check (as well as to check the F1 you get for each single label - 
# to check if your model present some problems in predicting some classes...)
NBres2 <- Function_NB(input=trainData, ML= multinomial_naive_bayes, k=5, DV=y, laplace=2)
NBres2
colMeans(NBres2 [ , c(1:6)])
head(arrange(dt_nb, -CV_Accuracy, -CV_Avg_F1 ))

# note that also here, you could explore a grid-search only on one given validation-set (if available).
# Of course here k=1
dt_nbVal <- run_grid_search(model_function=Function_NB, k=1, hyper_grid=hyper_gridNB, 
                         input=train, DV = yTrain, ML = multinomial_naive_bayes,
                         validationInput=valData, validationDV=yVal)
head(arrange(dt_nbVal, -CV_Accuracy, -CV_Avg_F1 ))

###################################################
# (let's tune!) the hyperparameters for the Random Forest
###################################################
# RF: accuracy at the moment .705. Let's try to improve on that by fine-tuning some
# hyperparameters
colMeans(Ranger_res [ , c(1:4)])

# The main deafault hyperparameters (on top of probability=FALSE or TRUE)
# in the case of a RF are the following ones: 

# 1) "num.trees" (Number of trees to grow; default=500),

# 2) "mtry" (Number of variables randomly sampled as candidates at each split, where p 
# is the number of variables in x; the default is the (rounded down) square root of the 
# number variables sqrt(p). In our case p=23 
floor(sqrt(ncol(trainData)))

# 3) "min.node.size": Minimum size of terminal nodes. This controls the complexity of the 
# trees. Smaller node size allows for deeper, more complex trees while larger node results 
# in shallower trees. 
# This is another bias-variance trade-off where deeper trees introduce more variance 
# (risk of overfitting) and shallower trees introduce more bias (risk of not fully capturing 
# unique patters and relationships in the data). 
# The default is min.node.size=1 for classification (our case)

# 4) "max.depth". The number of splits that each decision tree is allowed to make. 
# It controls once again the tree depth. A value of NULL or 0 (the default) corresponds to
# unlimited depth, a value of 1 to 1 split per tree. 

# 5) "sample.fraction": fraction of observations to sample. 
# Default is 1 for sampling with replacement (i.e., replace=TRUE - the default, that produced a total
# number of unique obs = 0.632 per bootstrap sample) and 0.632 for sampling without replacement
# (i.e., replace=FALSE)

# Here let's play with num.trees, mtry and min.node.size and let's also add the
# probability argument

hyper_gridRanger <- expand.grid(
  num.trees=seq(500, 700, by = 100), # hyperparameter values
  mtry =seq(floor(sqrt(ncol(trainData)))-1, floor(sqrt(ncol(trainData)))+1, by = 1), # hyperparameter values                   
  min.node.size=c(1, 3), # hyperparameter values  
  probability=c(FALSE, TRUE) # let's fit both a RF and a probability RF
)

nrow(hyper_gridRanger ) # 36 possibilities by crossing the tuning-pameters possible values
hyper_gridRanger 

dt_rf <- run_grid_search(model_function=Function_RF, hyper_grid=hyper_gridRanger, 
                         input=trainData, k = 5, DV = y, ML = ranger)

dt_rf
# a possible slight improvement is feasible compared to train a RF with the 
# default hyperparameters
head(arrange(dt_rf, -CV_Accuracy, -CV_Avg_F1 ))

# just to double-check:
Ranger_res2 <- Function_RF(input=trainData, k = 5, DV = y, ML = ranger,
                               num.trees=600, mtry=23, 
                                    min.node.size=3)
colMeans(Ranger_res2[ , c(1:6)])
head(arrange(dt_rf, -CV_Accuracy, -CV_Avg_F1 ))

# and if you want to add sample.fraction to the hypergrid together with replace (i.e., with bootstrapping
# or with sampling w/o replacement)? 
hyper_gridRanger2 <- expand.grid(
  num.trees=seq(500, 700, by = 100), # hyperparameter values
  probability=c(FALSE, TRUE),
  replace=c(FALSE, TRUE),
  sample.fraction=c(0.632, 0.8, 0.9, 1) 
)
nrow(hyper_gridRanger2 ) # 48 possibilities

# of course doing sampling with sample.fraction=1 does not make any sense! Similarly I want to keep
# for replace=TRUE only values from 0.8 and above. So let's subset the grid we created above.
hyper_gridRanger2 <- subset(
  hyper_gridRanger2,
  !(
    (replace == FALSE & sample.fraction == 1) |
      (replace == TRUE  & sample.fraction < 0.8)
  )
)
nrow(hyper_gridRanger2 ) # 36 possibilities
str(hyper_gridRanger2)

# do not run
# dt_rf2 <- run_grid_search(model_function=Function_RF, hyper_grid=hyper_gridRanger2, 
#                          input=trainData, k = 5, DV = y, ML = ranger)
#dt_rf2
# head(arrange(dt_rf2, -CV_Accuracy, -CV_Avg_F1 ))

###################################################
# (let's tune!) the hyperparameters for the NN
###################################################

# Also with respect to a NN model, there are several hyperparameters that are tunable:
# 1) the number of hidden-layers and the number of units in each of them
# 2) the size of the batches (batch_size in the function)
# 3) the number of the epochs (epochs in the function)
# 4) the optimizer algorithm (optimizer_type in the function)
# 5) the learning rate in the optimizer algorithm (learning_rate=0.001 in the function)
# 6) with or without scaling (scaling=TRUE in the function)
# 7) with or without batch normalization (norm=TRUE in the function)
# 8) with or without layer normalization (norm=TRUE & norm_type = "layer" in the function)
# 9) with or without dropout (and if with: the size of it) (rate1, rate2, rate3 = 0 in the function)
# 10) with or without regularization (and if with: the size of it) 
# (regularizer1_l2 (regularizer2_l2) = 0 in the function). The function also allows regularizer_l1_l2(l1, l2)
# if you specify a value for both regularizers
# 11) with or without kernel_initializer (use_init = TRUE in the function)
# 12) you can also decide to change the activation function (by default activation_hidden = "relu")

# Here let's first work directly with the matrix extracted from the DfM
max_features_toUSE <- ncol(trainData)
max_features_toUSE

# A simple example
hyper_gridNN <- expand.grid(
  hidden2      = c(NA, TRUE),        # absent vs present
  units1       = seq(18, 20, by = 2),
  units2       = 16,   # fixed size of second hidden layer
  rate1        = c(0, 0.2), # dropout value for the first hidden layer
  rate2        = c(0, 0.2), # dropout value for the second hidden layer
  norm         = c(TRUE, FALSE),
  norm_type    = c("batch","layer"),
  scaling      = c(TRUE, FALSE),
  optimizer_type=c("rmsprop", "adamw"),
  max_features = max_features_toUSE,  
  batch_size   = 10, 
  epochs       = 20,
  stringsAsFactors = FALSE # this avoids to transform "rmsprop" and "adamw" into factors. Similarly for "batch"
)

str(hyper_gridNN)
hyper_gridNN
nrow(hyper_gridNN ) # 256 possibilities

# Note that:
# norm      = c(TRUE, FALSE),
# norm_type = c("batch","layer"),
# This creates combinations like:
# norm = FALSE , norm_type = "batch"
# norm = FALSE , norm_type = "layer"
# But when norm = FALSE, norm_type is ignored in Function_NN. So it does not matter norm_type.
# So part of our grid is redundant — we are re-running the same model multiple times.
# That wastes time and biases tuning. So let's clean those lines from the grid-search!

hyper_gridNN2 <- subset(
  hyper_gridNN,
  norm == TRUE | (norm == FALSE & norm_type == "batch")
)

nrow(hyper_gridNN2 ) # 192 possibilities
str(hyper_gridNN2)

# you would need around 1 hour.
# to save time, let's not run the NN grid-search, and let's directly call the resulting object
# system.time(dt_nn <- run_grid_search(model_function=Function_NN, hyper_grid=hyper_gridNN, 
#                         input=trainData, k = 5, DV = y, ML = keras_model_sequential))
# saveRDS(dt_nn, "dt_nn.rds")

dt_nn <- readRDS("Input Data/Day 2/dt_nn.rds")
dt_nn

# a possible improvement is feasible compared to train a NN with our previous hyperparameters
head(arrange(dt_nn, -CV_Accuracy, -CV_Avg_F1  ))
colMeans(NN_res [ , c(1:4)])

# just to double-check:
NN_res2 <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                          max_features=max_features_toUSE, 
                          hidden2=TRUE, units1=20, units2=16, rate1=0.2,
                          scaling=FALSE, norm=FALSE,
                          epochs=20, batch_size=10)

colMeans(NN_res2[ , c(1:6)])
head(arrange(dt_nn, -CV_Accuracy, -CV_Avg_F1  ))

# if you want to run a grid-search with the texts in the original data-frame?
# let's follow the usual procedure!
x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)
text <- x$text
max_features_toUSE2 <- 1000

# Create the layer
vectorizer <- layer_text_vectorization(
  max_tokens = max_features_toUSE2,
  output_mode = "count"
)

# Adapt the layer to your text data
adapt(vectorizer, text)

# The vocabulary is stored in the layer:
x_trainALT2 <- vectorizer(text)
# transform the tensor into a matrix
ttt <- as.array(x_trainALT2)

# do not run!
# dt_nn2 <- run_grid_search(model_function=Function_NN_ALL, hyper_grid=hyper_gridNN, 
#                          input=ttt, k = 5, DV = y, ML = keras_model_sequential)


#####################################################
# What can we conclude from this grid-search exercise? 
#####################################################

df_list <- list(best_rowNB <- head(arrange(dt_nb[1:4], , -CV_Accuracy, -CV_Avg_F1), 1),
                best_rowRF <- head(arrange(dt_rf[1:4], , -CV_Accuracy, -CV_Avg_F1), 1),
                best_rowNN <- head(arrange(dt_nn[1:4], , -CV_Accuracy, -CV_Avg_F1), 1))

df_list
resultGrid <- bind_rows(df_list, .id = "Model_ID")  # Adds a Model_ID column to track list indices
resultGrid$ML <- c("NBmulti", "Ranger","NN")
str(resultGrid)
# reshape
df.long2 <- reshape2::melt(resultGrid)
str(df.long2)

# After grid-search, the NN appears to be the strongest candidate model

p1 <- ggplot(df.long,aes(algorithm,value,fill=variable, color = variable))+ geom_boxplot() + coord_flip() +
  theme_bw()  + labs(title = "Cross-validation with k=5") + 
  ylab(label="Values of Performance Statistics") 

p2 <- ggplot(df.long2,aes(ML,value,fill=variable, color = variable))+ geom_boxplot() + 
  coord_flip() +
  theme_bw()  + labs(title = "Grid-search with k=5") + 
  ylab(label="Values of Performance Statistics") 

grid.arrange(p1, p2, ncol = 2)  # Side by side

#####################################################
# A Summary: pipeline for a grid-search
#####################################################

# Always go through these 3 steps:
# 1) create the hyper-grid with the tuning parameters you want to investigate (and always
# begins with a grid-search rather then using the default parameters!)
# 2) run the grid-search function with the appropriate arguments
# 3) select the best model (with the best hyperparameters mix) for your ML according to 2) to predict the test-set

# Alternatively (if the size of your training-set allows it):
# 1) split the training-set into a training-set and a validation-set
# 2) run the grid-search function with the appropriate arguments on the training-set only
# 3) once you have selected the best hyperparameters mix for each ML algorithm, apply them to the
# validation-set once
# 4) select the best model (with the best hyperparameters mix) for your ML according to 3) to predict the test-set
# An example of this last path with our functions with NB:

# 1) create the validation set
set.seed(123)  # for reproducibility
idx <- sample(seq_len(nrow(trainData)), 300)
train <- trainData[idx, ]
dim(train)
yTrain <- y[idx]
length(yTrain)
valData <-trainData[-idx,]
dim(valData)
yVal <- y[-idx]
length(yVal)

# 2) apply the grid-search to the training-set - we focus here just on the NB case
dt_nb <- run_grid_search(model_function=Function_NB, hyper_grid=hyper_gridNB, 
                         input=train, k = 5, DV = yTrain, ML = multinomial_naive_bayes)
head(arrange(dt_nb, -CV_Accuracy, -CV_Avg_F1 ))

# 3) apply the NB with the best hyperparameters mix to the validation-set to store the performance statistics.
# Those are the performance statistics you will report in your paper!
NBmulti_val <- Function_NB(input=train, DV=yTrain, ML=multinomial_naive_bayes, k=1,
                           laplace=3, validationInput = valData, validationDV = yVal)
NBmulti_val

# 4) replicate steps 2) and 3) to identify the best performing ML on the validation-set, and then use it
# to predict the test-set

#####################################################
# Does it change anything if we have a multi-class problem rather 
# than a binary one as above? 
# NO! Let's see an example!
#####################################################

# Analyze how travelers in February 2015 expressed their feelings on Twitter about US Airline Sentiment.
# Here a sample of 1,300 tweets

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
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
topfeatures(Dfm_train , 20)  
train <- as(Dfm_train, "dgCMatrix")

# our classes
table(Dfm_train@docvars$airline_sentiment)
# our benchmark value for accuracy: .461
prop.table(table(Dfm_train@docvars$airline_sentiment))

###########################
# grid-search with the same structure as above (this is just an example of course!)
###########################

# 1) input: name of the training-set (it must be in matrix form; a compressed matrix is ok!) 
trainData <- as(Dfm_train, "dgCMatrix")

# 2) DV:  name of the DV (it must be a factor, exactly as in our training-set)
class(airlines$airline_sentiment)
y <- as.factor(airlines$airline_sentiment)
table(y)

##################
# NB
##################

hyper_gridNB <- expand.grid(
  laplace       = seq(0.5, 8, by = 0.5) # hyperparameter values
)

# if you want to explore for just a selected number of Laplace values, you can also write: 
# "Laplace=c(1, 3, 10)". In this case you would explore just 3 values (1, 3 and 10)

hyper_gridNB
nrow(hyper_gridNB) # 16 possibilities  

dt_nb3 <- run_grid_search(model_function=Function_NB, hyper_grid=hyper_gridNB, 
                         input=trainData, k = 5, DV = y, ML = multinomial_naive_bayes)

dt_nb3

# best NB with laplace=1
head(arrange(dt_nb3, -CV_Accuracy, -CV_Avg_F1 ))

# just to double check
NBres3 <- Function_NB(input=trainData, ML= multinomial_naive_bayes, k=5, DV=y, laplace=1)
colMeans(NBres3 [ , c(1:7)])
head(arrange(dt_nb3, -CV_Accuracy, -CV_Avg_F1 ))

##################
# Ranger
##################

hyper_gridRanger <- expand.grid(
  num.trees=seq(500, 700, by = 100), # hyperparameter values
  mtry =seq(floor(sqrt(ncol(trainData)))-1, floor(sqrt(ncol(trainData)))+1, by = 1), # hyperparameter values                   
  min.node.size=c(1, 3), # hyperparameter values  
  probability=c(FALSE, TRUE)
)

nrow(hyper_gridRanger ) # 36 possibilities
hyper_gridRanger 

# to save time, let's not run the NN grid-search, and let's directly call the resulting object
#dt_rf3 <- run_grid_search(model_function=Function_RF, hyper_grid=hyper_gridRanger, 
#                           input=trainData, k = 5, DV = y, ML = ranger)
# saveRDS(dt_rf3, "dt_rf3.rds")
dt_rf3 <- readRDS(file = "Input Data/Day 2/dt_rf3.rds")

dt_rf3
head(arrange(dt_rf3, -CV_Accuracy, -CV_Avg_F1 ))

# just to double check
Ranger_res3 <- Function_RF(input=trainData, k = 5, DV = y, ML = ranger,
                           num.trees=500, mtry=32, 
                           min.node.size=1, probability=TRUE)
colMeans(Ranger_res3[ , c(1:7)])
head(arrange(dt_rf3, -CV_Accuracy, -CV_Avg_F1 ))

##################
# NN
##################

max_features_toUSE <- ncol(trainData)
max_features_toUSE

# A very simple grid
hyper_gridNN <- expand.grid(
  hidden2      = c(NA, TRUE),        # absent vs present
  units1       = seq(18, 20, by = 2),
  units2       = 16,   # fixed size of second hidden layer
  rate1        = c(0, 0.2), # dropout value for the first hidden layer
  rate2        = c(0, 0.2), # dropout value for the second hidden layer
  norm         = c(TRUE, FALSE),
  scaling      = c(TRUE, FALSE),
  max_features = max_features_toUSE,  
  batch_size   = 10, 
  epochs       = 20
)

# to save time, let's not run the NN grid-search, and let's directly call the resulting object
# dt_nn3 <- run_grid_search(model_function=Function_NN, hyper_grid=hyper_gridNN, 
#                          input=trainData, k = 5, DV = y, ML = keras_model_sequential)
# saveRDS(dt_nn3, "dt_nn3.rds")

dt_nn3 <- readRDS(file = "Input Data/Day 2/dt_nn3.rds")

head(arrange(dt_nn3, -CV_Accuracy, -CV_Avg_F1  ))

# just to double check
NN_res3 <- Function_NN(input=trainData, k=5, DV=y, ML=keras_model_sequential, 
                       max_features=max_features_toUSE, 
                       units1=18,  rate1=0.2, units2=16, rate2=0.2,
                       scaling=FALSE, norm=FALSE,
                       epochs=20, batch_size=10)
colMeans(NN_res3 [ , c(1:7)])
head(arrange(dt_nn3, -CV_Accuracy, -CV_Avg_F1  ))

#####################################################
# What can we conclude from this grid-search exercise? 
#####################################################

# What can we conclude from this grid-search exercise? 

df_list <- list(best_rowNB <- head(arrange(dt_nb3[1:4], , -CV_Accuracy, -CV_Avg_F1), 1),
                best_rowRF <- head(arrange(dt_rf3[1:4], , -CV_Accuracy, -CV_Avg_F1), 1),
                best_rowNN <- head(arrange(dt_nn3[1:4], , -CV_Accuracy, -CV_Avg_F1), 1))
df_list
resultGrid <- bind_rows(df_list, .id = "Model_ID")  # Adds a Model_ID column to track list indices
resultGrid$ML <- c("NBmulti", "Ranger", "NN")
str(resultGrid)
# reshape
df.long2 <- reshape2::melt(resultGrid)
str(df.long2)

# the NN algorithm seems to be the best one in this example
 ggplot(df.long2,aes(ML,value,fill=variable, color = variable))+ geom_boxplot() + 
  coord_flip() +
  theme_bw()  + labs(title = "Grid-search with k=5") + 
  ylab(label="Values of Performance Statistics") 
 
