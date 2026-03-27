rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()

library(DALEX)
library(ingredients)
library(ranger)
library(naivebayes)
library(dplyr)
library(quanteda)
library(ggplot2)
library(cowplot)
library(PerformanceAnalytics)
library(cvTools)
library(caret)
library(reshape2)
library(yardstick)
library(data.table)
library(gridExtra)
library(tidyr)
library(reticulate)
conda_list()
use_condaenv("python_lib", required = TRUE)
library(keras3)

#####################################################
# let's prepare the training-set (2 class-labels example)
#####################################################

# Training-set
x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)
str(x)
table(x$choose_one)

x$choose_one <- factor(x$choose_one,  levels=c("0", "1"), labels=c("NoDisaster", 
                                                                   "SocialDisaster"))

myCorpusTwitterTrain <- corpus(x)
tok2 <- tokens(myCorpusTwitterTrain , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_remove(tok2, c("0*"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
Dfm_train  <- dfm_remove(Dfm_train , min_nchar = 2)

# with ingredients it is better working with a matrix rather than a sparsed matrix, at least when
# plotting PdP and ALE
trainData <- as.matrix(Dfm_train)
dim(trainData)
# our DV (it must be a factor!)
y <-  Dfm_train@docvars$choose_one

#####################################################
#####################################################
# Let's run a Global interpretation
#####################################################
#####################################################

######################################################
# RF case
######################################################

# An example of computing Global interpretation

# Let's generate a new training-set and a validation-set
set.seed(123) # for reproducibility
idx <- sample(seq_len(nrow(trainData)), 300)
train <- trainData[idx, ]
dim(train)
yTrain <- y[idx]
length(yTrain)
valData <-trainData[-idx,]
dim(valData)
yVal <- y[-idx]
length(yVal)

# Let's compute a probability RF
set.seed(123)
RFprob <- ranger(y=yTrain, x = train, probability = TRUE)

# We need to define a function to extract the probabilities as a matrix. Indeed ingredients always
# need the probabilities (not the class-labels) when using DALEX::loss_cross_entropy as a a loss function 
# (see below)
pred_funRFprob <- function(m, newdata) {
  as.matrix(predict(m, newdata)$predictions)
}

# it works!
head(pred_funRFprob(RFprob, trainData)) 

# we then need to define a function that keeps together: the ML model, the data on which we want to
# apply the model (here: valData - remember: always compute global interpretation on the validation set), 
# the DV of such data (here: yVal - it must be a factor!), and the function to extract the predictions
# as defined above

explRFProb  <- DALEX::explain(
  model = RFprob,
  data = valData,
  y = yVal, 
  predict_function = pred_funRFprob,
  label = "ranger RF prob"
)

# In the feature_importance function below, we need to specify:
# 1) the object we created above via DALEX::explain
# 2) the loss to compute via loss_function. When dealing with a classification task:
# DALEX::loss_cross_entropy (already discussed with NN!). 
# It measures how well the predicted probabilities match the true distribution (0 or 1). 
# Lower is better. A perfect model has loss = 0.
# Note that when you have a binary dependent variable, DALEX::loss_cross_entropy 
# automatically becomes a binary cross-entropy function. 
# 3) what to use to compute feature importance. When you set type = "ratio", the function transforms 
# the permutation loss metric to show the ratio of the model's error when a feature is permuted versus 
# the model's error on the original data (the "full model" error).
# How it works: the calculation is drop_loss / drop_loss_full_model, where drop_loss_full_model is the 
# baseline error. A value greater than 1.0 indicates that scrambling that variable's information increases 
# the model's prediction error. We can interpret it as the percentage increase in loss.
# You can alternatively select type="difference".
# 4) the repetitions via the argument B. It refers to the number of permutation you apply to each feature. 
# Below we will use B=1 to speed up the process (but typically you would need at least repetitions=5 or 10)

# around 15 seconds
set.seed(123) # for reproducibility
system.time(rf_imp <- feature_importance(
  explRFProb,
  loss_function = DALEX::loss_cross_entropy,
  type = "ratio",
  B = 1,
))

# let's see the results
str(rf_imp)
# let's filter ancillary labels (i.e., they are not tokens)
rf_imp2 <- filter(rf_imp, !variable %in% c("_full_model_", "_baseline_"))
# let's read the top-ten features in terms of their importance
head(rf_imp2[order(rf_imp2$dropout_loss, decreasing = TRUE), ], 10)

# But suppose your best RF model via the grid-search is not a probability RF model (i.e., probability=FALSE).
# How to compute feature importance given that ingredients requires probabilities? 
# Well, we can compute pseudo-probabilities as already discussed earlier!
# That is, by considering the fraction of votes for each class label across the bootstrapped trees

set.seed(123)
RF <- ranger(y=yTrain, x = train)

# The function to compute (pseudo)probabilities is therefore different than the one used above
pred_funRF <- function(m, newdata) {
  votes <- predict(m, newdata, predict.all = TRUE)$predictions
  class_levels <- m$forest$levels    # class labels from ranger
  class_codes  <- seq_along(class_levels)
  pseudo_prob <- sapply(class_codes, function(i) rowMeans(votes == i))
  colnames(pseudo_prob) <- class_levels
  pseudo_prob
}

# it works!
head(pred_funRF(RF, train)) 

explRF  <- DALEX::explain(
  model = RF,
  data = valData,
  y = yVal, 
  predict_function = pred_funRF,
  label = "ranger RF"
)

# around 10 seconds
set.seed(123) # for reproducibility
system.time(rf_impALT <- feature_importance(
  explRF,
  loss_function = DALEX::loss_cross_entropy,
  type = "ratio",
  B = 1,
))

str(rf_impALT)
rf_impALT2 <- filter(rf_impALT, !variable %in% c("_full_model_", "_baseline_")) 
# similar results with and w/o probability=TRUE, although not exactly the same
head(rf_impALT2[order(rf_impALT2$dropout_loss, decreasing = TRUE), ], 10)
head(rf_imp2[order(rf_imp2$dropout_loss, decreasing = TRUE), ], 10)

# Let's generalize what we just discussed by using a function!
# Let's call our usual source of the functions already seen for cross-validation
source("Scripts/Day 2/Function 2026 LUMACSS.R") 

# when you specify externalV=FALSE you will run a global-interpretation exercise. 
head(Function_RF)
# As further arguments for the global interpretation exercise you have:
# repetitions = the number of permutations for feature
# loss = the loss function to employ
# type = if using "ratio" or "difference" to compute feature importance

# Let's apply this function to the best RF model we obtained via the cross-validation exercise.
# To save time, let's directly call the resulting object (you would need around 150 seconds for it).
# repetitions=3 is pretty low as a number as we discussed above!
# system.time(GI_RF2 <- Function_RF (input=trainData, k=5, DV=y , ML=ranger, num.trees=600, mtry=23, 
#                                     min.node.size=3, externalV=FALSE,  repetitions=3,
#                                                  loss=DALEX::loss_cross_entropy, type="ratio"))
# saveRDS(GI_RF2, "GI_RF2.rds")

# If you need to compute global interpretation on a probability RF, just add the argument: 
# "probability=TRUE" (default is FALSE) in the function Function_RF above
GI_RF2 <- readRDS("Input Data/Day 3/GI rds Lumacss2026/GI_RF2.rds")
str(GI_RF2)
resRF2 <- rowMeans(GI_RF2) # let's compute the avg. value for each token across the 5 folds
head(resRF2[order(resRF2, decreasing=TRUE)],10)

# We can also plot them by importance (let's do that just for the top 10 tokens)
fi <- head(resRF2[order(resRF2, decreasing=TRUE)],10)
str(fi)
df_fi <- data.frame(
  attr = names(fi),
  value =fi,
  row.names = NULL
)
str(df_fi)

# all the bars are very close to each other!
ggplot(df_fi, aes(x = reorder(attr, value), value)) +
  geom_bar( stat = 'identity', fill = 'lightblue')+
  theme_bw()+
  labs(
    x = 'Features',
    y = 'Change in prediction error',
    title = 'Variable Importance for the RF'
  )

# let's use coord_cartesian to zoom the plot region we are interested about
ggplot(df_fi, aes(x = reorder(attr, value), value)) +
  geom_col(fill = "lightblue") +
  coord_cartesian(ylim = c(min(df_fi$value), max(df_fi$value))) +
  theme_bw() +
  labs(
    x = "Features",
    y = "Change in prediction error",
    title = "Variable Importance for the RF"
  )  

# Let's check the variability of the avg. feature importance values across the 5 folds (to check for their
# stability across folds)
# convert to long format
fi_long <- as.data.frame(GI_RF2) 
fi_long <- tibble::rownames_to_column(fi_long, "feature")
str(fi_long)
fi_long <- pivot_longer(fi_long, -feature, names_to = "fold",values_to = "FI") # Group by feature: 
# each group = one feature across folds
str(fi_long)

# compute summaries
fi_summary <- group_by(fi_long, feature) 
fi_summary <-  summarise(fi_summary,
                         mean_FI = mean(FI, na.rm = TRUE),
                         var_FI  = var(FI, na.rm = TRUE),
                         sd_FI   = sd(FI, na.rm = TRUE),
                         se_FI   = sd_FI / sqrt(n())
)
str(fi_summary)

# If you want exactly “top 10 by rank” according to mean_FI, even with ties:
fi_summary <- slice_max(fi_summary, mean_FI, n = 10, with_ties = FALSE)
str(fi_summary)
head(fi_summary)

# you can see that the avg. value of "via" is not really stable, contrary to "fire", "kill", "collaps", as
# we could expect
ggplot(fi_summary, aes(x = reorder(feature, mean_FI), y = mean_FI)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.5,  color = "grey40") +
  geom_point(color = "steelblue", size = 2) +
  geom_errorbar(
    aes(ymin = mean_FI - 1.96 * se_FI,
        ymax = mean_FI + 1.96 * se_FI),
    width = 0.2, color = "steelblue") +
  coord_flip() +
  labs(x = "Feature",
       y = "Mean permutation importance",
       title = "Feature importance (CV mean ± 95% CI)")

# note that you can run this global interpretation exercise on a validation set (assuming you have it)
# exactly as we did in the external validity exercise above. 
# Here we write repetitions=1 to speed up the process

# around 10 seconds
system.time(GI_RF2_val <- Function_RF (input=train, k=1, DV=yTrain , validationInput = valData, 
                                       validationDV = yVal, 
                                       ML=ranger, num.trees=600, mtry=23, 
                                       min.node.size=3, externalV=FALSE,  repetitions=1,
                                       loss=DALEX::loss_cross_entropy, type="ratio"))


resRF2_val <- rowMeans(GI_RF2_val)
head(resRF2_val[order(resRF2_val, decreasing=TRUE)],10)

# We now know which features are important for our RF model! 
# Remember, this importance reflects the intrinsic predictive value of a feature by itself. 
# But what about their relationship with the class-labels? Let's compute PDPs and ALE plots!

# To compute the PDPs you need:
# 1) the model trained on the training-set
RF <- ranger(x=trainData, y=y, num.trees=600, mtry=23, 
             min.node.size=3)

# 2) a function to extract the predictions of your trained model as probabilities (not class-labels!)
# here we will use pred_funRF and not pred_funRFprob cause we are fitting a model with probability=FALSE
head(pred_funRF(RF, trainData)) 

# 3) the DALEX::explain function. In our case, let's use the training-set for data and the corresponding
# DV. However, if you have a validation set, you can fit the ML algorithm on the training-set and then
# computing the PD plots on the validation set

explRF  <- DALEX::explain(
  model = RF,
  data = trainData,
  y = y, 
  predict_function = pred_funRF,
  label = "ranger RF"
)

# 4) let's now compute the PD for "fire" and "kill"
# first let's see their actual distribution in our data
dfm_need<- dfm_select( Dfm_train, pattern = "kill")
table(dfm_need@x )
dfm_need<- dfm_select( Dfm_train, pattern = "fire")
table(dfm_need@x )
# let's define the range
my_custom_splits <- list(kill=c(0,1,2),  fire = c(0, 1, 6))
# and now let's plot the PDP. I specify variable_type="categorical" to avoid to treat the two features
# as continuous variables in the plot (they are counts!)
pdp_rf <- partial_dependence(explRF, variables = c("kill", "fire"),
                             variable_type="categorical", variable_splits = my_custom_splits)
plot(pdp_rf)

# indeed w/o  variable_type="categorical": 
pdp_rf2 <- partial_dependence(explRF, variables = c("kill", "fire"),
                             variable_splits = my_custom_splits)
plot(pdp_rf2)

# to plot ALE: by default ingredients plots not-centered ALE (from here the 0s value for 0)
ale_rf <- accumulated_dependence(explRF, variables = c("kill", "fire"),
                                 variable_type="categorical", variable_splits = my_custom_splits)
plot(ale_rf)
# Positive ALE → feature value increases prediction. 
# Negative ALE → feature value decreases prediction.
# For example, when fire=1 the probabilityof SocialDisaster is more than 25% higher than 
# the overall mean prediction
prop.table(table(y))

# to plot a centered ALE:
# a) subtract the mean
ale_rf$ycentered <- ale_rf$`_yhat_` - attr(ale_rf, "mean_prediction") # This makes effects relative to the 
# overall average predicted probability.
# b) centering (so that the average effect across all levels of that feature = 0)
ale_rf <- group_by(ale_rf, `_vname_`, `_label_`) # compute summaries separately for each feature (_vname_) 
# and each class (_label_) so each combination like: fire × SocialDisaster, fire × NoDisaster,
# kill × SocialDisaster, kill × NoDisaster, becomes its own mini-dataset.
ale_rf <- mutate(ale_rf, yale_centered = ycentered - mean(ycentered, na.rm = TRUE)) # for every ALE curve 
# (one curve per feature per class), you subtract its own average
ale_rf <- ungroup(ale_rf) # remove the grouping so future operations are done on the whole dataset again
str(ale_rf)

# After centering: a centered ALE is not on the same scale as the raw model output. 
# It’s on a relative effect scale.
# Positive ALE → feature value increases prediction relative to average. For example, when fire = 1, 
# the model predicts the probability of SocialDisaster to be 10 percentage points higher than the avg effect 
# across all levels of the feature "fire" for that class (normalized to 0).
# Negative ALE → feature value decreases prediction relative to average.

ggplot(ale_rf, aes(x = `_x_`, y = yale_centered, fill = `_label_`)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~ `_vname_`, scales = "free_x") +
  theme_minimal() +
  labs(x = "Category", y = "ALE (centered)", fill = "Model")

######################################################
# NB case
######################################################

# to save time, let's directly call the resulting object (you would need around 70 seconds for it)
# system.time(GI_NB2 <- Function_NB (input=trainData, k=5, DV=y , ML=multinomial_naive_bayes, 
#                                              laplace=2,  externalV=FALSE,  repetitions=3, 
#                                                loss=DALEX::loss_cross_entropy, type="ratio"))
# saveRDS(GI_NB2, "GI_NB2.rds")

GI_NB2 <- readRDS("Input Data/Day 3/GI rds Lumacss2026/GI_NB2.rds")
resNB2 <- rowMeans(GI_NB2)
head(resNB2[order(resNB2, decreasing=TRUE)],10)

# let's now compute PDP and ALE for the NB case

# Remember what you need:
# 1) the model trained on the training-set
NB <- multinomial_naive_bayes(x=trainData, y=y, laplace=2)

# 2) a function to extract the predictions of your trained model as probabilities (not class-labels!)
pred_funP <- function(m, newdata) {
  pred_obj <- predict(m, newdata, type="prob")
  return(as.matrix(pred_obj))
}

# it works!
head(pred_funP(NB, trainData)) 

# 3) the DALEX::explain function. Note however here an important point: 
# DALEX has not an explicit model adapter for naivebayes. So if you write (as with the RF model):

explNB  <- DALEX::explain(
  model = NB,
  data = trainData,
  y = y, 
  predict_function = pred_funP,
  label = "NB"
)

# you get a warning message: 
#  -> model_info :  Model info detected regression task but 'y' is a factor (  WARNING  )
# Why? Because DALEX cannot automatically recognize the naivebayes model as a classification model.
# When DALEX does not recognize a model class, it applies heuristics:
# It calls predict_function(), it sees: numeric output and multiple columns, therefore
# it assumes: “This looks like a regression returning multiple predictions”

# So what to do here?
# First: you must convert y (your DV) to a numeric y
y_num <- as.numeric(y) - 1
table(y_num)

# then let's rewrite explNB with y_num (no warning!)
explNB  <- DALEX::explain(
  model = NB,
  data = trainData,
  y = y_num, 
  predict_function = pred_funP,
  label = "NB",
  residual_function = function(...) NA, # let's not compute residuals for this model 
  # (it makes no sense for a classification task!)
  model_info = DALEX::model_info( # This overrides DALEX guess - we tell it that we are doing a classification
    # task!
    model = NB,
    type = "classification"
  )
)

# now let's plot the PDP for kill and fire
pdp_rf <- partial_dependence(explNB, variables = c("kill", "fire"),
                             variable_type="categorical", variable_splits = my_custom_splits)
plot(pdp_rf)

# and the ALE plot
ale_rf <- accumulated_dependence(explNB, variables = c("kill", "fire"),
                                 variable_type="categorical", variable_splits = my_custom_splits)
plot(ale_rf)

######################################################
# NN case
######################################################

# To save time, let's directly call the resulting object (you would need around 816 seconds for it)
# system.time(GI_NN2 <- Function_NN (input=trainData, k=5, DV=y ,ML=keras_model_sequential, 
#                                    max_features=max_features_toUSE, 
#                                    hidden2=TRUE, units1=20, units2=16, rate1=0.2,
#                                    scaling=FALSE, norm=FALSE,
#                                    epochs=20, batch_size=10, externalV=FALSE,  repetitions=3,
#                                    loss=DALEX::loss_cross_entropy, type="ratio"))
# saveRDS(GI_NN2, "GI_NN2.rds")

GI_NN2 <- readRDS("Input Data/Day 3/GI rds Lumacss2026/GI_NN2.rds")
resNN2 <- rowMeans(GI_NN2)
head(resNN2[order(resNN2, decreasing=TRUE)],10)

# let's now compute PDP and ALE for the NN case

# Remember what you need:
# 1) the model trained on the training-set
tensorflow::set_random_seed(123)
model <- keras_model_sequential()
layer_dense(model , units = 20, activation = "relu", input_shape = ncol(trainData)) 
layer_dropout(model, rate = 0.2)
layer_dense(model , units = 16, activation = "relu") 
layer_dense(model, units=1, activation = "sigmoid") # this is for dependent variable

compile(model, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, trainData, y, epochs = 20, batch_size = 10, verbose = FALSE)

# 2) a function to extract the predictions of your trained model as probabilities (not class-labels!)
pred_funPnn <- function(m, newdata) {
  p1 <- as.numeric(predict(m, newdata))
  p0 <- 1 - p1
  pred_mat <- cbind(p0, p1)
  colnames(pred_mat) <- levels(y)
  return(pred_mat)
}

# it works!
head(pred_funPnn(model, trainData)) 

# 3) the DALEX::explain function. Note that also for keras3 we have the same issue with DALEX
# that we had for naivebayes package. Therefore:

explPnn <- DALEX::explain(
  model = model,
  data = trainData,
  y = y_num, 
  predict_function = pred_funPnn,
  label="NN",
  residual_function = function(...) NA, 
  model_info = DALEX::model_info( 
    model = model,
    type = "classification"
  )
)

# now let's plot the PDP for kill and fire
pdp_rf <- partial_dependence(explPnn, variables = c("kill", "fire"),
                             variable_type="categorical", variable_splits = my_custom_splits)
plot(pdp_rf)

# and the ALE plot
ale_rf <- accumulated_dependence(explPnn, variables = c("kill", "fire"),
                                 variable_type="categorical", variable_splits = my_custom_splits)
plot(ale_rf)

#####################################################
# Does it change anything if we have a multi-class problem rather than a binary one as above? 
# NO! But for the prediction function to use in the case of a Neural Network when plotting PDPs or ALEs
# that becomes (assuming that your DV is labelled "y" and it is a factor):
#####################################################

pred_funPnn <- function(m, newdata) {
  X <- as.matrix(newdata)
  pred_raw <- predict(m, X)
  pred_mat <- as.matrix(pred_raw)
  colnames(pred_mat) <- levels(y) # assuming that your DV is labelled "y" and it is a factor
  return(pred_mat)
}

# Analyze how travelers in February 2015 expressed their feelings on Twitter about US Airline Sentiment.
# Here a sample of 1,300 tweets

airlines <- read.csv("Input Data/Day 1/train_airlines.csv")
str(airlines)

# if you have problems to open this csv file, plz use the below rds file
# airlines <- readRDS("Input Data/Day 1/train_airlines.rds")

table(airlines$airline_sentiment)

myCorpusTwitter <- corpus(airlines)

tok2 <- tokens(myCorpusTwitter , remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)

# remember: with ingredients better working with a matrix rather than a sparsed matrix, at least when
# plotting PdP and ALE
trainData <- as.matrix(Dfm_train)
dim(trainData)
y <-  as.factor(Dfm_train@docvars$airline_sentiment)
table(y)

######################################################
# NB case
######################################################

# to save time, let's directly call the resulting object (you would need around 200 seconds for it)
# system.time(GI_NB3 <- Function_NB (input=trainData, k=5, DV=y , ML=multinomial_naive_bayes, 
#                                              laplace=1,  externalV=FALSE,  repetitions=3, 
#                                                 loss=DALEX::loss_cross_entropy, type="ratio"))
# saveRDS(GI_NB3, "GI_NB3.rds")

GI_NB3 <- readRDS("Input Data/Day 3/GI rds Lumacss2026/GI_NB3.rds")
resNB3 <- rowMeans(GI_NB3)
head(resNB3[order(resNB3, decreasing=TRUE)],10)

# We can also plot them by importance (let's do that just for the top 10 tokens)
fi <- head(resNB3[order(resNB3, decreasing=TRUE)],10)
str(fi)
df_fi <- data.frame(
  attr = names(fi),
  value =fi,
  row.names = NULL
)
str(df_fi)

# all the bars are very close to each other!
ggplot(df_fi, aes(x = reorder(attr, value), value)) +
  geom_bar( stat = 'identity', fill = 'lightblue')+
  theme_bw()+
  labs(
    x = 'Features',
    y = 'Change in prediction error',
    title = 'Variable Importance for the NB'
  )

# so let's use coord_cartesian to zoom the plot region we are interested about
ggplot(df_fi, aes(x = reorder(attr, value), value)) +
  geom_col(fill = "lightblue") +
  coord_cartesian(ylim = c(min(df_fi$value), max(df_fi$value))) +
  theme_bw() +
  labs(
    x = "Features",
    y = "Change in prediction error",
    title = "Variable Importance for the SVM"
  )  

# Let's check the variability of the avg. feature importance value across the 5 folds
# convert to long format
fi_long <- as.data.frame(GI_NB3) 
fi_long <- tibble::rownames_to_column(fi_long, "feature")
str(fi_long)
fi_long <- pivot_longer(fi_long, -feature, names_to = "fold",values_to = "FI") # Group by feature: 
# each group = one feature across folds
str(fi_long)

# compute summaries
fi_summary <- group_by(fi_long, feature) 
fi_summary <-  summarise(fi_summary,
                         mean_FI = mean(FI, na.rm = TRUE),
                         var_FI  = var(FI, na.rm = TRUE),
                         sd_FI   = sd(FI, na.rm = TRUE),
                         se_FI   = sd_FI / sqrt(n())
)
str(fi_summary)

# If you want exactly “top 10 by rank” according to mean_FI, even with ties:
fi_summary <- slice_max(fi_summary, mean_FI, n = 10, with_ties = FALSE)

# pretty stable features' scores
ggplot(fi_summary, aes(x = reorder(feature, mean_FI), y = mean_FI)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.5,  color = "grey40") +
  geom_point(color = "steelblue", size = 2) +
  geom_errorbar(
    aes(ymin = mean_FI - 1.96 * se_FI,
        ymax = mean_FI + 1.96 * se_FI),
    width = 0.2, color = "steelblue") +
  coord_flip() +
  labs(x = "Feature",
       y = "Mean permutation importance",
       title = "Feature importance (CV mean ± 95% CI)")

# Let's compute PDPs and ALE plots!

# To compute the PDPs you need to:
# 1) the model trained on the training-set
NB <- multinomial_naive_bayes(x=trainData, y=y, laplace=1)

# 2) a function to extract the predictions of your trained model as probabilities (not class-labels!)
pred_funP <- function(m, newdata) {
  pred_obj <- predict(m, newdata, type="prob")
  return(as.matrix(pred_obj))
}

# it works!
head(pred_funP(NB, trainData)) 

# 3) the DALEX::explain function (remember the problems discussed above with DALEX with naivebayes)
y_num <- as.numeric(y) - 1
table(y_num)

# then let's rewrite explNB with y_num (no warning!)
explNB  <- DALEX::explain(
  model = NB,
  data = trainData,
  y = y_num, 
  predict_function = pred_funP,
  label = "NB",
  residual_function = function(...) NA, # let's not compute residuals for this model 
  # (it makes no sense for a classification task!)
  model_info = DALEX::model_info( # This overrides DALEX guess - we tell it that we are doing a classification
    # task!
    model = NB,
    type = "classification"
  )
)

# let's explore the PDPs and the ALEs for the top 2 features
head(resNB3[order(resNB3, decreasing=TRUE)],10)
# let's see their actual distribution in our dfm
dfm_need<- dfm_select( Dfm_train, pattern = "thank")
table(dfm_need@x )
dfm_need<- dfm_select( Dfm_train, pattern = "hour")
table(dfm_need@x )

# let's define the range
my_custom_splits <- list(thank=c(0,1,2,3,4),  hour = c(0, 1, 2))

# and now let's plot the PDP
pdp_rf <- partial_dependence(explNB, variables = c("thank", "hour"),
                             variable_type="categorical", variable_splits = my_custom_splits)
plot(pdp_rf)

# to plot ALE 
ale_rf <- accumulated_dependence(explNB, variables = c("thank", "hour"),
                                 variable_type="categorical", variable_splits = my_custom_splits)
plot(ale_rf)

# to plot centered ALE
ale_rf$ycentered <- ale_rf$`_yhat_` - attr(ale_rf, "mean_prediction")
ale_rf <- group_by(ale_rf, `_vname_`, `_label_`) 
ale_rf <- mutate(ale_rf, yale_centered = ycentered - mean(ycentered, na.rm = TRUE)) 
ale_rf <- ungroup(ale_rf) 

ggplot(ale_rf, aes(x = `_x_`, y = yale_centered, fill = `_label_`)) +
  geom_col(position = "dodge") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~ `_vname_`, scales = "free_x") +
  theme_minimal() +
  labs(x = "Category", y = "ALE (centered)", fill = "Model")

######################################################
# the ranger and the NN case are exactly the same as above (but for what noted about the prediction 
# function to use in the case of a Neural Network when plotting PDPs or ALEs!
######################################################
