rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()
library(reticulate)
conda_list()
use_condaenv("python_lib", required = TRUE)

library(caret)
library(ggplot2)
library(dplyr)

transformers <- import("transformers")
transformers$logging$set_verbosity_error() # to avoid to report harmless warnings 

##################################
##################################
# NLI and social disaster: an application with a fine-tuned NLI
##################################
##################################

set_200 <- readRDS("Input Data/Day 4/Lab4a2026/set_200.RDS") # validation-set
str(set_200)

nli_classifier <- transformers$pipeline("text-classification", 
                     model = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

# multiple premises 
premises <- set_200$text
# REMEMBER! The hypothesis that you define can greatly affect the results you get! So always
# check what happens if you slightly change it!!!
hypothesis <- c("This text is about a disaster.")
#let's pass a list-of-dicts (one example per element) so the pipeline handles each example individually:
inputs_list <- lapply(seq_along(premises), function(i) {
  reticulate::dict(text = premises[i], text_pair = hypothesis)
})
inputs_list

# 15 seconds
system.time(results_multi <- nli_classifier(inputs_list))

source("Scripts/Day 1/function ENCODER 2026 LUMACSS.R")
df_wide <- make_nli_df_auto(results_multi, premises, hypothesis)
df_wide
table(df_wide$label)

# let's consider contradiction and neutral as "not disaster"
df_wide$target <- ifelse(df_wide$label=="entailment", 1,0)
table(df_wide$target)
table(set_200$target)

mt_NLI <- table(set_200$target, df_wide$target)
mt_NLI
# 0.7 balanced accuracy
confusionMatrix(mt_NLI)
# our baseline: 0.6
prop.table(table(set_200$target))

# Remember: you need always to validate your results! We can once again follow the approach we adopted 
# to enter into the "black box" of an encoder model and using such approach also here (I leave that to you)

###################
# Let's fine-tune the NLI with two labels
# (given that in the fine-tuning set you have just two labels!)
###################

set_300 <- readRDS("Input Data/Day 4/Lab4a2026/set_300.RDS") # fine-tuning set
str(set_300)

premises <-set_300$text
# why doing this replication? Cause in the fine-tuning function we use, the number of
# premises must be equal to the number of hypotheses!
hypotheses <- rep("This text is about a disaster.", length(premises))
length(premises)
length(hypotheses)

labels <- set_300$target  # where 0=no disaster and 1=disaster
# remember that when num_labels = 2L by default train_nli_classifierGPU2 0 = NOT_ENTAILMENT
# 1 = ENTAILMENT - in this case of the socialdisaster class
class(labels)
table(labels)

# let's call the source of all the functions
head(train_nli_classifier)

# around 40 minutes - let's avoid to run the model here and let's open the folder with the saved results
# system.time(trainer <- train_nli_classifier(
#    model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
#    num_labels = 2L, # you have 2 class-labels
#    premises = premises,
#    hypotheses = hypotheses,
#    labels = labels,
#    num_train_epochs = 3L, 
#    test_size = 0.2,
#    learning_rate = 2e-5,
#    device="cpu",
#    dir = "./results_nli2"
#  ))

# trainer_info <- list(
    # best_metric = trainer$state$best_metric,
   #  best_model_checkpoint = trainer$state$best_model_checkpoint,
   # epoch = trainer$state$epoch,
   #   log_history = trainer$state$log_history
   # )
# saveRDS(trainer_info, "Input Data/Day 4/Lab4a2026/trainer_infoFT_dis.RDS")

trainer_info <- readRDS("Input Data/Day 4/Lab4a2026/trainer_infoFT_dis.RDS")
str(trainer_info)

log_df <- bind_rows(trainer_info$log_history)
head(log_df)

# let's extract the performance results over the different epochs
log_df2 <- log_df[!is.na(log_df$eval_f1), ]
str(log_df2)

ggplot(log_df2, aes(x = epoch, y = eval_f1)) +
  geom_point() +
  geom_line() +
  theme_minimal()

# Load fine-tuned model and tokenizer
model_dir <- "./results_nli2/results_nli2/best_model"

# just two labels!
model <- transformers$AutoModelForSequenceClassification$from_pretrained(model_dir)
print(model$config$id2label)
# here we need also to call the tokenizer with two class-labels
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_dir)

nli_classifierFT <- transformers$pipeline(
  "text-classification",
  model = model,
  tokenizer=tokenizer # you need to specify the tokenizer as well with 2 class-labels
)

# multiple premises 
premises <- set_200$text
hypothesis <- c("This text is about a disaster.")
inputs_list <- lapply(seq_along(premises), function(i) {
  reticulate::dict(text = premises[i], text_pair = hypothesis)
})

# around 10 seconds
system.time(results_multiFT <- nli_classifierFT(inputs_list))

df_wide2 <- make_nli_df_auto(results_multiFT, premises, hypothesis)
df_wide2
table(df_wide2$label)
df_wide2$target <- ifelse(df_wide2$label=="ENTAILMENT", 1,0) 
table(df_wide2$target)
table(set_200$target)

mt_NLI2 <- table(set_200$target, df_wide2$target)
mt_NLI2
# 0.84 balanced accuracy vs. 0.7 of the NLI w/o fine-tuning
confusionMatrix(mt_NLI2)