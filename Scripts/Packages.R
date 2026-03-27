### Installing Packages ###

# Week 1 -------------------------------------------------------------------

install.packages("quanteda", repos="http://cran.us.r-project.org")
install.packages("quanteda.textstats", repos="http://cran.us.r-project.org")
install.packages("quanteda.textplots", repos="http://cran.us.r-project.org")
install.packages("quanteda.textmodels", repos="http://cran.us.r-project.org")
install.packages("readtext", repos="http://cran.us.r-project.org")
install.packages("devtools", repos="http://cran.us.r-project.org")
devtools::install_github("quanteda/quanteda.corpora")
install.packages("ggplot2", repos="http://cran.us.r-project.org")
install.packages("SnowballC", repos="http://cran.us.r-project.org")
install.packages('ggplot2', repos='http://cran.us.r-project.org')
install.packages('SnowballC', repos='http://cran.us.r-project.org')
install.packages('corrplot', repos="http://cran.us.r-project.org")
install.packages('DT', repos="http://cran.us.r-project.org")
install.packages('dplyr', repos="http://cran.us.r-project.org")
install.packages('reshape2', repos="http://cran.us.r-project.org")

install.packages("naivebayes", repos='http://cran.us.r-project.org')
install.packages("ranger", repos='http://cran.us.r-project.org')
install.packages("vip", repos='http://cran.us.r-project.org')
install.packages("cowplot", repos='http://cran.us.r-project.org')

install.packages("PerformanceAnalytics", repos='http://cran.us.r-project.org')
install.packages("cvTools", repos='http://cran.us.r-project.org')
install.packages("caret", repos='http://cran.us.r-project.org')
install.packages("yardstick", repos='http://cran.us.r-project.org')
install.packages("data.table", repos='http://cran.us.r-project.org')
install.packages("gridExtra", repos="http://cran.us.r-project.org")
install.packages("tidyr", repos="http://cran.us.r-project.org")


# KERAS3 ------------------------------------------------------------------

install.packages("reticulate", repos='http://cran.us.r-project.org')
library(reticulate)
# create the virtual ENV
conda_create("keras3") # or call your virtual ENV as you want
conda_list()
use_condaenv("keras3", required = TRUE)
py_config()$python
py_install( c("tensorflow", "keras"),pip = TRUE)
install.packages("keras3")

# Then clear environment and restart R session
.rs.restartR()

library(reticulate)
conda_list()
use_condaenv("keras3", required = TRUE)
py_config()$python

library(keras3)
library(reticulate)
tf <- import("tensorflow")

# TensorFlow version
tf$`__version__`

#Keras version
tf$keras$`__version__`
install.packages("tensorflow")

# check if it works
set.seed(123)
x <- matrix(runif(50), nrow= 10, ncol= 5)
y <- sample(0:1, 10, replace = TRUE)
tensorflow::set_random_seed(123)
model2 <- keras_model_sequential()
layer_dense(model2, units = 4, activation = "relu",
  input_shape= 5)
layer_dense(model2, units = 1, activation = "sigmoid")
compile(model2,
        optimizer="rmsprop",loss="binary_crossentropy",
        metrics = c("accuracy"))
hist <- fit(model2, x, y, epochs = 5, batch_size= 2)
hist
hist$metrics


# Week 2 ------------------------------------------------------------------

install.packages("ingredients", repos="http://cran.us.r-project.org")
install.packages("DALEX", repos="http://cran.us.r-project.org")
install.packages("text2vec", repos="http://cran.us.r-project.org")
install.packages("Rtsne", repos="http://cran.us.r-project.org")
install.packages("lsa", repos="http://cran.us.r-project.org")
install.packages("tidyverse", repos="http://cran.us.r-project.org")
install.packages("glue", repos="http://cran.us.r-project.org")
install.packages("tidytext", repos = "https://cloud.r-project.org")


# Encoders ----------------------------------------------------------------

library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
py_install(c("transformers", "numpy",
             "datasets", "random", "scikit-learn"), pip =
             TRUE)
py_install(c("accelerate", "torch",
             "transformers[torch]"), pip = TRUE)
conda_list()


# BERT --------------------------------------------------------------------

library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
py_install(c("transformers", "numpy",
             "datasets", "random", "scikit-learn"), pip =
             TRUE)
py_install(c("accelerate", "torch",
             "transformers[torch]"), pip = TRUE)

conda_list()
