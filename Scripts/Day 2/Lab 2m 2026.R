rm(list=ls(all=TRUE))
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()

library(reticulate)
# Show available conda environments


reticulate::conda_list()
reticulate::use_condaenv("python_lib")#, conda = "/path/to/conda")
reticulate::use_condaenv("python_lib", conda = "/Users/nicolaswaser/anaconda3/bin/conda")


conda_list()
# I use the virtual environment in my pc where keras and tensorflow are installed as Python packages
# i.e., "python_lib". 
use_condaenv("python_lib")
# Double check: you must get the same conda environment as above
py_config()$python

library(quanteda)
library(keras3)

# let's check if everything is fine
tf <- import("tensorflow")
# TensorFlow version
tf$`__version__`
# Keras version
tf$keras$`__version__`

# Note that keras3 can run on a GPU (not only on a CPU). Using a GPU can
# dramatically speed up the training of your neural network models.
# To run keras3 on a GPU locally, you need:
# 1) A machine with a compatible GPU, and...
# 2) A properly configured Python virtual/conda environment with GPU support.
# The setup process can be somewhat complex and differs across operating systems
# (Windows, macOS, Linux), as it depends on CUDA, drivers, and backend configuration.
# To make things easier, you can instead use the GPU available on Google Colab.
# Please see the EXTRA file on the course homepage for detailed instructions.
# Note that the GPU on Google Colab is available only for a limited amount of time.
# After the allocated usage period is exhausted, Colab will revert to CPU
# (unless you are using a paid Colab plan with extended GPU access).

###########################
###########################
###########################
# Binary exercise
###########################
###########################
###########################

# As usual, let's employ the "disaster" dataset

# There are two ways to work with texts in Keras. Either by starting from a DfM or by 
# working directly with the texts included in our a dataframe (below: "x$text"). 
# We first follow the former way to directly compare our results with our previous 
# exercises with the other MLs we discussed

x <- read.csv("Input Data/Day 1/train_disaster.csv", stringsAsFactors=FALSE)
x$choose_one <- factor(x$choose_one,  levels=c("0", "1"), 
                       labels=c("NoDisaster", "SocialDisaster"))
myCorpusTwitterTrain <- corpus(x)
tok2 <- tokens(myCorpusTwitterTrain , remove_punct = TRUE, remove_numbers=TRUE, 
               remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
# let's also remove the unicode symbols
tok2 <- tokens_remove(tok2, c("0*"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)
Dfm_train  <- dfm_remove(Dfm_train , min_nchar = 2)

# Here we have 2 options to use Keras: 
# FIRST option: we convert as usual our dfm into a matrix, and use this matrix as our input 
x_trainALT <-as(Dfm_train, "dgCMatrix")
dim(x_trainALT)

# Let's initialize a sequential model using the keras_model_sequential() function. 
# Because our corpus is pretty small, let's employ a small network with 2 hidden
# layers, each with 16 units. In general, the less training data you have, the worse
# overfitting will be, and using a small network is one way to mitigate overfitting

# Note that the model needs to know what input shape it should expect in the first hidden
# layer (i.e., the number of nodes in the input layer), so we must specify it in the 
# first layer of our sequential model via the argument input_shape. In this case:
ncol(x_trainALT)
max_features <- ncol(x_trainALT) # total number of features in our matrix
max_features 
# Therefore, we have to write below: "input_shape = c(max_features)".
# You don’t have however to worry about compatibility (after the FIRST PASSAGE), 
# because in Keras the layers you add to your models are dynamically built to match 
# the shape of the incoming layer.
# The last layer is the ouptut one and in this case we write 1 cause we have a binary DV

# Let's use ReLU as the activation function in the hidden layers, and a sigmoid activation 
# in the final (output) layer to output a probability

# We specify a set.seed for replication purpose
tensorflow::set_random_seed(123)
model <- keras_model_sequential()
layer_dense(model , units = 16, activation = "relu", input_shape = c(max_features)) 
layer_dense(model , units = 16, activation = "relu")
layer_dense(model, units=1, activation = "sigmoid") # this is for dependent variable
model

# why 9120 params for the first layer? 
# We have 16 nodes*569 inputs. Why 569 inputs? 
dim(x_trainALT)
max_features
# And for each node you have also a bias besides the weight
16*569+16
# why 272 params for the second layer?
# we have 16 nodes*16 inputs (from the previous hidden layers) + 1 bias for each node
16*16+16
# why 17 nodes for the last layer (the output one)?
# Cause for the final node you have 16 weights coming from the 16 nodes of the second hidden 
# layer + 1 bias for the final node

# Let's compile the model to prepare it for training. When compiling the model, we
# specify a loss function, an optimizer and (eventually) one or more metrics in order to evaluate the model 
# during training.

# Here as a loss function we select as discussed in the class, a binary_crossentropy loss 
# given we are dealing with a binary classification problem.
# Interpretation of Cross-Entropy values:
# Cross-Entropy = 0.00: Perfect predictions.
# Cross-Entropy < 0.02: Great predictions.
# Cross-Entropy < 0.05: On the right track.
# Cross-Entropy < 0.20: Fine.
# Cross-Entropy > 0.30: Not great.
# Cross-Entropy > 1.00: Terrible.

# The optimizer specifies the exact way in which the gradient of the loss will be used to 
# update parameters.
# Here we specify the RMSProp optimizer [rmsprop()] to find the weights and biases that 
# minimize our objective loss function, i.e., binary_crossentropy. 
# If you write optimizer_rmsprop(lr = 0.0001), you can also specify the learning rate!
# The default value for the learning rate is 0.001. Other optimizers are however available, 
# such as "Adam" or "AdamW" via "optimizer = optimizer_adamw()" below

# Formally, all optimizers minimize the same loss and all use gradients from backprop.
# Practically, they behave differently.
# For example, RMSProp optimizer scales learning rate by recent gradient magnitude:
# - If a weight has large gradients → smaller step (to prevent very large changes/explosion)
# - If a weight has small gradients → larger step
# The main alternative (used for example in Transformers) are Momentum, Adam or AdamW
# Momentum:
# Instead of updating parameters using only the current gradient,
# it maintains a velocity vector (moving average of past gradients).
# Like a heavy ball rolling downhill:
# - If gradients point in the same direction for many steps →
#   velocity increases → faster movement in that direction
# - If gradients oscillate (flip sign) →
#   velocity shrinks → oscillations are dampened
# Adam:
# Combines Momentum (first moment estimate) + RMSProp-style adaptive scaling (second moment estimate)
# AdamW:
# Same as Adam but with decoupled weight decay (this controls model complexity,
# improves generalization, makes solutions “simpler”). A quite good option to consider!

# As an evaluation metric, we select accuracy. Note however the difference between loss and accuracy!
# As already explained, the loss can be seen as the distance between the true values 
# and the values predicted by the model.The greater the loss is, more huge is the errors you made
# on the data. Accuracy on the other side can be seen as the number of error you made on the 
# data. That means:
# a low accuracy and huge loss means you made huge errors on a lot of data
# a low accuracy but low loss means you made little errors on a lot of data
# a great accuracy with low loss means you made low errors on a few data (best case)
# a great accuracy but a huge loss, means you made huge errors on a few data

compile(model, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

# Let's finally fit the model!
# Batch_size identifies the size of the batches (typically between 8 and 128). 
# The number of samples is often a power of 2, to facilitate memory allocation on GPU 
# (if you have it on your laptop!). Here we select 10.
# The one entire passing of all the training data through the algorithm is called an Epoch.
# The number of minibatches in our database multiplied by the number of epochs tells us 
# the number of iterations, that is, how many times the gradient was updated during 
# the training phase.

# When you call fit below, the network will start therefore to iterate
# on the training data in mini-batches of 10 samples, 20 times (given that we select epochs=20). 
# At each iteration, the network will compute the gradients of the weights with regard to 
# the loss on the batch, and update the weights accordingly.
# After these 20 epochs, the network will have performed 800 gradient updates 
# (40 per epoch given that you have 40 batches of 10 in your 400 texts-corpus)
ndoc(Dfm_train)
ndoc(Dfm_train)/10 # number of mini-bacthes of 10 samples in our training-set
40*20

# keras3 works both with factors as well as integers as DV. Let's work with factors
class(Dfm_train@docvars$choose_one)
y <- Dfm_train@docvars$choose_one

history <- fit(model, x_trainALT, y, epochs = 20, batch_size = 10)
history
# We can also visualize the metrics of the trained model as follows:
plot(history)
# This graph is crucial to understand if we have any problems of overfitting on our training-set
# (here huge problems!) and if such risk increases over epochs (so to eventually reduce their numbers).
# You can also decide to add "verbose = FALSE"
history <- fit(model, x_trainALT, y, epochs = 20, batch_size = 10, verbose = FALSE)

# Finally, let’s evaluate the model on the test data
xtest <- read.csv("Input Data/Day 1/test_disaster.csv", stringsAsFactors=FALSE)
str(xtest)
myCorpusTwitterTest <- corpus(xtest)
tok2test <- tokens(myCorpusTwitterTest , remove_punct = TRUE, remove_numbers=TRUE, 
                   remove_symbols = TRUE, 
                   split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2test <- tokens_remove(tok2test, stopwords("en"))
# let's also remove the unicode symbols
tok2test <- tokens_remove(tok2test, c("0*"))
tok2test <- tokens_wordstem (tok2test)
Dfm_test <- dfm(tok2test)
Dfm_test <- dfm_trim(Dfm_test , min_docfreq = 2, verbose=TRUE)
Dfm_test  <- dfm_remove(Dfm_test , min_nchar = 2)

# as always, let's use the function setequal to ensure that the features in the training-set
# are included in the test-set
setequal(featnames(Dfm_train), featnames(Dfm_test)) 
test_dfm  <- dfm_match(Dfm_test, features = featnames(Dfm_train))
setequal(featnames(Dfm_train), featnames(test_dfm ))

x_testALT <-as(test_dfm, "dgCMatrix")
dim(x_testALT)
dim(x_trainALT)

# Now, we generate predictions for the test data. Here, we use
# the predict() function to predict the classes for the test data. 
classesPr <-predict(model,x_testALT) # probabilities for class=1
head(classesPr)
# we can also extract the classes
classes <- as.integer(classesPr > 0.5) # classification
table(classes)

# or in one line of command
classes_alt <- as.integer(predict(model, x_testALT) > 0.5)
table(classes_alt)

prop.table(table(classes))
# let's replace 0 and 1 with the correct labels
levels(y)

classes_labels <- factor(
  classes,
  levels = min(classes):max(classes),
  labels = levels(y)
)

table(classes_labels)

# Note that you can easily save and (later) load your model.
# A saved model contains the weight values, the model's configuration, 
# and the optimizer's configuration.

save_model(model, "my_model.keras")
modelSaved <- load_model("my_model.keras")
modelSaved
model

# Keras3 has not it own way of quantifying the importance of each feature (contrary for example
# to a Random Forest).
# As we will see, in this respect we will need to employ another package

# SECOND option to use Keras3: we can work directly on an object that includes the
# texts, and applying to it the tokenization procedure provided by Keras3 itself. 
# This approach is more flexible, in the sense that you can apply it directly to any corpus 
# of texts (i.e., x$text in our case), rather than to a dfm.

str(x)
text <- x$text

# Let's define the number of tokens to extract from the analysis. 
# Here we select 1,000. It means that we are going to extract the top-1,000 words 
# in terms of frequency in our corpus when we create below our matrix
# But we can choose any value of course. 

max_features2 <- 1000 # let's keep the top 1,000 features in terms of frequencies!

# Using the tokenization option from keras3, we have 3 options for the output_mode:
# "binary" → bag-of-words: 1 if word appears, 0 if not
# "count" → bag-of-words: counts of each word
# "tf_idf" → weighted by term frequency–inverse document frequency

# Create the layer - this will convert raw text into numeric features
vectorizer <- layer_text_vectorization(
  max_tokens = max_features2,
  output_mode = "count"
)

# Now let's adapt the layer to your text data. By doing that, it:
# ✔ learns the vocabulary from your text
# ✔ ranks words by frequency
# ✔ builds a lookup table
adapt(vectorizer, text)

# The vocabulary is stored in the layer:
vocab <- get_vocabulary(vectorizer)
head(vocab[-1], 20) # first 20 tokens. Why [-1]? Cause Keras3 automatically inserts [UNK] at index 1 
# (by default) to represent any word not seen during adapt(). That means vocab[1] is [UNK], 
# and all real words start from index 2.
tail(vocab,20)

# Now that the vectorizer is adapted, by applying it to our texts, we:
# ✔ tokenize the texts
# ✔ convert tokens into integer indices
# ✔ compute the bag-of-words counts
# ✔ return a matrix 
x_train2 <- vectorizer(text)
dim(x_train2)

# Let's fit the model
tensorflow::set_random_seed(123)
model <- keras_model_sequential()
layer_dense(model , units = 16, activation = "relu", input_shape = c(max_features2)) 
layer_dense(model , units = 16, activation = "relu")
layer_dense(model, units=1, activation = "sigmoid") 

model

# why do we have now 16016 params for the first layer? 
# We have 16 nodes*1000 inputs. Why 1000 inputs? 
dim(x_train2)
# And for each node you have also a bias besides the weight
16*1000+16
# why 272 params for the second layer?
# we have 16 nodes*16 inputs (from the previous hidden layers) + 1 bias for each node
16*16+16
# why 17 nodes for the last layer (the output one)?
# cause for the final node you have 16 weights coming from the 16 nodes of the second hidden layer + 
# 1 bias for the final node

compile(model, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyNEW <- fit(model, x_train2, y, epochs = 20,batch_size = 10)
historyNEW

# Finally, let’s evaluate the new model we computed on the test data
# let's predict the test-set
str(xtest)
textAlarmtest <- xtest$text
# Note that we use the same tokenizer we created above (i.e., vectorizer) to ensure
# that the same features included in the training-set are also included in the test-set
x_test2 <- vectorizer(textAlarmtest) 
# here you see that we have once again 1,000 columns 
dim(x_test2)
dim(x_train2)

# Let's predict the test data
classes2Alt <- as.integer(predict(model, x_test2) > 0.5)
table(classes2Alt)

# Should we be happy for the high level of accuracy (and low loss) we got above? 
historyNEW
# Not necessarily! Perhaps we have problems with overfitting (i.e., remember the
# bias-variance trade-off)!

# How to check for it? We could run a cross-validation exercise (as we will discuss
# in the next Lab class) or...we can directly use the function validation_split directly from keras!
# The validation_split argument can take float values between 0 and 1
# and specifies the fraction of the data to be used as validation data.
# Below we select 0.2. That implies that we won't run our NN model on 20% of the documents 
# included in our training-set. We will then use this 20% to validate the model. 

tensorflow::set_random_seed(123)
model <- keras_model_sequential()
layer_dense(model , units = 16, activation = "relu", input_shape = c(max_features2)) 
layer_dense(model , units = 16, activation = "relu")
layer_dense(model, units=1, activation = "sigmoid") 

compile(model, optimizer = "rmsprop", loss="binary_crossentropy", metrics = c("accuracy"))

history2V <- fit(model, x_train2, y, epochs = 20, batch_size = 10, validation_split = 0.2)

# Typically, if the accuracy for the validation set is larger than .7/.8 for a binary
# classification problem (as ours), you are in a good shape. Here the accuracy value is not that bad.
# But note also a pretty high loss value. So we are making few overall errors, but when we make it, 
# they are huge!
history2V

# The most common ways to prevent overfitting in neural networks are:
# 1. Get more training data (we just have 400 observations..)
# 2. Reduce the capacity of the network (for example, employ just 1 hidden-layer!)
# 3. Add dropout
# 4. Add weight regularization

# Let’s review the latter two regularization techniques 

# Dropout is one of the most effective and commonly used approaches to prevent overfitting in NNs.
# Dropout randomly drops out (setting to zero) a number of output features in a layer during training. 
# Let’s say a given layer would normally return a vector [0.2, 0.5, 1.3, 0.8, 1.1] 
# for a given input sample during training. After applying dropout, 
# this vector will have a few zero entries distributed at random: 
# for example, [0, 0.5, 1.3, 0, 1.1].
# By randomly removing different nodes, we help prevent the model from fitting patterns to 
# fortuity patterns (noise) that are not significant. We apply drop out with layer_dropout.
# A typical dropout rate is 0.2 and 0.5. 
# In this example I drop out 20% of the inputs going into each layer. 

tensorflow::set_random_seed(123)
modelDrop <- keras_model_sequential()
layer_dense(modelDrop , units = 16, activation = "relu", input_shape = c(max_features2)) 
layer_dropout(modelDrop, rate = 0.2)
layer_dense(modelDrop , units = 16, activation = "relu")
layer_dropout(modelDrop, rate = 0.2)
layer_dense(modelDrop, units=1, activation = "sigmoid") 

compile(modelDrop, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyDrop <- fit(modelDrop, x_train2, y, epochs = 20, batch_size = 10,
                   validation_split = 0.2)

# slightly lower accuracy, but a quite improved loss!
historyDrop
history2V

# Regularization implies putting constraints on the size that the weights can take. 
# Without regularization, weights are updated solely to minimize the data loss.
# This can lead to overfitting if the model is complex (e.g., large weights fitting noise).
# Regularization adds constraints to prevent this.

# L1 regularization adds a penalty to the loss function that the model is minimizing during training 
# equivalent to the absolute value of the weights' magnitude multiplied by λ. 
# Total Loss = Original Loss + λ * Σ|weights|
# Such penalization encourages the model to have as many weights 
# closer to 0 as possible, given that it punishes the model for having large weight values, 
# removing some of the features for the classification (i.e., allowing feature selection).
# This encourages simpler models, given that it prefers smaller, more distributed weights rather than
# a few dominant ones that are too specific to training data (i.e., overfitting)

# L2 regularization adds on the other hand a penalty based on the squared 
# magnitude of weights, preventing weights from reaching extremely high values (since squaring 
# increases with size). However, it does not produce any feature selection. 
# Total Loss = Original Loss + λ * Σ(weights²)

# Both methods therefore can help in preventing overfitting, penalizing complex models, 
# and producing a more generalized model.

# In this example I add the L1 norm regularizer with a multiplier (λ) of 0.001. 
# We could also add an L2 regularization or a combination of L1 and L2 (aka elastic net) 
# with regularizer_l1 and regularizer_l1_l2 respectively by writing:
# kernel_regularizer = regularizer_l1_l2(l1 = 0.001, l2 = 0.001)

# Typical L1 Values:
# Medium regularization: 0.001 to 0.01
# Very strong regularization: 0.01 to 0.1

# Typical L2 Values:
# with small datasets you could consider to increase λ to 0.1 (strong regularization needed)
# with medium datasets also 0.01 can be a good value

tensorflow::set_random_seed(123)
modelR <- keras_model_sequential()
layer_dense(modelR , units = 16, activation = "relu", input_shape = c(max_features2),
            kernel_regularizer = regularizer_l1(0.001))
layer_dense(modelR , units = 16, activation = "relu",
            kernel_regularizer = regularizer_l1(0.001))
layer_dense(modelR, units=1, activation = "sigmoid") 

compile(modelR, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyR<- fit(modelR, x_train2, y, epochs = 20,batch_size = 10, 
               validation_split = 0.2)

# a nice improvement on both loss and accuracy! 
historyR
history2V

# Note that you can also mix all the models above, using dropout & L1/L2 regularization 
# together in the same model as below:

# tensorflow::set_random_seed(123)
# modelAll <- keras_model_sequential()
# layer_dense(modelAll , units = 16, activation = "relu", input_shape = c(max_features2),
#             kernel_regularizer = regularizer_l1(0.001)) 
# layer_dropout(modelAll, rate = 0.2)
# layer_dense(modelAll , units = 16, activation = "relu",
#             kernel_regularizer = regularizer_l1(0.001))
# layer_dropout(modelAll, rate = 0.2)
# layer_dense(modelAll, units=1, activation = "sigmoid")

# In class, we have discussed about the importance of normalizing our input data before 
# feeding it into our NN model. But data normalization can be a concern after every 
# transformation operated by the network. Batch normalization (BN) allows you to do that!

# That is, input normalization (e.g., scale(x)) standardizes your raw features before they enter 
# the network → ensures each input dimension contributes comparably.
# Batch normalization standardizes activations (the outputs of hidden layers) during training.
# So both reduce instability caused by widely varying value ranges — but at different stage,
# while improving model accuracy.

# For each mini-batch during training, BN:
# 1) Computes the mean and variance of the outputs (activations) of a layer
# For example, let's assume: Batch size = 3 and Hidden layer has 4 units/nodes
# So the layer output is a matrix:
# Sample 1: (2, 3, 1, 4)
# Sample 2: (5, 2, 0, 3)
# Sample 3: (4, 1, 2, 6)
# In other words, the layer output for the first obs in the mini-batch is 2 for the first unit,
# for the second unit is 3, etc.
# Batch normalization implies computing the mean and the variance across batches for the same unit/node. 
# That implies moving vertically through the 3 vectors above. 
# Here: for the first node, mean will be mean(2, 5, 4) and variance var(2, 5, 4); 
# for the second unit mean(3, 2, 1) etc.
# 2) Normalizes the outputs of a layer to have mean 0 and variance 1, i.e.
# (2, 3, 1, 4)=(2-mean(2, 5, 4))/var(2, 5, 4), (3-mean(3, 2, 1)/var(3, 2, 1)), etc.
# 3) Then learns 4 extra parameters/weights for each node of your NN: two learnable parameters (γ and β), 
# which can shift and scale the normalized values, plus two non-learnable parameters.
# That is, (2, 3, 1, 4)=γ(2-mean(2, 5, 4))/var(2, 5, 4)+β, γ(3-mean(3, 2, 1)/var(3, 2, 1))+β, etc. 
# Why doing it? Cause we let the NN decides how much normalization it really wants — 
# w/o such parameters, BN would force each layer’s outputs to have a fixed distribution 
# (mean=0, variance=1). But this could limit the model’s flexibility. For example, if the next layer 
# actually benefits from non-zero-centered inputs, BN would remove that ability. That is, the
# learnable weights allow the network to "undo" normalization if needed.
# The two non-learnable (but stored) parameters refer to the running_mean and running_variance. 
# running_mean represents an estimate of the global population mean of that specific neuron over many batches
# during training; similarly for running_variance. These two non-learnable parameters are used
# during inference.
# Why BN is useful? It reduces internal covariate shift (loosely: keeps activations well-scaled), 
# it allows higher learning rates, it makes training faster and more stable and also acts 
# as a mild regularizer. That is, the noise introduced by the BN parameters acts like dropout, 
# reducing overfitting while helping generalization.

# Still BN is dependent on batch size and it is noisy for small batches (like the ones in our example).
# An alternative to BN, is Layer Normalization (LN). LN  normalizes within (not across) each sample and
# across the features of that layer. In other words, going back to our example, this implies computing the mean
# and the variance for each batch across nodes/units:
# Sample 1: (2, 3, 1, 4)
# Sample 2: (5, 2, 0, 3)
# Sample 3: (4, 1, 2, 6)
# (2, 3, 1, 4)=(2-mean(2, 3, 1, 4))/var(2, 3, 1, 4), (3-mean(2, 3, 1, 4)/var(2, 3, 1, 4)), etc.
# So: each sample is normalized independently, and no dependence on batch size.
# When LN is preferred: small batch sizes, NLP models (as we will see), and any case where batch statistics
# are unstable. BN is usually better for large batches and vision models.
# Note moreover that LN has just two learnable weights because the mean and variance are computed 
# from the current sample itself (not across batches as it happens with BN)

# Note that you implement batch_normalization  before the ReLu activation functions.
# Why before ReLU? ReLU kills negative values:
# if you apply ReLU first, then pass it through batch norm, you've already lost the negative part 
# of the data — you can't "center" it properly anymore
# Why use_bias = FALSE below? Keeping b is redundant, given that BatchNorm already has a learnable shift β

tensorflow::set_random_seed(123)
modelBN <- keras_model_sequential()
layer_dense(modelBN , units = 16, activation=NULL, input_shape = c(max_features2), use_bias = FALSE) 
layer_batch_normalization(modelBN)
layer_activation(modelBN, "relu")
layer_dense(modelBN , units = 16, activation = NULL, use_bias = FALSE)
layer_batch_normalization(modelBN) 
layer_activation(modelBN, "relu")
layer_dense(modelBN, units=1, activation = "sigmoid")

# indeed now you can see you compute 64 new weights for each batch_normalization
modelBN
16*4

compile(modelBN, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyBN <- fit(modelBN, x_train2, y, epochs = 20,batch_size = 10, 
                 validation_split = 0.2)

# Not such a dramatic improvement here. Moreover, you clearly see that reducing the # of epochs would
# improve the model (such as epochs=7)
historyBN
history2V

# let's try layer normalization by using layer_layer_normalization() 
# This is directly analogous to the layer_batch_normalization version.

tensorflow::set_random_seed(123)
modelLN <- keras_model_sequential()
layer_dense(modelLN , units = 16, activation=NULL, input_shape = c(max_features2), use_bias = FALSE) 
layer_layer_normalization(modelLN)
layer_activation(modelLN, "relu")
layer_dense(modelLN , units = 16, activation = NULL, use_bias = FALSE)
layer_layer_normalization(modelLN) 
layer_activation(modelLN, "relu")
layer_dense(modelLN, units=1, activation = "sigmoid")

# indeed now you can see you compute 32 new weights for each layer_normalization
modelLN
16*2

compile(modelLN, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyLN <- fit(modelLN, x_train2, y, epochs = 20,batch_size = 10, 
                 validation_split = 0.2)

# No improvement here
historyLN
history2V

# Finally, you can also decide to use kernel_initializer = "he_normal" for the hidden layer 
# and the kernel_initializer = "glorot_uniform" for the ouptput layer. 
# kernel_initializer controls how the weights are randomly selected when the neural network is first created, 
# before any training happens.
# Think of it like different strategies for dealing cards at the start of a game:
# Default/No specification: "Deal cards completely randomly from the deck"
# This might work, but could start with weights that are:
# - Too small → gradients vanish during backpropagation
# - Too large → gradients explode during backpropagation

# he_normal: "Deal cards, but make sure high-value cards are distributed in a specific pattern that works well 
# with ReLU"
# glorot_uniform: "Deal cards with a different pattern that works better with sigmoid/softmax"

# Empirical benefits:
# - Faster convergence (reaches same accuracy in fewer epochs)
# - Better performance on deep networks
# - More stable training (less sensitive to learning rate)
# - Reduces the risk of vanishing and exploding gradients

# with kernel_initializer
tensorflow::set_random_seed(123)
modelKI <- keras_model_sequential()
layer_dense(modelKI , units = 16, activation = "relu", input_shape = c(max_features2),
            kernel_initializer = "he_normal") 
layer_dense(modelKI , units = 16, activation = "relu", kernel_initializer = "he_normal")
layer_dense(modelKI, units=1, activation = "sigmoid", kernel_initializer = "glorot_uniform") 

compile(modelKI, 
        optimizer = "rmsprop", 
        loss="binary_crossentropy",
        metrics = c("accuracy"))

historyKI <- fit(modelKI, x_train2, y, epochs = 20,batch_size = 10, 
                 validation_split = 0.2)

# No significant improvement here
historyKI
history2V

###########################
###########################
###########################
# Multi-class exercise
###########################
###########################
###########################

airlines <- read.csv("Input Data/Day 1/train_airlines.csv")
str(airlines)
table(airlines$airline_sentiment)

myCorpusTwitter <- corpus(airlines)
tok2 <- tokens(myCorpusTwitter , remove_punct = TRUE, remove_numbers=TRUE, 
               remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_wordstem (tok2)
Dfm_train <- dfm(tok2)
Dfm_train <- dfm_trim(Dfm_train , min_docfreq = 2, verbose=TRUE)

# that's our DV (saved as a factor)
y <-as.factor(Dfm_train@docvars$airline_sentiment)
table(y)
class(y)

# Once again we have two options! Either using the matrix of the dfm, or working with a 
# dataframe including texts

# First option:

x_trainALT <-as(Dfm_train, "dgCMatrix")
dim(x_trainALT)

max_features <- ncol(x_trainALT) # total number of features in our matrix
max_features 

# Which changes with respect to a binary DV?
# First: in the output layer we write "units=nlevels(y)" cause we have 3 categories!
# Second: will use a softmax activation so as to output a probability.
# It means the network will output a probability distribution over the 3 different output 
# classes, i.e., for every input sample, the network will produce a 3-dimensional
# output vector, where output[[i]] is the probability that the sample belongs to class i. 
# The 3 scores will sum to 1

# N.B.: if your data is divided into many categories, you may cause information bottlenecks
# if you make the intermediate layers too small compared to the number of the categories. 
# For example, 40 class-labels and your last hidden layer has 16 units

tensorflow::set_random_seed(123)
model <- keras_model_sequential()
layer_dense(model , units = 16, activation = "relu", input_shape = c(max_features)) 
layer_dense(model , units = 16, activation = "relu")
layer_dense(model, units=nlevels(y), activation = "softmax") # this is for dependent variable

model

# why do we have now 18160 params for the first layer? 
# We have 16 nodes*1134 inputs. Why 1134 inputs? 
dim(x_trainALT)
# And for each node you have also a bias besides the weight
16*1134+16
# why 272 params for the second layer?
# we have 16 nodes*16 inputs (from the previous hidden layers) + 1 bias for each node
16*16+16
# why 51 nodes for the last layer (the output one)?
# cause for the final 3 nodes you have 16 weights coming from the 16 nodes of the second hidden layer + 
# 1 bias for each final node
16*3+3

# here we define as loss: sparse_categorical_crossentropy (we have a multi-class DV!)
compile(model, 
        optimizer = "rmsprop", 
        loss="sparse_categorical_crossentropy",
        metrics = c("accuracy"))

# let's fit the model!

history <- fit(model, x_trainALT, y, epochs = 20,  batch_size = 10)
history
# Once again: should we be happy about these values? We don't know! Perhaps our model is overfitting the
# training-set! To check for this we must run a cross-validation exercise (as we will discuss
# in the next lab class), or using the validation_split option
historyV <- fit(model, x_trainALT, y, epochs = 20,  batch_size = 10, validation_split = 0.2)
# clearly a difference between the performance of our NN model on training-set vs. validation-set.
# We should therefore once again explore the impact of drop-out, regularization and batch-normalization
historyV

# let's predict the test-set
xtest <- read.csv("Input Data/Day 1/test_airlines.csv", stringsAsFactors=FALSE)
myCorpusTwitterTest <- corpus(xtest)
tok2 <- tokens(myCorpusTwitterTest , remove_punct = TRUE, remove_numbers=TRUE, 
               remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_wordstem (tok2)
Dfm_test <- dfm(tok2)
Dfm_test <- dfm_trim(Dfm_test , min_docfreq = 2, verbose=TRUE)

# as always, let's use the function setequal 
setequal(featnames(Dfm_train), featnames(Dfm_test)) 
test_dfm  <- dfm_match(Dfm_test, features = featnames(Dfm_train))
setequal(featnames(Dfm_train), featnames(test_dfm ))

x_testALT <-as(test_dfm, "dgCMatrix")
dim(x_testALT)
dim(x_trainALT)

# Now, we generate predictions
classesPr <-predict(model,x_testALT) # probabilities
head(classesPr)
classes <- apply(classesPr, 1, which.max) # classification
table(classes)

# doing everything with one line of command
classes_alt <- apply(predict(model, x_testALT), 1, which.max)
table(classes_alt)

# let's add the proper labels
levels(y)

classes_labels <- factor(
  classes,
  levels = min(classes):max(classes),
  labels = levels(y)
)

table(classes_labels)

# Second option: let's extract the original texts and let's apply to it text_tokenizer
textAir <- airlines$text
max_featuresAir <- 1000 # let's keep the top 1,000 features in terms of frequencies

# Create the layer - this will convert raw text into numeric features
vectorizerAir <- layer_text_vectorization(
  max_tokens = max_featuresAir,
  output_mode = "count"
)

adapt(vectorizerAir, textAir)

# The vocabulary is stored in the layer:
vocab <- get_vocabulary(vectorizerAir)
head(vocab[-1], 20) 
tail(vocab,20)

x_trainAir2 <- vectorizerAir(textAir)
# here you see that we have now 1,000 columns 
dim(x_trainAir2)

tensorflow::set_random_seed(123)
model2 <- keras_model_sequential()
layer_dense(model2 , units = 16, activation = "relu", input_shape = c(max_featuresAir)) 
layer_dense(model2 , units = 16, activation = "relu")
layer_dense(model2, units=nlevels(y), activation = "softmax") # this is for dependent variable

compile(model2,  optimizer = "rmsprop",  loss="sparse_categorical_crossentropy",
        metrics = c("accuracy"))

history2 <- fit(model2,  x_trainAir2, y, epochs = 20, batch_size = 10)
history2 

# let's predict the test-set
textAirtest <- xtest$text
# Note that we use the same tokenizer we created above (i.e., vectorizerAir) to ensure
# that the same features included in the training-set are also included in the test-set
x_testAir2 <- vectorizerAir(textAirtest) 
# here you see that we have once again 1,000 columns 
dim(x_testAir2)
dim(x_trainAir2)

# Now, we generate predictions
classes2 <- apply(predict(model2, x_testAir2), 1, which.max) # classification
table(classes2)
