rm(list=ls(all=TRUE))
# set your working directory  
#setwd("C:/Users/luigi/Dropbox/TOPIC MODEL")
getwd()

library(readtext)
library(quanteda)
library(ggplot2)
library(quanteda.textstats)
library(quanteda.textplots )
library(SnowballC)
library(corrplot)
library(DT)
library(dplyr)
library(reshape2)

#########################################################################
#########################################################################
# Creating and Working with a Corpus
#########################################################################
#########################################################################

# There are several different ways to create a corpus in Quanteda. 
# Let's look at 2 of them: single (i.e., pre-formatted files) vs. multiple files

#########################################################################
# FIRST: you have already a matrix file with a text for each row (such as .csv or .xls) - 
# i.e., you have already a pre-formatted file that come in a “spreadsheet format” where 
# one column contains the text and additional columns might store document-level variables
# (e.g. author, or language)
#########################################################################

# This dataset is a sample of 100 tweet from Boston area discussing about food
# Data have been collected through Twitter API (when it was still possible to do that...)
# also specifying language and origin of tweets 

x <- read.csv("Input Data/Day 1/boston.csv", stringsAsFactors=FALSE)
glimpse(x)

# if you have problems to open this csv file, plz use the below rds file
#x <- readRDS("Input Data/Day 1/boston.rds")
glimpse(x)
# Rather than using the standard "read.csv" function as above, you could import a .csv file 
# via the function "readtext". This can give you more flexibility (as we will see below).
# On top of that, readtext allows you to import several different types of files 
# (.csv,.html,.txt,.xls,.pdf,.doc).
# When using readtext, you have however to specify the name of the column in the dataset 
# that includes the texts (in the present case: "text"). 
# If you already have in your dataset a column named "text", as in this case, you can 
# avoid to specify it in the command below. Having said that, my suggestion is the following one: 
# a) when you have to open a csv file, go with the usual read.csv; 
# b) for all the other types of files, go with readtext!

#myText2 <- readtext("Input Data/Day 1/boston.rds", text_field = "text")
myText2 <- readtext("Input Data/Day 1/boston.csv", text_field = "text")
# Note that a new column "doc_id" (i.e., an index of texts) has been automatically created.
glimpse(myText2)

# You then can create your corpus via the function "corpus"
myCorpus2 <- corpus(myText2)

# Jargon: types=number of unique terms; tokens=number of words 
head(summary(myCorpus2))

# number of documents in the corpus
ndoc(myCorpus2 )
# print the first text
as.character(myCorpus2)[1]
# same thing but w/o interruption of the text
strwrap(as.character(myCorpus2)[1])

# Let's move from the corpus to the document-feature-matrix! 
# In Quanteda, we first tokenize the texts via the function "tokens". 
tokens(myCorpus2)

tok2 <- tokens(myCorpus2)
tok2

# Then we use the function "dfm" to produce a dfm.
# When you create a dfm by default "tolower=TRUE", i.e., we convert all features 
# to lowercase
myDfm <- dfm(tok2)

# We can get the number of documents and features ndoc() and nfeat() that build our DfM
ndoc(myDfm)
nfeat(myDfm)

# We can also obtain the name of documents and features by docnames() and featnames()
str(myDfm)
head(docnames(myDfm), 20)
head(featnames(myDfm), 20)

# Let's see the first five documents and the first 10 words of our dfm
myDfm[1:5, 1:10]

# To find the document-level variables attached to each document and stored in our dfm:
str(myDfm@docvars)
head(docvars(myDfm))

# 20 top features in the dfm
topfeatures(myDfm , 20) 

# let's improve the dfm! 
# FIRST: let's remove numbers, separators,  etc. Note that I decided also 
# to remove the URLs
tok2_clean  <- tokens(myCorpus2, remove_punct = TRUE, remove_numbers=TRUE, 
                      remove_symbols = TRUE, 
                      split_hyphens = TRUE, remove_separators = TRUE, remove_url=TRUE)

# SECOND: let's remove the stopwords
head(stopwords("english"), 20)
head(stopwords("russian"), 10)
head(stopwords("italian"), 10)

# The stopwords options available in Quanteda (based on the Snowball stopwords list: 
# see http://snowball.tartarus.org/) works with all the main European languages
# (ftp://cran.r-project.org/pub/R/web/packages/stopwords/stopwords.pdf)
getStemLanguages()

# For other languages, things are a bit more complex. The source "marino" for Asian 
# languages (Japanese, Chinese, Korean; but also Arabic and Hebrew)
# can be a good option. See: https://github.com/koheiw/marimo
stopwords("en")
stopwords("en", source = "marimo")
# For Arabic ones, for example, a good source is also the package arabicStemR.
# If you are interested in dealing with Japanese/Chinese texts, just drop me an email!

tok2_clean <- tokens_remove(tok2_clean , stopwords("english"))
# Note that via "tokens_remove" you can identify any list of tokens you want to remove

# THIRD: let's stem the words
tok2_clean<- tokens_wordstem (tok2_clean, language =("english"))

# Let's now re-create the dfm
myDfm2 <- dfm(tok2_clean)
nfeat(myDfm)
nfeat(myDfm2)

topfeatures(myDfm , 50) 
topfeatures(myDfm2 , 20) # the features "for", "with" and "the" have disappeared! 
# while #smoked is now #smoke

# still some symbols to remove! for example the words starting with "00" 
# (they are unicode characters)
"\U00BD"

tok2_clean <- tokens_remove(tok2_clean, c(("rt"), ("00*"), ("ed"), ("u")))
myDfm2 <- dfm(tok2_clean )
topfeatures(myDfm2 , 20)  # 20 top features [better!]

# alternatively (exactly as we did above) you can first identify the list of tokens 
# you want to delete and then pass it to tokens_remove
x <- c(("rt"), ("00*"), ("ed"), ("u"))
str(x)
tok2_clean <- tokens_remove(tok2_clean, x)

# We can also create a dfm identifying only some specific words, such as for example 
# only the hashtags in the tweets using select = "#*" when creating the dfm
dfm_hashtag <- dfm_select(myDfm2, pattern = c("#*"))

topfeatures(myDfm2 , 20)  # 20 top features 
topfeatures(dfm_hashtag, 20)  # 20 top features 

# You can also decide to exclude some features. For example, let's exclude all the hashtags
dfm_NOhashtag <- dfm_remove(myDfm2, pattern = c("#*"))
topfeatures(myDfm2 , 20)  # 20 top features 
topfeatures(dfm_hashtag, 20)  # 20 top features 
topfeatures(dfm_NOhashtag , 20)  # 20 top features 

# Following the same logic, you can remove stopwords also with dfm_remove, 
# after you have created a dfm 
dfm_remove(myDfm, pattern = stopwords("en"))

# We can also trim the DfM.
# For example, let's keep only words occurring >= 10 times and in >= 2 documents
dfm_trim(myDfm, min_termfreq = 10, min_docfreq = 2)

# Let's keep only words occurring <= 10 times and in <=2 documents
dfm_trim(myDfm, max_termfreq = 10, max_docfreq = 2)

# Let's keep only words occurring in 4 of 10 of documents
dfm_trim(myDfm, min_docfreq = 0.4, termfreq_type = "prop")

# Let's keep only words occurring in all the 100 documents of my corpus at least once
dfm_trim(myDfm,  min_termfreq = 1, min_docfreq = 100)

# weighting a dfm according to the relative term frequency, i.e.,  
# normalizing a dfm by considering the proportions of the feature counts 
# within each document

myDfm_weight <- dfm_weight(myDfm, scheme = "prop")
# compare the two matrices below (the first one: unweighted; the second one: weighted)
myDfm[1:5, 1:5]
myDfm_weight [1:5, 1:5]

# weighting a dfm by tf-idf
# remember: tf-idf adds a weight that approaches zero as the number of documents in which
# a term appears (in any frequency) approaches the number of documents in the collection. 
# And indeed here #dinner close to 0! 
myDfm_tf <- dfm_tfidf(myDfm)
myDfm[1:5, 1:5]
myDfm_tf [1:5, 1:5]

#########################################################################
# SECOND: you have saved in a directory a set of files (one for each document) in a given 
# format (.txt, .doc, .pdf) - i.e., you have multiple text-files that are stored in 
# the same folder or subfolders and you want to load them 
#########################################################################

# SOURCE: http://www.presidency.ucsb.edu/inaugurals.php

# The txt files are included in the folder called "Inaugural Speeches" included in my case in 
# my working directory
#myText <- readtext("Input Data/Day 1/Inaugural Speeches/George_Washington_1789.txt")
glimpse(myText)

myText <- readtext("Input Data/Day 1/Inaugural Speeches/*.txt")
glimpse(myText)

# We can actually extract three pieces of info from each file name (Name, Surname, Year: 
# i.e., "George_Washington_1789.txt")
# Note: to use the docvarsfrom = "filenames" option, the "file names" should be consistent,
# i.e., in the below example, in ALL the txt title you should have the same ordering: 
# Name, Surname, Year using the same separators (i.e., "_")

# N.B. it is always a good idea to save your .txt files using the UTF-8 encoding when you are 
# analyzing texts written in English (or Japanese/Chinese for example) before reading 
# them in R. With other languages, UTF-8 is also fine. 
# Other options could be ISO-8859-1 (Latin-1) or Windows-1252 

myText <- readtext("Input Data/Day 1/Inaugural Speeches/*.txt", docvarsfrom = "filenames", 
                   dvsep = "_", docvarnames = c("Name", "Surname", "Year"))
glimpse(myText)
print(myText)

testCorpus <- corpus(myText )
summary(testCorpus)

# inspect the document-level variables
head(docvars(testCorpus))

# Let's now tokenize the texts
tok4 <- tokens(testCorpus,  remove_punct = TRUE, remove_numbers=TRUE, 
               remove_symbols = TRUE, split_hyphens = TRUE, remove_separators = TRUE)
# hyphen: the sign "-" used to join words
tok4 <- tokens_remove(tok4 , stopwords("en"))
tok4 <- tokens_wordstem (tok4 )
myDfm <- dfm(tok4)
topfeatures(myDfm , 20)  # 20 top words

# Note that you can also extract the topfeatures according to some document level variable,
# for example let's do it with respect to the Surname of the President (i.e., we have 
# two speeches for President in our dfm)
topfeatures(myDfm , 5, groups=Surname)  # 5 top words for each President

# we can also use some other document level variables such as Year in which a speech 
# was given
topfeatures(myDfm , 5, groups=Year)  # 5 top words for each Year

#########################################################################
# Playing with the corpus
#########################################################################

# let's load a corpus already presented in Quanteda: the corpus of all the US Presidents'
# Inaugural Speeches.
# To summarize the texts from a corpus, we can call a summary() method defined for a corpus
summary(data_corpus_inaugural)
# inspect the document-level variables 
head(docvars(data_corpus_inaugural))

# We can subset a corpus according to some document-level variable value via the 
# corpus_subset command. For example, let's extract only the Trump speech 
# (by taking advantage of the document-level variable "President" present in our corpus).
# Note: you can also subset a token according to some document-level variable 
# via the tokens_subset command

trump <- corpus_subset(data_corpus_inaugural, President == "Trump")
summary(trump)
# let's read the first Presidential inaugural speech made by Trump
strwrap(as.character(trump [[1]]))

# Let's extract the first five inaugural speeches 
mycorpus1<-  corpus_subset(data_corpus_inaugural, Year <1806)
summary(mycorpus1)

# We can also subset a corpus according to more than one single condition: note the use 
# of the logical operator "&"
summary(corpus_subset(data_corpus_inaugural, Year > 1990 & Party== "Republican"))

# The function corpus_reshape() allows to change the unit of texts between documents 
# and sentences
summary(data_corpus_inaugural)
ndoc(data_corpus_inaugural)

# This is going to be very important when we will discuss about WE!
corp_sent <- corpus_reshape(data_corpus_inaugural, to = "sentences")
ndoc(corp_sent )
print(corp_sent)

# You can use corpus_subset() for example keep only long sentences
# (more than 50 words for example)
corp_sent_long <- corpus_subset(corp_sent, ntoken(corp_sent) >= 50)
ndoc(corp_sent_long)
print(corp_sent_long)

#########################################################################
#  Let's explore some statistical summaries methods
#########################################################################

#########################################################################
# Statistical summaries (1): Plotting the wordclouds 
#########################################################################

# A tag cloud is a visual representation of text data, in which tags are single words 
# whose frequency is shown with different font size (and/or color)

myCorpus <- corpus_subset(data_corpus_inaugural, Year > 1990)
summary(myCorpus)
tok2 <- tokens(myCorpus, remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, 
               remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
myDfm <- dfm(tok2)

# if you define a seed, each time you get always the same plot
set.seed(123)
textplot_wordcloud(myDfm ,  min_count = 6, rotation  = .25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

textplot_wordcloud(myDfm , min_count = 10,
     color = c('red', 'pink', 'green', 'purple', 'orange', 'blue'))

# You can also plot a "comparison cloud", but this can only be done with fewer than eight
# documents. Let's plot for example a "comparison cloud" between Biden, Trump and Obama
corp2 <- corpus_subset(data_corpus_inaugural, President %in% c("Biden", "Trump", "Obama"))
summary(corp2)

# alternatively you can write:
corp3 <- corpus_subset(data_corpus_inaugural, President== "Biden" |  
                         President==   "Trump" |  President==  "Obama")
summary(corp3)

tok2 <- tokens(corp2, remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
myDfm <- dfm(tok2)
myDfm
# let's group the speeches made by the same President (i.e., the two speeches made 
# by Obama and Trump) in one single dfm using the function "dfm_group" (i.e., it sums the
# frequencies across rows beloning to the same President)
myDfm2 <- dfm_group(myDfm, groups = President)
myDfm2 

ndoc(myDfm)  # 5 documents before grouping
ndoc(myDfm2) # 3 documents after grouping (the two speeches by Obama & Trump are compressed into 1)

set.seed(123)
textplot_wordcloud(myDfm2, comparison = TRUE,  min_count = 5, 
                   color = c("blue", "green", "red"))

# how is that the feature "america" is shown just for Trump? After all "america" is typically 
# employed in all the US presidential speeches!

# A "comparison cloud" works like that: for each feature - 
# 1) it computes its relative frequency in each document
# 2) in the comparison wordcloud plot it will assign a feature to ONLY that document 
# that presents the maximum value in 1)

# we could also have used another document level variables such as Party in our previous
# subsample of 3 Presidential speeches
myDfm
myDfm2 <- dfm_group(myDfm, groups = Party)
myDfm2
set.seed(123)
textplot_wordcloud(dfm_trim(myDfm2 , min_termfreq = 5, verbose = FALSE), 
                   comparison = TRUE, color = c("blue", "red"))

#########################################################################
# Statistical summaries (2): Cosine similarity
#########################################################################

# As discussed in the class, "cosine similarity" is an intuitive measure of semantic similarity

# Let's create a dfm from inaugural addresses from Reagan onwards
myCorpus <- corpus_subset(data_corpus_inaugural, Year > 1980)
summary(myCorpus)
tok2 <- tokens(myCorpus, remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
tok2 <- tokens_wordstem (tok2)
presDfm<- dfm(tok2)
presDfm

# compute some document similarities
Simil <- textstat_simil(presDfm, method = "cosine")
Simil
# what do you notice? The most "dissimilar" speeches are the ones by Bush in 2005 
# (after the "Global War on Terror") and the two speeches made by Trump 
colMeans(Simil)

# Let's plot the previous results!
Simil2 <-as.matrix(Simil)
str(Simil2)
corrplot(Simil2, method = 'number')
corrplot(Simil2, method = 'color') 
corrplot(Simil2, method = 'shade', type = 'lower') 

# for specific comparisons: here the two speeches by Obama
obamaSimil <- textstat_simil(presDfm, presDfm[c("2009-Obama", "2013-Obama"), ], 
                             method = "cosine")
obamaSimil

#########################################################################
#  Statistical summaries (3): Checking for token-context and counting things (Dictionary approach)
#########################################################################

# The kwic function (keywords-in-context) performs a search for a word in a corpus and 
# it allows us to view the contexts in which it occurs.
# NOTE: by working on a token object, we retain the original text sequence!

options(width = 200)
kwic(tokens(data_corpus_inaugural), "terror")

# if you want to display the search in a cooler way:
x <- kwic(tokens(data_corpus_inaugural), "terror")
datatable(x, caption="Keywords in context", rownames=FALSE, 
          options = list(scrollX = TRUE,pageLength = 10, lengthMenu = c(5, 10, 15, 20)))

# with the window argument, we can specify the number of words to be displayed around 
# the keyword (here=1)
kwic(tokens(data_corpus_inaugural), "terror", window = 1)

# also the words starting with "terror" including "terrorism"
kwic(tokens(data_corpus_inaugural), "terror*")

# Note that by default, the kwic() is word-based. If you like to look up a multiword 
# combination, use phrase()
kwic(tokens(data_corpus_inaugural), phrase("by terror"))
kwic(tokens(data_corpus_inaugural), phrase("make america"))

# We can plot a kwic object via a Lexical dispersion plot.
# A Lexical dispersion plot allows you to detect both the relative frequency of an employed
# word across documents as well as the timing of that word in a given text

textplot_xray(
     kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1945 )), "people"))

textplot_xray(
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "america"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "people"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "business*")
)

# You might also have noticed that the x-axis scale is the absolute token index for single
# texts and relative token index when multiple texts are being compared. 
# If you prefer, you can specify that you want an absolute scale 

textplot_xray(
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "america"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "people"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "business*"),
     scale = 'absolute'
)

# The object returned is a ggplot object, which can be modified using ggplot

plot <- textplot_xray(
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "america"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "people"),
    kwic(tokens(corpus_subset(data_corpus_inaugural, Year > 1980)), "business*")
)
plot + aes(color = keyword) + scale_color_manual(values = c('red', 'blue', "green"))

# If you want simply to count how many times a list of words appear in a corpus, 
# you can create a dictionary and then apply it to your corpus.
# Let's create a dictionary with 3 entries via the dictionary function

myDict <- dictionary(list(terror = c("terror*", "threat"),
                          economy = c("job*", "business*", "grow", "work"),
   pop= c("people", "washington")))
myDict 

# Let's apply this dictionary to our usual US Presidential inaugural speeches corpus after 1991.
# When applying a dictionary it could be a good idea to avoid stemming if the words included in the
# dictionary are not stemmed as well (not our case though)
recentCorpus <- corpus_subset(data_corpus_inaugural, Year > 1991)
tok2 <- tokens(recentCorpus, remove_punct = TRUE, remove_numbers=TRUE, remove_symbols = TRUE, 
               split_hyphens = TRUE, remove_separators = TRUE)
tok2 <- tokens_remove(tok2, stopwords("en"))
Mydfm <- dfm(tok2)

# the function to apply a dictionary to a DfM is dfm_lookup 
byPresMat <- dfm_lookup(Mydfm  , dictionary = myDict)
byPresMat
Dictionary <-convert(byPresMat, to="data.frame")
head(Dictionary )

# Let's reshape the data-frame and let's plot it
str(Dictionary)
Dictionary_long<-melt(Dictionary,id.vars=c("doc_id"))
str(Dictionary_long)

# bar plot with flipped coords
ggplot(Dictionary_long, aes(x = doc_id, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  labs(title = "Terror, Economy, and Pop counts by Document",
       x = "Document ID",
       y = "Count") +
  theme_minimal()

# you can also apply a dictionary to a tokenized object via the token_lookup option
byPresMatToks <- tokens_lookup(tok2, myDict )
print(byPresMatToks)

# the advantage of tokens_lookup is that allows you to include in a dictionary multi-word expressions
# for example:
head(kwic(tokens(data_corpus_inaugural), phrase(c ("United States", "New York"))))

# let's define a dictionary with 2 multi-word expressions
multiword_dict <- dictionary(list(country = "United States", 
                                  city = "New York"))
multiword_dict 

toks <- tokens(data_corpus_inaugural)
Mydfm <- dfm(toks)
byPresMat <- dfm_lookup(Mydfm  , dictionary = multiword_dict)
byPresMat
Dictionary <-convert(byPresMat, to="data.frame")
# all 0s!
head(Dictionary )

# however with tokens_lookup...
head(tokens_lookup(toks, dictionary = multiword_dict))

# So how to deal with multi-word expressions in a DfM? You could first use tokens_compound() 
# to join elements of multi-word expressions by underscore, so they become United_States and New_York.

comp_toks <- tokens_compound(toks, pattern =phrase(c ("United States", "New York")))
head(tokens_select(comp_toks, pattern= c("United_States", "New_York")))

# and then define a new dictionary and a new dfm with the compounded tokens
multiword_dict2 <- dictionary(list(country = "United_States", 
                                  city = "New_York"))
multiword_dict2 
Mydfm <- dfm(comp_toks)
byPresMat <- dfm_lookup(Mydfm  , dictionary = multiword_dict2)
byPresMat
Dictionary <-convert(byPresMat, to="data.frame")
head(Dictionary)
# same result as above via tokens_lookup
head(tokens_lookup(toks, dictionary = multiword_dict))

tail(Dictionary)
tail(tokens_lookup(toks, dictionary = multiword_dict))

# if you want to know more things about the dictionary approach in a BoW framework (for example
# how to use it to detect the stance/sentiment - more negative or positive - in a given
# corpus) drop me an email!

