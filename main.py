"""Import the libraries needed"""
import pandas as pd             # To manipulate DataFrames
import pyarrow                  # Mandatory with pandas
import numpy as np              # Library to do math
import re                       # RegEx, regular expression
import nltk                     # Natural Language Tool Kit
from bs4 import BeautifulSoup   # To clean html text

# These are the same as just importing 'nltk', but allow an easier access to some functions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


"""Download stuff"""
# These downloads are necessary to support the work of 'nltk'
#nltk.download('stopwords')
#nltk.download('punkt')                        # Needed for tokenizing
#nltk.download('wordnet')                      # Needed for lemmatizing
#nltk.download('averaged_perceptron_tagger')   # Needed for POS Tagging


"""Load the dataframe"""
# The dataframe is taken from:
# https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert/
# as part of a Keggle competition

# The data we have is of type 'csv', which means 'comma separated files'.

# The file will be loaded with Pandas:
# Pandas is a library that uses DataFrame type files
# These DataFrames can be seen as an excel file, with columns (Series) and rows (index, i.e. the actual data)
# The 'data' in the index can be of various kind: (integers, strings, floating point numbers, Python objects, etc.).

# The parameter "delimiter=','" takes into account that the file is a csv
# Moreover, with the parameter quotechar='"' we explicitly say taht the character " has to be treated specially
# Finally, the parameter escapechar='\\' we say that the character \ is used to escape other characters
dataframe = pd.read_csv("./data/train.csv", delimiter=",", escapechar="\\")


"""Randomize the dataframe"""
# The function "sample" returns a random sample of items
# With the "frac = 1" we make sure that the random sample is actually the whole dataframe (the whole fraction)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)

dataframe = dataframe.rename(columns={"text": "review"})

print(dataframe.columns)

print(dataframe.sentiment.value_counts())

"""Clean the dataframe"""
# We have to clean it: we use BeautifulSoup which does exactly this
# The function "BeautifulSoup(HTML_DIRTY_TEXT).get_text()" will clean and get text removing html
dataframe.review = dataframe.review.apply(lambda raw_text: BeautifulSoup(raw_text, features="html.parser").get_text(separator=" "))


# Let's create a function that tokenizes and POS-tagges each review
def relevant_POS_tag(raw_review):
  """
  This function takes as input the raw text of a review, and gives as output
  a list with only NOUNS, VERBS, ADJECTIVES, ADVERBS and NUMERALS/CARDINALS.
  """

  # Tokenize the raw review
  # This gives back a LIST of tokens
  tokens = word_tokenize(raw_review)

  # Tag each word
  # This creates a list of touples, like [("Michael", "NN"), ("was", "VB"), ...]
  # where NN: Noun, VB: Verb, ...
  tagged = nltk.pos_tag(tokens)
  #print(tagged)

  # We will use RegEx to only select the relevant POS, and to connect NLTK terms to WordNetLemmatizer terms
  # We take into account "Nouns OR Verbs OR Adjectives OR Adverbs"
  # Moreover, re.search it takes into account the various form each input (in this case, the POS) could have
  # For example, in RB we could find: RBR which is a comparative adverb; RBS which is a superlative adverb...
  #relevant_POS = "NN|VB|JJ|RB|CD"
  relevant_POS = {"NN":"n", "VB":"v", "JJ":"a", "RB":"r"}  # Numbers (CD) are taken care without the Lemmatizer

  # We then create an empty list to store the filtered review
  # which will now be in fact a list of words
  filtered_rev = []

  # Load the Lemmatizer
  lemmatizer = WordNetLemmatizer()

  # Check what kind of tag each token has, and only stores the one you want
  for word, pos in tagged:

    # We take care of Numerals and Cardinals differently, as we do not want to Lemmatize them
    if re.search("CD", pos):
      filtered_rev.append(word)

    # We perform a RegEx search, which will return all the elements that have what we are looking for
    # Here, t[1] refers to the second item in the tuple: e.g. in ("Michael", "NN"), it refers to "NN"
    for p in relevant_POS.keys():
      if re.search(p, pos):
        x = relevant_POS.get(p)
        word = lemmatizer.lemmatize(word, str(x))
        filtered_rev.append(word)

  return filtered_rev

