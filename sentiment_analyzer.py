########## IMPORT AND DOWNLOAD ##########

"""Import the libraries needed"""
import pandas as pd             # to manipulate DataFrames
import numpy as np              # library to do math
import re                       # RegEx, regular expressions
import nltk                     # Natural Language Tool Kit
from bs4 import BeautifulSoup   # to clean html text
import matplotlib               # not explicitly called but needed to plot the confusion matrix graph

# these are the same as just importing 'nltk' and 'sklearn', but allow an easier access to some functions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# to suppress a warning related to Beautiful Soup
import warnings
from bs4 import MarkupResemblesLocatorWarning

"""Download stuff"""
# these downloads are necessary to support the work of 'nltk'
nltk.download('stopwords')                    # needed to remove stopwords
nltk.download('punkt')                        # Needed for tokenizing
nltk.download('wordnet')                      # Needed for lemmatizing
nltk.download('averaged_perceptron_tagger')   # Needed for POS Tagging

"""Set pandas option"""
pd.options.mode.copy_on_write = True
# Copy-on-write makes operations return copies instead of views
# it is advised and will be the default in pandas
# see: https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html


########## DEFINE FUNCTIONS ##########

# we create a function that tokenizes each review,
# chooses only the words that we deem relevant according to their POS
# and lemmatizes them

def get_relevant_lemmas(text):
	"""
	This function takes as input the text of a review, and gives as output
	a list with only NOUNS, VERBS, ADJECTIVES, ADVERBS and NUMERALS/CARDINALS.
	:param text: the ext of the review
	:type text: string
	:return lemma_list: the relevant lemmas that we will use in the embeddings
	:rtype: list
	"""

	# tokenize the text to get a list of tokens
	tokens = word_tokenize(text)

	# remove stopwords from the list of tokens
	stop_words = set(stopwords.words('english'))  # We set the language of the stopwords to english
	# for each t in tokens, we check whether its lower case version is in the stop words or not
	tokens = [t for t in tokens if t.lower() not in stop_words]

	# give to each token its POS tag
	# this creates a list of tuples, like [("token", "tag"), ...]
	tagged = nltk.pos_tag(tokens)
	# print(tagged)

	# we will use RegEx to only select the relevant POS, and to connect NLTK terms to WordNetLemmatizer terms
	# we take into account nouns, verbs, adjectives, adverbs, and numbers
	# moreover, re.search it takes into account the various form each input (in this case, the POS) could have
	# for example, in RB we could find: RBR which is a comparative adverb; RBS which is a superlative adverb, etc.
	# we create a dictionary {"NLTK POS tag" : "WordNet POS tag", ...}
	# to relate both types of tags
	relevant_POS = {"NN":"n", "VB":"v", "JJ":"a", "RB":"r"}

	# we create an empty list to store the relevant tokens of the review
	lemma_list = []

	# name the lemmatizer
	lemmatizer = WordNetLemmatizer()

	# we check what kind of tag each token has, and only store the one we want
	for token, pos in tagged:
	# we take care of numbers (whose NLTK POS tag is CD) differently, as we do not want to lemmatize them
		if re.search("CD", pos):
			lemma_list.append(token)

		# we perform a RegEx search which will return all the elements with the relevant POS tag
		for NLTK_tag in relevant_POS.keys(): # we look for the relevant NLTK POS tags
			if re.search(NLTK_tag, pos):
				WordNet_tag = relevant_POS.get(NLTK_tag) # we take the corresponding WordNet POS tag
				token = lemmatizer.lemmatize(token, str(WordNet_tag)) # we lemmatize each token as its POS
				lemma_list.append(token) # and then we append the lemma to the list

	return lemma_list


# now that the review text are pretty clean, we have to go and actually build the feature extractor
# note that the previous function has only been tested on specific reviews,
# and we still have to apply it to each review in the DataFrame

def feature_extractor(text):
	"""
	This function takes as input the raw review, and gives as output
	a vector in form of a dictionary which takes into account how many times each word
	appears in that given string of text.

	An example would be:
	"this movie was really really beautiful"
	{"movie": 1, "be": 1, "really": 2, "beautiful": 1}

	:param raw_review: the text of the review
	:type raw_review: string
	:return feature_vector: a vector {"token":count, ...}
	:rtype: dict
	"""

	# first apply the get_relevant_lemmas function we prepared above
	review_lemmas = get_relevant_lemmas(text)

	# prepare the dictionary
	feature_vector = {}

	# for each relevant token in the review
	for token in review_lemmas:
		# check if the token is already in the 'feature_vector' dictionary
		if token in feature_vector:
			# if the token is already in the dictionary, increment its frequency by 1
			feature_vector[token] += 1
		else:
			# if the word is not in the dictionary, add it to the dictionary with a frequency of 1
			feature_vector[token] = 1

	return feature_vector


def create_feature_corpus(starting_dataframe):
	"""
	This function takes as input a starting dataframe, and transforms it into a list.
	In the list, each review is a tuple: ({dictionary of words: count}, sentiment associated)
	:param starting_dataframe: dataframe review, sentiment
	:type starting_dataframe: DataFrame
	:return feature_corpus: a list of tuples [(feature_vector, sentiment), ...]
	:rtype: list
	"""

	# prepare the empty list to store the corpus
	feature_corpus = []

	# for each row in the dataframe:
	for index, row in starting_dataframe.iterrows():
		# create a vector by applying the feature extractor to each review
		# each row is made of a "sentiment" and a "review", so we have to access the actual review with row["review"]
		vector = feature_extractor(row["review"])
		# append the tuple (vector, sentiment) to the list
		feature_corpus.append((vector, row["sentiment"]))

	return feature_corpus


# Divide the dataframe into different subsets (train, test, evaluation)

def divide_corpus(corpus):
	"""
	:param corpus: either the starting corpus or the feature corpus
	:type corpus: DataFrame OR list
	:return training_part: 80% of the corpus, will be used for training
	:return evaluation_part: 20% of the corpus, will be used for evaluation
	:rtype: DataFrame OR list, same as :type corpus:
	"""

	# the limit is where the corpus will be split
	# 0.80 here means the 80% of the corpus
	limit = int(len(corpus) * 0.80)

	training_part = corpus[:limit]
	evaluation_part = corpus[limit:]

	return training_part, evaluation_part

########## START WORKING ##########

"""Load the data"""
# with the parameter quotechar='"' we explicitly say taht the character " has to be treated specially
# instead, the parameter escapechar='\\' we say that the (special) character \ is used to escape other characters
dataframe = pd.read_csv("./data/toy_data.csv", delimiter=",", escapechar="\\")
print("DataFrame loaded!")


"""Randomize the dataframe"""
# the function "sample" returns a random sample of items
# with the "frac = 1" we make sure that the random sample is actually the whole dataframe (the whole fraction)
dataframe = dataframe.sample(frac=1).reset_index(drop=True)
print("DataFrame randomized!")


# to remove these expressions we use BeautifulSoup, a Python library for pulling data out of HTML files
# the function "BeautifulSoup(HTML_TEXT).get_text()" will give back the text without html expressions
# there is a warning that tells us that it does not resemble an html file, but we can confidently suppress it
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)  # Suppress bs4 warning
dataframe.review = dataframe.review.apply(lambda raw_text: BeautifulSoup(raw_text, features="html.parser").get_text(separator=" "))
print("HTML tags removed!")


# we divide the initial dataframe into training and evaluation
dataframe_train, dataframe_eval = divide_corpus(dataframe)
print("DataFrame divided into training and evaluation!")


# we create the feature corpus
print("Creating feature corpus...(This process can take a while)")
feature_corpus = create_feature_corpus(dataframe)
print("Feature corpus created!")


# we divide the feature corpus into training and evaluation
feature_train, feature_eval = divide_corpus(feature_corpus)
print("Feature corpus divided into training and evaluation!")


# we train the Naive Bayes classifier in NLTK using the training part of the feature corpus
print("Training the Naive Bayes classifier...")
classifier = nltk.NaiveBayesClassifier.train(feature_train)
print("Training completed!")

########## RESULTS ##########

# we check the most informative features of this classifier
print("\nLet's take a look at the results of this classifier!\n")
print("The 10 most informative features to the classifier are:")
classifier.show_most_informative_features()  # print() not needed as this already prints
print()


# we add a column to the evaluation dataframe which specifies the sentiment given by the classifier
dataframe_eval["classifier_sentiment"] = dataframe_eval.review.apply(lambda row: classifier.classify(feature_extractor(row)))


# we add another column to the dataframe which tells us if the evaluation was correct (True) or not (False)
comparison_column = np.where(dataframe_eval["sentiment"] == dataframe_eval["classifier_sentiment"], "Correct", "Incorrect")
dataframe_eval["correct_evaluation"] = comparison_column


# we check how many tags were correct
print("Check how many reviews were correctly classified:")
print(dataframe_eval.correct_evaluation.value_counts().to_string())  # the 'to_string()' eliminates the name and the type of the data
print()


# we get the accuracy using sklearn
accuracy = accuracy_score(dataframe_eval["sentiment"], dataframe_eval['classifier_sentiment'])
print(f"The accuracy of the classifier is {accuracy}!")


# generate confusion matrix for the predictions
conf_matrix = confusion_matrix(dataframe_eval["sentiment"], dataframe_eval["classifier_sentiment"], labels=["pos","neg"])
print("The confusion matrix of the classifier is:")
print(f"{conf_matrix}\n")

print("Exporting confusion_matrix.png to the current folder...")
matrix_plot = ConfusionMatrixDisplay(conf_matrix, display_labels=["pos","neg"])
plot = matrix_plot.plot()
matrix_plot.figure_.savefig("confusion_matrix.png")
print("confusion_matrix.png exported!\n")


# we get a classification report for our classifier
print("And these are the evaluation metrics from our classifier:")
report = classification_report(dataframe_eval["sentiment"], dataframe_eval["classifier_sentiment"])
print(report)
