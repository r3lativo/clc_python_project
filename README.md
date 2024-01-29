
# Sentiment Classifier of Movie Reviews
This program consists of a classifier that is able to decide whether a movie review is good or bad. The classifier is trained on an IMDB movie review dataset of 50k reviews. The program also spits out the accuracy and precision of the classifier.

## Dependencies
- Numpy
- Pandas
- NLTK
- BeautifulSoup
- SciKit Learn
- Matplotlib

## Data
The data we used is taken from [Kaggle - IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/atulanandjha/imdb-50k-movie-reviews-test-your-bert/), which was in turn taken from a Stanford dataset (look to References for more).  
In particular, `merged_data.csv` is the the combination of the 'train' and 'test' files into one big file. We decided to merge them to be able to manage the proportion of the train and test set.  
Instead, `toy_data.csv` is just a smaller version of the data.

## Getting started
First of all, the required libraries have to be installed on the local machine to be able to run the program.
With pip, it can be done by 
`$ pip install -r requirements.txt`.

## Usage
The program is stored as `sentiment_classifier.py`, so running it with python is all you need to do.

## About the project
We decided to use NLTK because, being a toolkit released in 2001, it can give us insights in how things changed through time, to be able to compare it to the newer models that other courses will delve into.

The classifier training has various steps:
1. Loading and cleaning the data
2. Extract features from the reviews
3. Feed the features to the classifier
4. Analyze its accuracy

The most fundamental step was the Feature Extraction, as it required us to define what is important from the text and what is not.

We also decided to create features with the Bag of Words method: we do not have a specific list of words we are looking for, but we decided to look into various grammatical categories (Nouns, Verbs, Adjectives, Adverbs and Numbers) and lemmatize them (reduce them to their dictionary form) to have a manageable size of the features.

## References
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). _The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)_.
