import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK datasets
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
english_stopwords = set(stopwords.words("english"))


def tokenize_words(text):
    """
    Tokenize the input text and return alphanumeric tokens.
    """
    tokens = word_tokenize(str(text).lower())
    return [token for token in tokens if token.isalnum()]


def remove_stopwords(tokens):
    """
    Remove stopwords from a list of tokens.
    """
    return [token for token in tokens if token not in english_stopwords]


def stem(tokens):
    """
    Stem a list of tokens using Porter Stemming.
    """
    return [ps.stem(token) for token in tokens]


def preprocess_text_baseline(text):
    """
    Process the input text: Tokenize, remove stopwords, and stem.
    """
    return " ".join(stem(remove_stopwords(tokenize_words(text))))
