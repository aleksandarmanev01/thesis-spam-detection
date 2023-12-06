import re
from torchtext.data import get_tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def preprocess_text_lstm(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    words = tokenizer(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
