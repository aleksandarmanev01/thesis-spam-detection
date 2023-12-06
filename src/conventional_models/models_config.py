from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Model definitions
models = {
    'NB': MultinomialNB(),
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel="sigmoid", gamma=1.0),
    'XGBoost': XGBClassifier(learning_rate=0.01, n_estimators=150),
    'LightGBM': LGBMClassifier(learning_rate=0.1, num_leaves=20, verbose=-1)
}

# Maximum features for each model
max_features_dict = {
    'NB': 1000,
    'LR': 500,
    'KNN': 150,
    'SVM': 3000,
    'XGBoost': 2000,
    'LightGBM': 3000
}
