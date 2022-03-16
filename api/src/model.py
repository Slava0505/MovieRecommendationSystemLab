import logging
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import os

from datetime import datetime
from imp import reload
reload(logging)
logging.basicConfig(filename='app.log',level = logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
class BaseRecommendationModel():
    """Baseline model that of the form mu + b_u + b_i,
    where mu is the overall average, b_u is a damped user
    average rating residual, and b_i is a damped item (movie)
    average rating residual. See eqn 2.1 of
    http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf

    Attributes
    ----------
    mu : float
        Average rating over all training samples
    b_u : pandas Series, shape = [n_users]
        User residuals
    b_i : pandas Series, shape = [n_movies]
        Movie residuals
    """

    def fit(self, X):
        """Fit training data.

        Parameters
        ----------
        X : DataFrame, shape = [n_samples, >=3]
            User, movie, rating dataFrame. Columns beyond 3 are ignored

        Returns
        -------
        self : object
        """
        X = X.iloc[:, :3].copy()
        X.columns = ['user', 'item', 'rating']
        self.mu = np.mean(X['rating'])
        user_counts = X['user'].value_counts()
        movie_counts = X['item'].value_counts()
        b_u = (
            X[['user', 'rating']]
            .groupby('user')['rating']
            .sum()
            .subtract(user_counts * self.mu)
            .divide(user_counts)
            .rename('b_u')
        )
        X = X.join(b_u, on='user')
        X['item_residual'] = X['rating'] - X['b_u'] - self.mu
        b_i = (
            X[['item', 'item_residual']]
            .groupby('item')['item_residual']
            .sum()
            .divide(movie_counts)
            .rename('b_i')
        )
        self.b_u = b_u
        self.b_i = b_i
        return self

    def predict(self, user_id, m=5):
        item_pred_rating = self.mu + self.b_u[user_id] + self.b_i
        top_m_items = list(item_pred_rating.sort_values(ascending =False).iloc[:m].index())
        return top_m_items

    def train(self, filepath = 'data/ratings_train.dat'):
        ratings_df = pd.read_csv(filepath, sep='::', header=None, 
                                names=['userId', 'movieId', 'rating', 'timestamp'])
        ratings_df['timestamp'] = ratings_df['timestamp'].apply(datetime.fromtimestamp)
        ratings_df = ratings_df.sort_values('timestamp')
        self.fit(ratings_df)
        filename = 'model/{}.sav'.format(filepath.split('.')[0].split('/')[-1])
        pickle.dump(model, open(filename, 'wb'))
        logging.info("The model was trained and saved in a file {}".format(filename))

    def evaluate(self, modelpath='model/ratings_train.sav',  filepath = 'data/ratings_test.dat'):

        ratings_df = pd.read_csv(filepath, sep='::', header=None, 
                                names=['userId', 'movieId', 'rating', 'timestamp'])
        ratings_df['timestamp'] = ratings_df['timestamp'].apply(datetime.fromtimestamp)
        ratings_df = ratings_df.sort_values('timestamp')

        self.warmup(modelpath=modelpath)

        pred = self.predict_df(ratings_df)
        rmse = mean_squared_error(pred, ratings_df['rating'])**0.5

        print("RMSE on eval set from {} = {}".format(filepath, rmse))
        logging.info("RMSE on eval set from {} = {}".format(filepath, rmse))

    def warmup(self, modelpath='model/ratings_train.sav'):
        model = pickle.load(open(modelpath, 'rb'))
        self.b_u = model.b_u      
        self.b_i = model.b_i
        self.mu = model.mu
        logging.info("Model loaded from {}".format(modelpath))

    def predict_df(self, X):
        """Return rating predictions

        Parameters
        ----------
        X : DataFrame, shape = (n_ratings, 2)
            User, item dataframe

        Returns
        -------
        y_pred : numpy array, shape = (n_ratings,)
            Array of n_samples rating predictions
        """
        X = X.iloc[:, :2].copy()
        X.columns = ['user', 'item']
        X = X.join(self.b_u, on='user').fillna(0)
        X = X.join(self.b_i, on='item').fillna(0)
        return (self.mu + X['b_u'] + X['b_i']).values