import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import chi2, mutual_info_classif, VarianceThreshold
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from scipy.sparse import *
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from scipy.optimize import minimize

from IPython.display import clear_output


data_train = pd.read_csv('/Users/ludoviclepic/.cache/kagglehub/datasets/uciml/human-activity-recognition-with-smartphones/versions/2/train.csv')
data_test = pd.read_csv('/Users/ludoviclepic/.cache/kagglehub/datasets/uciml/human-activity-recognition-with-smartphones/versions/2/test.csv')
# Split features and target for both train and test
X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1]
y_test = data_test.iloc[:, -1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def construct_W(X, **kwargs):
    """
    Construct the affinity matrix W through different ways
    """
    # ...existing docstring...

    # Set default parameters
    kwargs.setdefault('metric', 'cosine')
    kwargs.setdefault('neighbor_mode', 'knn')
    kwargs.setdefault('k', 5)
    kwargs.setdefault('weight_mode', 'binary')
    kwargs.setdefault('fisher_score', False)

    if kwargs['neighbor_mode'] == 'supervised' and 'y' not in kwargs:
        print('Warning: label is required in the supervised neighborMode!!!')
        exit(0)

    n_samples, n_features = np.shape(X)

    # Handle fisher_score case first
    if kwargs['neighbor_mode'] == 'supervised' and kwargs['fisher_score']:
        y = kwargs['y']
        label = np.unique(y)
        n_classes = len(label)
        W = lil_matrix((n_samples, n_samples))
        for i in range(n_classes):
            class_idx = (y == label[i])
            class_idx_all = (class_idx[:, np.newaxis] & class_idx[np.newaxis, :])
            W[class_idx_all] = 1.0/np.sum(np.sum(class_idx))
        return W

    # Prepare data based on metric
    if kwargs['metric'] == 'cosine':
        # Normalize data for cosine metric
        X_normalized = np.power(np.sum(X*X, axis=1), 0.5)
        X = X / np.maximum(1e-12, X_normalized[:, np.newaxis])
        D = np.dot(X, X.T)
        D = -D  # Convert to ascending order
    else:  # euclidean
        D = pairwise_distances(X)
        D **= 2

    # Sort distances and get indices
    k = kwargs['k']
    idx = np.argsort(D, axis=1)
    idx_new = idx[:, 0:k+1]

    # Compute weights based on weight_mode
    if kwargs['weight_mode'] == 'heat_kernel':
        t = kwargs['t']
        dump_new = np.exp(-D[np.arange(n_samples)[:, None], idx_new]/(2*t*t))
    elif kwargs['weight_mode'] == 'cosine':
        dump_new = -D[np.arange(n_samples)[:, None], idx_new]
    else:  # binary
        dump_new = np.ones((n_samples, k+1))

    # Build the sparse affinity matrix
    G = np.zeros((n_samples*(k+1), 3))
    G[:, 0] = np.repeat(np.arange(n_samples), k+1)
    G[:, 1] = idx_new.ravel()
    G[:, 2] = dump_new.ravel()
    
    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    
    return W


def lap_score(X, **kwargs):
    """
    This function implements the laplacian score feature selection, steps are as follows:
    1. Construct the affinity matrix W if it is not specified
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples)
            input affinity matrix

    Output
    ------
    score: {numpy array}, shape (n_features,)
        laplacian score for each feature

    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    """

    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
        W = construct_W(X)
    # construct the affinity matrix W
    W = kwargs['W']
    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    return np.transpose(score)


def feature_ranking(score):
    """
    Rank features in ascending order according to their laplacian scores, the smaller the laplacian score is, the more
    important the feature is
    """
    idx = np.argsort(score, 0)
    return idx

def ranking_features_lap(X_scaled):
    W = construct_W(X_scaled)  # Affinity matrix
    laplacian_scores = lap_score(X_scaled, W=W)
    ranked_features = feature_ranking(laplacian_scores)
    return ranked_features

ranked_features = ranking_features_lap(X_scaled)

def fisher_score(X, y):
    """
    This function implements the fisher score feature selection, steps are as follows:
    1. Construct the affinity matrix W in fisher score way
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    score: {numpy array}, shape (n_features,)
        fisher score for each feature

    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
    """

    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
    W = construct_W(X, **kwargs)

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0/lap_score - 1
    return np.transpose(score)


def feature_ranking(score):
    """
    Rank features in descending order according to fisher score, the larger the fisher score, the more important the
    feature is
    """
    idx = np.argsort(score, 0)
    return idx[::-1]

# Fisher Score Pipeline
# Compute Fisher Scores for the dataset
fisher_scores = fisher_score(X_scaled, y_encoded)
ranked_features_fisher = feature_ranking(fisher_scores)

print(ranked_features_fisher, ranked_features)