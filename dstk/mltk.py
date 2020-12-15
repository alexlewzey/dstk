"""
machine learning tool-kit
a collection of functions for training, applying and evaluating machine learning algorithms. Mainly thin wrapper
for sci-kit learn"""
import itertools
import logging
from collections import OrderedDict
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from lightgbm import LGBMRegressor, LGBMClassifier
from plotly.offline import plot
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

# from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)

TreeModel = Union[
    RandomForestClassifier,
    RandomForestRegressor,
    # XGBRegressor,
    # XGBClassifier,
    LGBMRegressor,
    LGBMClassifier,
]


def split_df(df: pd.DataFrame, flag_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data frame on sequence of booleons returning the subset that matches true first"""
    return df[df[flag_col]].reset_index(drop=True), df[~df[flag_col]].reset_index(drop=True)
    # return df[df[flag_col]], df[~df[flag_col]]


def log_sample(n: int = 1) -> float:
    """sample evenly from a log distribution eg if n == 2
    (1, 0.1)=50% of obvs (0.1, 0.01)=50% of obvs"""
    return 10 ** (-n * np.random.rand())


# metrics ##############################################################################################################

def mean_absolute_percentage_error(y_true, y_pred):
    """mean absolute percentage error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def root_mean_squared_error(y_true, y_pred) -> float:
    """numpy implementation of root mean squared error"""
    return np.sqrt(((y_true - y_pred) ** 2).mean())


# preprocessing ########################################################################################################


def apply2features(df: pd.DataFrame, features: List, processor: Callable) -> pd.DataFrame:
    not_features = [col for col in df.columns if col not in features]
    return processor(df.set_index(not_features)).reset_index()


def standardise(df: pd.DataFrame, features: Optional[List] = None) -> pd.DataFrame:
    """return a df where every columns is standardised, or pass the specific feature list"""
    processor = lambda x: pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)
    if features:
        return apply2features(df, features, processor)
    return processor(df)


# cross validation / optimisers ########################################################################################

params_svm = [
    {'C': [0.1, 1, 100, 1000], 'kernel': ['linear']},
    {'C': [0.1, 1, 100, 1000], 'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 0.001]},
]
params_rf = {'n_estimators': [1, 3, 10, 30, 100, 300]}

params_xgb = {
    "learning_rate": [0.01, 0.10, 0.20],
    'max_depth': [3, 4, 5, 8, 10],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.1, 0.5, 1, 1.5, 2, 5],
    'colsample_bytree': [0.3, 0.6, 0.8, 1.0],
    'subsample': [0.5, 0.75, 1.0],
}

param_lgbm = {
    'learning_rate': [0.01, 0.10, 0.20],
    'num_leaves': [31, 127],
    'max_depth': [3, 10],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'reg_alpha': [0.1, 0.5],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
}

specs = {
    'log': (LogisticRegression, {'C': [1, 0.01, 0.001, 0.0001]}),
    'knn': (KNeighborsClassifier, {'n_neighbors': [1, 6, 16, 32, 56, 88]}),
    'svc': (SVC, {'C': [1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}),
    'rand_forest': (RandomForestClassifier, {'n_estimators': [1, 5, 10, 30, 50, 100, 200, 300]})
}


def cross_val_regression(est, x, y, params: List[Dict], kfold=10):
    """
    get avg rmse and r2 scores for a regression estimator using k-fold cross validation
    """
    metrics = []
    scoring = {'mse': 'neg_mean_squared_error'}
    for param in tqdm(params):
        cv = cross_validate(est(**param), x=x, y=y, scoring=scoring, cv=kfold, n_jobs=-1)
        pos_cv = np.sqrt(cv['test_mse'] * -1)
        metrics.append(OrderedDict([
            ('params', str(param)),
            ('avg_mse', pos_cv.mean()),
            ('std_mse', pos_cv.std()),
            ('std_y', np.std(y)), ]
        ))
    return metrics


def feature_importance(est: TreeModel, x: pd.DataFrame) -> pd.DataFrame:
    """return a DataFrame of features and their importance"""
    return (pd.DataFrame({'features': x.columns, 'importance': est.feature_importances_})
            .sort_values('importance', ascending=False))


def plot_feature_importance(est: TreeModel, x: pd.DataFrame, path: Optional[Path] = None) -> go.Figure:
    """Plot the feature importance of the model variables as a bar chart with plotly"""
    importance = feature_importance(est, x)
    fig = px.bar(importance.sort_values('importance'), x='importance', y='features', orientation='h',
                 title='feature_importance')
    plot(fig, filename=path.as_posix()) if path else fig.plot()
    return fig


def sk_cvres2df(grid: GridSearchCV) -> pd.DataFrame:
    """transform grid search object into DataFrame of cv_results, unpacking the parameters"""
    cv_results = pd.DataFrame(grid.cv_results_)
    return pd.concat([cv_results.drop('params', axis=1), pd.json_normalize(cv_results['params'])], axis=1)


def plot_partial_dependence_bigger(estimator, X, features, size: float = 15, *args, **kwargs) -> None:
    """Plot partial dependence with ability to specify plot size"""
    pdp = plot_partial_dependence(estimator, X, features, *args, **kwargs)
    fig = pdp.figure_
    fig.set_figwidth(size)
    fig.set_figheight(size)
    fig.tight_layout()
    plt.show()


# baysiean optimisation ################################################################################################

def bayesopt_res2df(optimizer) -> pd.DataFrame:
    """transform baysain optimiser class into a DataFrame of results"""
    df = pd.DataFrame(optimizer.res)
    return pd.concat([df.drop('params', axis=1), pd.json_normalize(df['params'])], axis=1)


# clustering and nearest neighbours ####################################################################################


def repeating_rng(rng, n_repeat): return list(range(rng)) * n_repeat


def add_trial_to_idx_column(neighbors: Tuple[np.ndarray, np.ndarray], trial_idx: pd.Series):
    """add the idx of the trial and the distance (distance to its self is 0) to the sklearn knn
    output so it can be transformed into a tidy format with neighbors"""
    dist, idxs = neighbors
    dist_trial = np.zeros(dist.shape[0]).reshape(-1, 1)
    trial_idx = trial_idx.values.reshape(-1, 1)
    return np.hstack([dist_trial, dist]), np.hstack([trial_idx, idxs])


def neighbours2df(neighbors: Tuple[np.ndarray, np.ndarray], trial_idx: pd.Series, ctrl_idx: pd.Series) -> pd.DataFrame:
    """Transform output of sklearn nearest neighbours (with distance) into a tidy DataFrame format."""
    neighbors = add_trial_to_idx_column(neighbors, trial_idx)
    df = np.stack([arr.ravel() for arr in neighbors], 1)
    df = pd.DataFrame(df, columns=['distance', f'idx_ctrl'])
    n_trial, n_ctrl = neighbors[0].shape
    # order of nearest neighbours corresponding to single input
    df.insert(0, 'ord', repeating_rng(n_ctrl, n_trial))
    # the index corresponding to which trial the row belongs
    idx_ctrl_repeated = list(itertools.chain.from_iterable([[no] * n_ctrl for no in trial_idx.tolist()]))
    df.insert(0, f'idx_trial', idx_ctrl_repeated)
    # making a category col that indicates if the idx_ctrl col is a trial or a ctrl
    in_trial = list(itertools.chain.from_iterable([['trial'] + (['ctrl'] * (n_ctrl - 1)) for _ in range(n_trial)]))
    df['in_trial'] = in_trial
    assert (df['idx_trial'].unique() == trial_idx.values).all(), (
        'the idx_trial should include only but all the original trial idxs')
    # replaceing the df index values of the ctrls with the values from the idx series in the df
    df.loc[df['in_trial'] == 'ctrl', 'idx_ctrl'] = df[f'idx_ctrl'].map(ctrl_idx)
    return df


def nearest_neighbours(feat: pd.DataFrame, features: List[str], in_trial_col: str, idx_col: str,
                       n_neighbours: int = 15, norm: bool = True) -> pd.DataFrame:
    """
    Perform nearest neighbours returning the selected controls and the original trials in a tidy format.
    Args:
        feat: DataFrame of features and additional columns ie index
        features: a list of the to input the model
        in_trial_col: a string name corresponding to a booleon column, True is in the trial
        idx_col: a string name corresponding to a column that uniuqly identidies the elements
        n_neighbours: the number of neighbours the model returns for each trial item

    Returns:
        DataFrame of trials/controls and features
    """
    feat_norm = standardise(feat, features) if norm else feat
    feat_norm.index = feat.index
    trial_norm, control_norm = split_df(feat_norm, in_trial_col)
    assert feat_norm.shape[0] == (trial_norm.shape[0] + control_norm.shape[0])
    knn = NearestNeighbors(n_neighbours, n_jobs=-1)
    knn.fit(control_norm[features])
    neighbours = knn.kneighbors(trial_norm[features])
    neighbours = neighbours2df(neighbours, trial_norm[idx_col], control_norm[idx_col])
    neighbours = neighbours.merge(feat, how='left', left_on='idx_ctrl', right_on=idx_col,
                                  suffixes=('', '_DROP')).filter(regex='^(?!.*_DROP)')
    return neighbours


def kmeans_elbow(data, n=15, path: Optional[Path] = None) -> None:
    """perform elbow method using kmeans visualising the loss with plotly"""
    losses = {}
    for i in tqdm(range(1, n), total=n):
        km = KMeans(i).fit(data)
        losses[i] = km.inertia_
    losses = pd.Series(losses).to_frame(name='loss')
    fig = px.line(losses, x=losses.index, y='loss')
    filename = path.as_posix() if path else 'temp-plot.html'
    plot(fig, filename=filename)


# dimensionality reduction #############################################################################################


def umapper(df: Union[pd.DataFrame], n_components: int = 3, normalise: bool = True) -> pd.DataFrame:
    """reduce dimensions of DataFrame by applying umap"""
    logger.info(f'starting umap')
    idx = df.index if isinstance(df, pd.DataFrame) else None
    if normalise:
        df = StandardScaler().fit_transform(df)
    reduced = umap.UMAP(n_components=n_components).fit_transform(df)
    return pd.DataFrame(reduced, index=idx, columns=[f'dim{i}' for i in range(reduced.shape[1])])


def pca(x: Union[pd.DataFrame, np.ndarray], n_components=None, show_var: bool = True,
        normalise: bool = True) -> Tuple[Union[pd.DataFrame, np.ndarray], PCA]:
    """
    run pca and return principle components and fitted estimator
    Args:
        x:
        n_components:
        show_var:
        normalise:

    Returns:
        pcs: Principal components as numpy arrays
        pca: fitted pca model object
    """
    pca = PCA(n_components)
    if normalise:
        pcs = pca.fit_transform(StandardScaler().fit_transform(x))
    if show_var:
        print(pca_explained_var(pca))
    if isinstance(x, pd.DataFrame):
        pcs = pd.DataFrame(pcs, index=x.index, columns=[f'dim{i}' for i in range(pcs.shape[1])])
    return pcs, pca


def pca_heatmap(pca: PCA, width=14, height=5.5):
    """heatmap of principle components"""
    df = pd.DataFrame(pca.components_, columns=[f'dim{i}' for i in range(pca.components_.shape[1])])
    df = df.round(2)
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    return fig, ax


def pca_explained_var(pca: PCA, fmt: bool = True) -> pd.DataFrame:
    """
    return the variance captured by each principle component
    """
    ratios = {
        'var': pca.explained_variance_,
        'var_ratio': pca.explained_variance_ratio_,
        'var_ratio_cum': pca.explained_variance_ratio_.cumsum(),
    }
    df = pd.DataFrame(ratios)
    if fmt:
        df = pd.concat([df.iloc[:, 0], df.iloc[:, 1:].applymap('{:.1%}'.format)], axis=1)
    return df


# forecasting ##########################################################################################################


def make_ts_features(array, Tx: int, Ty: int = 1):
    """
    split a series into ~ N / T test train splits. example of one of the splits [t0, t1, ... tT-1] [tT]
    Args:
        array: a series to be transformed into features
        Tx: number of time steps in features
        Ty: number of time steps in target

    Returns:
        x: training data with windows of size Tx
        y: target data with windows of size Ty
    """
    if Tx > len(array):
        raise ValueError('Window length cannot be longer than series...')
    nsplits: int = len(array) - Tx - Ty + 1
    x = np.zeros((nsplits, Tx))
    y = np.zeros((nsplits, Ty))
    for i in range(nsplits):
        x[i] = array[i:i + Tx]
        y[i] = array[i + Tx: i + Tx + Ty]
    x = x.reshape(*x.shape, 1)
    print(f'x shape: {x.shape}, y shape: {y.shape}')
    return x, y


def predict_ts(last_x, y_test: np.array, predictor: Callable) -> pd.array:
    predictions: List = []
    idx = 1
    while len(predictions) < len(y_test):
        pred = predictor(last_x.reshape(1, -1, 1))[0, 0]
        predictions.append(pred)

        last_x = np.roll(last_x, -1)
        last_x[-1] = pred
        idx += 1
    return predictions


# deep learning ########################################################################################################


def train_log(epoch, loss, **metrics):
    return f'epoch: {epoch}, loss: {loss:.4f} ' + ''.join([f'{k}: {v:.4f}, ' for k, v in metrics.items()])


def embeddings2df(learn, cats, cat) -> pd.DataFrame:
    """extract embedding weights and corresponding classes from fastai learner retruning
    them as a DataFrame"""
    get_weights: Callable = lambda cat: learn.model.embeds[cats.index(cat)].weight.detach().numpy()
    get_classes: Callable = lambda cat: learn.data.label_list.train.x.classes[cat]
    weights, classes = get_weights(cat), get_classes(cat)
    df = pd.DataFrame(weights, columns=[f'dim{i}' for i in range(weights.shape[1])])
    df.insert(loc=0, column=cat, value=classes)
    return df


def umap_df_dim3(df) -> pd.DataFrame:
    """transforming weights matrix into DataFrame with 3 columns, reducing with umap where
    for embeddings with dim > 3"""
    df = df.set_index(df.columns[0])
    if df.shape[1] > 3:
        return umapper(df)
    return df


# fasttest #############################################################################################################


def make_fasttext_dataset(train: pd.DataFrame, valid: pd.DataFrame, text: str, lbl: str, path: Path, fname: str
                          ) -> Tuple[Path, Path]:
    """take train and valid dataframes and format/save them as files in a fasttext ready format and returns Path
    objects that correspond to the saved files"""
    train[text] = train[text].str.split().str.join(' ').str.replace('[^\w\s]', '')
    valid[text] = valid[text].str.split().str.join(' ').str.replace('[^\w\s]', '')
    train['label'] = '__label__' + train[lbl].astype(str)
    valid['label'] = '__label__' + valid[lbl].astype(str)
    train[['label', text]].to_csv((path / f'{fname}.train').as_posix(), header=None, sep=' ', index=False)
    valid[['label', text]].to_csv((path / f'{fname}.valid').as_posix(), header=None, sep=' ', index=False)
    return path / f'{fname}.train', path / f'{fname}.valid'


def add_binary_predictions(df: pd.DataFrame, model, text: str) -> pd.DataFrame:
    """Apply model predictions to a DataFrame column and add columns for the predicted class and probability"""
    df['pred_ft'], df['prob_ft'] = model.predict(df[text].tolist())
    df['pred_ft'], df['prob_ft'] = df['pred_ft'].str[0].str.strip('__label__').astype(int), df['prob_ft'].str[0]
    return df


# data creation ########################################################################################################


def get_data():
    """Synthetic binary classification dataset."""
    data, targets = make_classification(
        n_samples=10_000,
        n_features=45,
        n_informative=5,
        n_redundant=7,
        n_clusters_per_class=5,
        flip_y=0.2,
    )
    return data, targets


# monkey patching sklearn estimators ###################################################################################


from sklearn.base import BaseEstimator


def fit_add_feature_names(model, df):
    model.fit(df)
    model.feature_names = df.columns.tolist()


BaseEstimator.fit_add_feature_names = fit_add_feature_names
pd.DataFrame.standardise = standardise
