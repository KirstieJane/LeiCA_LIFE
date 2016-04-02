
import os, pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age, residualize_group_data
from sklearn.decomposition import PCA
from nipype.utils import filemanip

os.chdir('/nobackup/clustercache/liem/wd_learning/learning_prepare_data_wf/_selection_criterium_F/_multimodal_in_data_name_alff/prediction_split/')
i=filemanip.loadpkl('_inputs.pklz')

for k in i.keys():
    s='%s = i[\'%s\']'%(k,k)
    exec(s)




df = pd.read_pickle(df_file)
# add ouput cols to df
df['split_group'] = ''
df['pred_age_train'] = np.nan
df['pred_age_test'] = np.nan

X = np.load(X_file)
y = df[['age']].values.squeeze()

confounds = df[['mean_FD_P']].values

ind = range(X.shape[0])
x_train, x_test, y_train, y_test, confounds_train, confounds_test, ind_train, ind_test = train_test_split(X, y,
                                                                                                                  confounds,
                                                                                                                  ind,
                                                                                                                  test_size=0.5,
                                                                                                                  random_state=666)



import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm




#####
# we explicitly generate the outer_cv decorator so we can use it twice
#outer_cv = optunity.cross_validated(x=X, y=y, num_folds=3)


#fixme try with fd ana_stream
if False:
    x_train = residualize_group_data(x_train, confounds_train)
    x_test = residualize_group_data(x_test, confounds_test)



def compute_mse_poly_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""
    # PREPROCESSING
    fill_missing = Imputer()
    var_thr = VarianceThreshold()
    normalize = MinMaxScaler()

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, epsilon):
        model = sklearn.svm.SVR(C=C, epsilon=epsilon, kernel='linear')
        pipeline = Pipeline([
            ('fill_missing', fill_missing),
            ('var_thr', var_thr),
            ('normalize', normalize),
            ('regression_model', model),
        ])

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        mae = mean_absolute_error(y_test, predictions)
        return mae

    # optimize parameters
    pmap = optunity.parallel.create_pmap(27)
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[0, .1], epsilon=[0,1], pmap=pmap)
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = sklearn.svm.SVR(kernel='linear', **optimal_pars)
    pipeline = Pipeline([
    ('fill_missing', fill_missing),
    ('var_thr', var_thr),
    ('normalize', normalize),
    ('regression_model', tuned_model),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return mean_absolute_error(y_test, predictions), optunity.metrics.r_squared(y_test, predictions)

# wrap with outer cross-validation

a = compute_mse_poly_tuned(x_train, y_train, x_test, y_test)
print a

