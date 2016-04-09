###############################################################################################################
# DATA WRANGLING
# aggregate and select data
def aggregate_multimodal_metrics_fct(multimodal_list, unimodal_lookup_dict):
    import numpy as np
    import os, pickle

    multimodal_backprojection_info = []

    X_multimodal = None
    for unimodal_name in multimodal_list:
        this_unimod_backprojection_info = {}

        unimodal_X_file = unimodal_lookup_dict[unimodal_name]['X_file']
        X_unimodal = np.load(unimodal_X_file)
        unimodal_backprojection_info = pickle.load(
            open(unimodal_lookup_dict[unimodal_name]['unimodal_backprojection_info_file']))

        if X_multimodal is None:
            this_unimod_backprojection_info['start_index'] = 0
            X_multimodal = X_unimodal
        else:
            this_unimod_backprojection_info['start_index'] = X_multimodal.shape[1]
            X_multimodal = np.hstack((X_multimodal, X_unimodal))

        this_unimod_backprojection_info['end_index'] = X_multimodal.shape[1]
        this_unimod_backprojection_info['data_type'] = unimodal_backprojection_info['data_type']
        this_unimod_backprojection_info['save_template'] = unimodal_backprojection_info['save_template']
        this_unimod_backprojection_info['masker'] = unimodal_backprojection_info['masker']

        multimodal_backprojection_info.append(this_unimod_backprojection_info)
        this_unimod_backprojection_info['unimodal_name'] = unimodal_name

    multimodal_name = '__'.join(multimodal_list)
    X_multimodal_file = os.path.abspath('X_multimodal.npy')
    np.save(X_multimodal_file, X_multimodal)

    return X_multimodal_file, multimodal_backprojection_info, multimodal_name


def select_subjects_fct(df_all_subjects_pickle_file, subjects_selection_crit_dict, selection_criterium):
    import pandas as pd
    import os
    import numpy as np

    df = pd.read_pickle(df_all_subjects_pickle_file)

    #   EXCLUSION HERE:
    q_str = '(' + ') & ('.join(subjects_selection_crit_dict[selection_criterium]) + ')'
    df['select'] = False
    selected_subjects_list = df.query(q_str).index.values
    df.loc[selected_subjects_list, ['select']] = True
    subjects_selection_index = df['select'].values

    df_use = df.loc[subjects_selection_index]
    df_use_file = os.path.abspath('df_use.csv')
    df_use.to_csv(df_use_file)
    df_use_pickle_file = os.path.abspath('df_use.pkl')
    df_use.to_pickle(df_use_pickle_file)

    return df_use_file, df_use_pickle_file, subjects_selection_index


def select_multimodal_X_fct(X_multimodal_file, subjects_selection_index):
    import os, numpy as np
    X_multimodal = np.load(X_multimodal_file)
    X_multimodal_selected = X_multimodal[subjects_selection_index, :]
    X_multimodal_selected_file = os.path.abspath('X_multimodal_selected.npy')
    np.save(X_multimodal_selected_file, X_multimodal_selected)
    return X_multimodal_selected_file


###############################################################################################################
# PREDICTION
# and helpers for residualizing and plotting
def run_prediction_split_fct(X_file, target_name, selection_criterium, df_file, data_str, regress_confounds=False,
                             use_grid_search=False, scaler='standard', rfe=False, strat_split=False):
    import os, pickle
    import numpy as np
    import pandas as pd
    from sklearn.svm import SVR
    from sklearn.cross_validation import cross_val_score, cross_val_predict
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.preprocessing import Imputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV
    from LeiCA_LIFE.learning.learning_utils import pred_real_scatter, plot_brain_age, residualize_group_data
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectKBest, f_regression
    data_str = target_name + '__' + selection_criterium + '__' + data_str

    df = pd.read_pickle(df_file)
    # add ouput cols to df
    df['split_group'] = ''
    df['pred_age_train'] = np.nan
    df['pred_age_test'] = np.nan

    X = np.load(X_file)
    # FIXME
    # X = np.concatenate((X,np.sqrt(X)), axis=1)

    y = df[[target_name]].values.squeeze()

    confounds = df[['mean_FD_P']].values

    ind = range(X.shape[0])
    X_train, X_test, y_train, y_test, \
    confounds_train, confounds_test, ind_train, ind_test = train_test_split(
        X, y,
        confounds,
        ind,
        test_size=0.5,
        random_state=666)
    df.ix[ind_train, ['split_group']] = 'train'
    df.ix[ind_test, ['split_group']] = 'test'

    # REGRESS OUT CONFOUNDS IF NEEDED
    if regress_confounds:
        X_train = residualize_group_data(X_train, confounds_train)
        X_test = residualize_group_data(X_test, confounds_test)

    # PREPROCESSING
    fill_missing = Imputer()
    var_thr = VarianceThreshold()

    if scaler == 'standard':
        normalize = StandardScaler()
    elif scaler == 'robust':
        normalize = RobustScaler()
    elif scaler == 'minmax':
        normalize = MinMaxScaler()
    else:
        raise Exception('scaler not detected: %s' % scaler)

    regression_model = SVR(kernel='linear', cache_size=1000)
    # fixme
    # C = 50000
    # epsilon = .1
    # regression_model = SVR(kernel='linear', C=C, epsilon=epsilon)
    # regression_model = SVR(kernel='poly', degree=2, C=C, epsilon=epsilon)


    pipeline_list = [('fill_missing', fill_missing),
                     ('var_thr', var_thr),
                     ('normalize', normalize)]

    if rfe:
        n_features = X.shape[1]
        n_features_to_select = int(n_features * .2)
        if n_features_to_select > 0:  # only perform rfe if features remain
            anova_filter = SelectKBest(f_regression, k=n_features_to_select)
            #eliminate = RFE(estimator=regression_model, n_features_to_select=n_features_to_select, step=.5)
            pipeline_list.append(('anova_filter', anova_filter))

    pipeline_list.append(('regression_model', regression_model))

    pipeline = Pipeline(pipeline_list)

    if use_grid_search:
        params = {
            'regression_model__C': [.00001, .0001, .001, .01, .1, 1, 10,
                                    100, 2000, 14450],
            # 'regression_model__epsilon': [0, .005, .01, .05, .1, 1, 5, 10],
        }
        pipe = GridSearchCV(pipeline, params, cv=5,
                            scoring='mean_absolute_error', n_jobs=2)
    elif strat_split:
        pass # create pipe later
    else:
        pipe = pipeline

    if strat_split:
        # n_bins = 10
        # df['mean_FD_P_bins'] = pd.cut(df['mean_FD_P'], n_bins, labels=range(n_bins))
        # cv = StratifiedShuffleSplit(df['mean_FD_P_bins'].values, 10, test_size=0.5, random_state=0)
        # pipe.fit(X, y, cv=cv)
        # r2 = cross_val_score(pipe, X, y, cv=cv, scoring='r2')
        # mae = cross_val_score(pipe, X, y, cv=cv, scoring='mean_absolute_error')
        from sklearn.ensemble import BaggingRegressor
        n_bins = 10
        df['mean_FD_P_bins'] = pd.cut(df['mean_FD_P'], n_bins, labels=range(n_bins))
        cv = StratifiedShuffleSplit(df['mean_FD_P_bins'].values, 2, test_size=0.5, random_state=0)
        ind_train = cv.y==0
        X_train = X[ind_train,:]
        y_train = y[ind_train]
        confounds_train = confounds[ind_train]

        ind_test = cv.y==1
        X_test = X[ind_test,:]
        y_test = y[ind_test]
        confounds_test = confounds[ind_test]

        df.ix[ind_train, ['split_group']] = 'train'
        df.ix[ind_test, ['split_group']] = 'test'

        # REGRESS OUT CONFOUNDS IF NEEDED
        if regress_confounds:
            X_train = residualize_group_data(X_train, confounds_train)
            X_test = residualize_group_data(X_test, confounds_test)

        pipe = BaggingRegressor(base_estimator=pipeline, n_estimators=10, max_samples=.75)
        pipe.fit(X_train, y_train)
        y_predicted = pipe.predict(X_test)

        df.ix[ind_train, ['pred_age_train']] = np.nan
        df.ix[ind_test, ['pred_age_test']] = y_predicted

        test_mae = mean_absolute_error(y_test, y_predicted)
        test_r2 = r2_score(y_test, y_predicted)
        test_rpear2 = np.corrcoef(y_test, y_predicted)[0, 1]

        train_mae = np.nan
        train_r2 = np.nan

        test_r2_std = np.nan
        test_mae_std = np.nan

        # train_r2 = np.nan
        # train_mae = np.nan
        # test_r2 = r2.mean()
        # test_r2_std = r2.std()
        # test_mae = np.abs(mae.mean())
        # test_mae_std = mae.std()
        # test_rpear2 = np.nan
        #
        # out_str = 'r2:  M = %0.7f \t SD = %0.7f\n' % (test_r2, test_r2_std)
        # out_str += 'mae: M = %0.7f \t SD = %0.7f\n' % (test_mae, test_mae_std)
        # scatter_file = os.path.abspath('score_%s.txt' % data_str)
        # with open(scatter_file, 'w') as fi:
        #     fi.write(out_str)
        #
        #
        # # fixme needed:?
        # empty_file = os.path.abspath('empty.txt')
        # with open(empty_file, 'w') as fi:
        #     fi.write('')
        # brain_age_scatter_file = df_use_file = empty_file  # or = ''
        ###

        # test_mae_list = []
        # test_r2_list = []
        # test_rpear2_list = []
        # coef_list = []
        # intercept_list=[]
        # for train_index, test_index in cv:
        #     X_train, X_test = X[train_index], X[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #
        #     pipe.fit(X_train, y_train)
        #     y_predicted_train = pipe.predict(X_train)
        #     y_predicted = pipe.predict(X_test)
        #
        #     test_mae_list.append(mean_absolute_error(y_test, y_predicted))
        #     test_r2_list.append(r2_score(y_test, y_predicted))
        #     test_rpear2_list.append(np.corrcoef(y_test, y_predicted)[0, 1])
        #
        #     coef_list.append(backproject_weights(pipe))
        #     intercept_list.append(pipe.named_steps['regression_model'].intercept_)


    else:
        pipe.fit(X_train, y_train)
        y_predicted_train = pipe.predict(X_train)
        y_predicted = pipe.predict(X_test)

        df.ix[ind_train, ['pred_age_train']] = y_predicted_train
        df.ix[ind_test, ['pred_age_test']] = y_predicted

        test_mae = mean_absolute_error(y_test, y_predicted)
        test_r2 = r2_score(y_test, y_predicted)
        test_rpear2 = np.corrcoef(y_test, y_predicted)[0, 1]

        train_mae = mean_absolute_error(y_train, y_predicted_train)
        train_r2 = r2_score(y_train, y_predicted_train)

        test_r2_std = np.nan
        test_mae_std = np.nan

    title_str = 'r2: {:.3f} MAE:{:.3f}'.format(test_r2, test_mae)
    scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)

    # grid scores as textfile
    if use_grid_search:
        sorted_grid_score = sorted(pipe.grid_scores_,
                                   key=lambda x: x.mean_validation_score,
                                   reverse=True)
        score_str = [str(n) + ': ' + str(g) for n, g in
                     enumerate(sorted_grid_score)]
        gs_text_file = os.path.abspath('gs_txt_' + data_str + '.txt')
        with open(gs_text_file, 'w') as f:
            f.write('\n'.join(score_str))
        scatter_file = [scatter_file, gs_text_file]

    brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)

    df_use_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
    df.to_pickle(df_use_file)

    # performace results df
    df_res_out_file = os.path.abspath(data_str + '_df_results.pkl')
    df_res = pd.DataFrame(
        {'FD_res': regress_confounds, 'r2_train': [train_r2], 'MAE_train': [train_mae],
         'r2_test': [test_r2], 'r2_test_std': [test_r2_std], 'rpear2_test': [test_rpear2],
         'MAE_test': [test_mae], 'MAE_test_std': [test_mae_std]},
        index=[data_str])
    df_res.to_pickle(df_res_out_file)

    model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
    with open(model_out_file, 'w') as f:
        pickle.dump(pipe, f)
    return scatter_file, brain_age_scatter_file, df_use_file, model_out_file, df_res_out_file


def residualize_group_data(signals, confounds):
    '''
    regresses out confounds from signals
    signals.shape: subjects x n_data_points
    confounds.shape: subjects x n_confounds
    returns residualized_signals.shape: subjects x n_data_points
    '''
    from nilearn.signal import clean
    residualized_signals = clean(signals, detrend=False, standardize=False, confounds=confounds, low_pass=None,
                                 high_pass=None, t_r=None)
    return residualized_signals


def pred_real_scatter(y_test, y_test_predicted, title_str, in_data_name):
    import os
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure()
    plt.scatter(y_test, y_test_predicted)
    plt.plot([10, 80], [10, 80], 'k')
    plt.xlabel('real')
    plt.ylabel('predicted')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_' + in_data_name + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


def plot_brain_age(y_test, y_test_predicted, in_data_name):
    import os
    import pylab as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    brain_age = y_test_predicted - y_test
    title_str = 'mean brain age: %s\nr = %s' % (brain_age.mean(), np.corrcoef(brain_age, y_test)[0, 1])
    plt.figure()
    plt.scatter(y_test, brain_age)
    plt.xlabel('real')
    plt.ylabel('brain_age (pred-real)')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_brain_age_' + in_data_name + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


###############################################################################################################
# BACKPROJECTION
def backproject_and_split_weights_fct(trained_model_file, multimodal_backprojection_info, data_str, target_name):
    from LeiCA_LIFE.learning.learning_utils import backproject_weights_to_full_space, save_weights

    data_str = target_name + '__' + data_str

    out_file_list = []
    out_file_render_list = []

    weights = backproject_weights_to_full_space(trained_model_file)

    for unimodal_backprojection_info in multimodal_backprojection_info:
        out_data = weights[:, unimodal_backprojection_info['start_index']:unimodal_backprojection_info['end_index']]
        data_type = unimodal_backprojection_info['data_type']
        save_template = unimodal_backprojection_info['save_template']
        masker = unimodal_backprojection_info['masker']

        out_name = data_str + '__' + unimodal_backprojection_info['unimodal_name']

        out_file, out_file_render = save_weights(out_data, data_type, save_template, out_name, masker)
        out_file_list.append(out_file)
        out_file_render_list.append(out_file_render)

    return out_file_list, out_file_render_list


def backproject_weights(trained_model):
    '''
    Takes a vector of weights and fills NaNs for all the features that have been eliminated during preprocessing
    '''
    import numpy as np
    import sklearn

    ADD_CONSTANT = 1000

    if isinstance(trained_model, sklearn.grid_search.GridSearchCV):  # necessary because gs object is set up diffrently
        steps = trained_model.best_estimator_.named_steps
    else:
        steps = trained_model.named_steps

    imp = steps['fill_missing']
    var_thr = steps['var_thr']
    scaler = steps['normalize']
    regression_model = steps['regression_model']

    if 'anova_filter' in steps.keys():
        # back trainsform weights from sparse (anova selection) to full
        # insert nans in eliminated features
        anova_filter = steps['anova_filter']
        weights = anova_filter.inverse_transform(regression_model.coef_ + ADD_CONSTANT)
        sparse_ind = (weights == 0)
        weights -= ADD_CONSTANT

    elif 'regression_model' in steps.keys():
        weights = regression_model.coef_
        sparse_ind = np.zeros_like(weights).astype(np.bool)

    else:
        raise Exception('neither rfe nor regression_model found in steps.keys() %s' % steps.keys())

    # add constant to the weights in order to differentiate between 0 weights and zeros that have been iputed in the
    # case of 0-variance features
    # after replacing 0s with nans, substract constant
    mapped_back_var_thr = var_thr.inverse_transform(weights + ADD_CONSTANT)
    mapped_back_var_thr[mapped_back_var_thr == 0] = np.nan
    mapped_back_var_thr -= ADD_CONSTANT

    mapped_back = np.zeros((1, imp.statistics_.shape[0]))
    mapped_back.fill(np.nan)
    mapped_back[0, ~np.isnan(imp.statistics_)] = mapped_back_var_thr

    # set sparse weights to nan
    mapped_back[sparse_ind] = np.nan
    return mapped_back


def backproject_weights_to_full_space(trained_model_file):
    '''
    Loads trained model and call backproject_weights
    '''
    import pickle
    from LeiCA_LIFE.learning.learning_utils import backproject_weights
    trained_model = pickle.load(open(trained_model_file))
    return backproject_weights(trained_model)


def save_weights(data, data_type, save_template, outfile_name, masker=None):
    '''
    saves weights as nii or other file
    '''
    import nibabel as nb
    import numpy as np
    import pandas as pd
    import os
    from nilearn import plotting
    import pylab as plt
    from LeiCA_LIFE.learning.prepare_data_utils import vector_to_matrix

    weights_file_str = '_weights'
    render_file_str = '_rendering'
    outfile_render = os.path.abspath(outfile_name + render_file_str + '.pdf')

    plt.figure()

    if data_type == '3dnii':
        nii = masker.inverse_transform(data)
        outfile = os.path.abspath(outfile_name + weights_file_str + '.nii.gz')
        nii.to_filename(outfile)

        plotting.plot_stat_map(outfile)


    elif data_type == 'matrix':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.npy')
        data = vector_to_matrix(data, use_diagonal=False)
        np.save(outfile, data)

        plt.imshow(data, interpolation='nearest')
        plt.colorbar()

    elif data_type == 'fs_cortical':
        fs_img = nb.load(save_template)
        # Sabina says that's how nb handels this case
        fs_data = fs_img.get_data()
        fs_data[:] = data.reshape(fs_data.shape[0], 1, 1)
        # check whether writing data into image worked
        np.testing.assert_array_almost_equal(fs_img.get_data().squeeze(), data.squeeze())

        outfile = os.path.abspath(outfile_name + weights_file_str + '.mgz')
        fs_img.to_filename(outfile)

        # # fixme DOES NOT WORK
        # from surfer import Brain
        # # replace nans by 0, otherwise pysurfer cannot display them
        # data_nonans = data.copy()
        # data_nonans[np.isnan(data_nonans)] = 0
        # hemi =  outfile_name[outfile_name.find('__')+2:][:2]
        # brain = Brain('fsaverage5', hemi, 'inflated', config_opts={"background": "white"}) #, background="white") ,
        # brain.add_data(data_nonans.squeeze())
        # brain.save_montage(outfile_render, order=['lat', 'med'], orientation='h', border_size=10)
        # brain.close()

    elif data_type == 'fs_tab':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        df = pd.read_csv(save_template, index_col=0, delimiter='\t')
        df.index.name = 'subject_id'
        df.drop(df.index[0], axis=0, inplace=True)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    elif data_type == 'behav':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        # for behav save_template contains colnames
        df = pd.DataFrame([], columns=save_template)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    else:
        raise Exception('wrong data type %s' % data_type)

    plt.savefig(outfile_render)
    plt.close()

    return outfile, outfile_render


###############################################################################################################
# PREDICTION from trained model
def run_prediction_from_trained_model_fct(trained_model_file, X_file, target_name, selection_criterium, df_file,
                                          data_str, regress_confounds=False):
    import os, pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score, mean_absolute_error
    import sklearn.ensemble.bagging
    from LeiCA_LIFE.learning.learning_utils import pred_real_scatter, plot_brain_age, residualize_group_data

    data_str = target_name + '__' + selection_criterium + '__' + data_str

    df = pd.read_pickle(df_file)
    df['pred_age_test'] = np.nan

    X = np.load(X_file)
    y = df[[target_name]].values.squeeze()
    confounds = df[['mean_FD_P']].values

    # REGRESS OUT CONFOUNDS IF NEEDED
    if regress_confounds:
        X = residualize_group_data(X, confounds)

    with open(trained_model_file, 'r') as f:
        pipe = pickle.load(f)

    # bagging can't deal with missing values-> impute
    if np.any(np.isnan(X)) & isinstance(pipe, sklearn.ensemble.bagging.BaggingRegressor):
        # get mean of means
        statistics_ = np.asarray([estimator.named_steps['fill_missing'].statistics_ for estimator in pipe.estimators_]).mean(0)
        fill_matrix = np.repeat([statistics_],X.shape[0], axis=0)
        X[np.isnan(X)] = fill_matrix[np.isnan(X)]



    # RUND PREDICTION
    y_predicted = pipe.predict(X)

    df.ix[:, ['pred_age_test']] = y_predicted

    test_mae = mean_absolute_error(y, y_predicted)
    test_r2 = r2_score(y, y_predicted)
    test_rpear2 = np.corrcoef(y, y_predicted)[0, 1]

    test_r2_std = np.nan
    test_mae_std = np.nan
    train_r2 = np.nan
    train_mae = np.nan

    title_str = 'r2: {:.3f} MAE:{:.3f}'.format(test_r2, test_mae)
    scatter_file = pred_real_scatter(y, y_predicted, title_str, data_str)

    brain_age_scatter_file = plot_brain_age(y, y_predicted, data_str)

    df_use_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
    df.to_pickle(df_use_file)

    # performace results df
    df_res_out_file = os.path.abspath(data_str + '_df_results.pkl')
    df_res = pd.DataFrame(
        {'FD_res': regress_confounds, 'r2_train': [train_r2], 'MAE_train': [train_mae],
         'r2_test': [test_r2], 'r2_test_std': [test_r2_std], 'rpear2_test': [test_rpear2],
         'MAE_test': [test_mae], 'MAE_test_std': [test_mae_std]},
        index=[data_str])
    df_res.to_pickle(df_res_out_file)

    return scatter_file, brain_age_scatter_file, df_use_file, df_res_out_file
