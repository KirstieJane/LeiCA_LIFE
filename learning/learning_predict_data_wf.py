def learning_predict_data_wf(working_dir,
                             ds_dir,
                             in_data_name_list,
                             subjects_selection_crit_dict,
                             subjects_selection_crit_names_list,
                             aggregated_subjects_dir,
                             target_list,
                             use_n_procs,
                             plugin_name):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from itertools import chain
    import pandas as pd


    # ensure in_data_name_list is list of lists
    in_data_name_list = [i if type(i) == list else [i] for i in in_data_name_list]
    in_data_name_list_unique = list(set(chain.from_iterable(in_data_name_list)))



    #####################################
    # GENERAL SETTINGS
    #####################################
    wf = Workflow(name='learning_predict_data_wf')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': False,
                                                                       'remove_unnecessary_outputs': False,
                                                                       'job_finished_timeout': 120})
    config.update_config(nipype_cfg)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = os.path.join(ds_dir, 'group_learning_prepare_data')

    ds.inputs.regexp_substitutions = [
        # ('subject_id_', ''),
        ('_parcellation_', ''),
        ('_bp_freqs_', 'bp_'),
        ('_extraction_method_', ''),
        ('_subject_id_[A0-9]*/', '')
    ]
    ds_pdf = Node(nio.DataSink(), name='ds_pdf')
    ds_pdf.inputs.base_directory = os.path.join(ds_dir, 'pdfs')
    ds_pdf.inputs.parameterization = False


    #####################################
    # SET ITERATORS
    #####################################
    # SUBJECTS ITERATOR
    in_data_name_infosource = Node(util.IdentityInterface(fields=['in_data_name']), name='in_data_name_infosource')
    in_data_name_infosource.iterables = ('in_data_name', in_data_name_list_unique)

    multimodal_in_data_name_infosource = Node(util.IdentityInterface(fields=['multimodal_in_data_name']),
                                              name='multimodal_in_data_name_infosource')
    multimodal_in_data_name_infosource.iterables = ('multimodal_in_data_name', in_data_name_list)

    subject_selection_infosource = Node(util.IdentityInterface(fields=['selection_criterium']),
                                        name='subject_selection_infosource')
    subject_selection_infosource.iterables = ('selection_criterium', subjects_selection_crit_names_list)

    target_infosource = Node(util.IdentityInterface(fields=['target_name']),
                             name='target_infosource')
    target_infosource.iterables = ('target_name', target_list)



    #####################################
    # GET INFO AND SELECT FILES
    #####################################
    df_all_subjects_pickle_file = os.path.join(aggregated_subjects_dir, 'df_all_subjects_pickle_file/df_all.pkl')
    df = pd.read_pickle(df_all_subjects_pickle_file)

    # build lookup dict for unimodal data
    unimodal_lookup_dict = {}
    for k in in_data_name_list_unique:
        unimodal_lookup_dict[k] = {'X_file': os.path.join(aggregated_subjects_dir,
                                                          'X_file/_in_data_name_{in_data_name}/vectorized_aggregated_data.npy'.format(
                                                              in_data_name=k)),
                                   'unimodal_backprojection_info_file': os.path.join(aggregated_subjects_dir,
                                                                                     'unimodal_backprojection_info_file/_in_data_name_{in_data_name}/unimodal_backprojection_info.pkl'.format(
                                                                                         in_data_name=k))
                                   }


    ###############################################################################################################
    # AGGREGATE MULTIMODAL METRICS
    # stack single modality arrays

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

        multimodal_name = '__'.join(multimodal_list)
        X_multimodal_file = os.path.abspath('X_multimodal.npy')
        np.save(X_multimodal_file, X_multimodal)

        return X_multimodal_file, multimodal_backprojection_info, multimodal_name

    aggregate_multimodal_metrics = Node(util.Function(input_names=['multimodal_list', 'unimodal_lookup_dict'],
                                                      output_names=['X_multimodal_file',
                                                                    'multimodal_backprojection_info',
                                                                    'multimodal_name'],
                                                      function=aggregate_multimodal_metrics_fct),
                                        name='aggregate_multimodal_metrics')
    wf.connect(multimodal_in_data_name_infosource, 'multimodal_in_data_name', aggregate_multimodal_metrics,
               'multimodal_list')
    aggregate_multimodal_metrics.inputs.unimodal_lookup_dict = unimodal_lookup_dict



    ###############################################################################################################


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

    select_subjects = Node(
        util.Function(input_names=['df_all_subjects_pickle_file',
                                   'subjects_selection_crit_dict',
                                   'selection_criterium'],
                      output_names=['df_use_file',
                                    'df_use_pickle_file',
                                    'subjects_selection_index'],
                      function=select_subjects_fct),
        name='select_subjects')

    select_subjects.inputs.df_all_subjects_pickle_file = df_all_subjects_pickle_file
    select_subjects.inputs.subjects_selection_crit_dict = subjects_selection_crit_dict
    wf.connect(subject_selection_infosource, 'selection_criterium', select_subjects, 'selection_criterium')



    # select subjects (rows) from multimodal X according to selection criterium
    def select_multimodal_X_fct(X_multimodal_file, subjects_selection_index):
        import os, numpy as np
        X_multimodal = np.load(X_multimodal_file)
        X_multimodal_selected = X_multimodal[subjects_selection_index, :]
        X_multimodal_selected_file = os.path.abspath('X_multimodal_selected.npy')
        np.save(X_multimodal_selected_file, X_multimodal_selected)
        return X_multimodal_selected_file

    select_multimodal_X = Node(
        util.Function(input_names=['X_multimodal_file', 'subjects_selection_index', 'selection_criterium'],
                      output_names=['X_multimodal_selected_file'],
                      function=select_multimodal_X_fct),
        name='select_multimodal_X')
    wf.connect(aggregate_multimodal_metrics, 'X_multimodal_file', select_multimodal_X, 'X_multimodal_file')
    wf.connect(select_subjects, 'subjects_selection_index', select_multimodal_X, 'subjects_selection_index')




    ###############################################################################################################
    # RUN PREDICTION
    #

    def run_prediction_split(X_file, target_name, selection_criterium, df_file, data_str, regress_confounds=False,
                             use_grid_search=False):
        import os, pickle
        import numpy as np
        import pandas as pd
        from sklearn.svm import SVR
        from sklearn.cross_validation import cross_val_score, cross_val_predict
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from sklearn.preprocessing import Imputer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.cross_validation import train_test_split
        from sklearn.grid_search import GridSearchCV
        # for some reason, only full path works
        from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age, residualize_group_data
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import RFE

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
        confounds_train, confounds_test, ind_train, ind_test = train_test_split(X, y,
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
        normalize = MinMaxScaler()  # StandardScaler()

        regression_model = SVR(kernel='linear')
        # fixme
        # C = 50000
        # epsilon = .1
        # regression_model = SVR(kernel='linear', C=C, epsilon=epsilon)
        # regression_model = SVR(kernel='poly', degree=2, C=C, epsilon=epsilon)

        rfe = RFE(estimator=regression_model, n_features_to_select=10000, step=10000)

        pipeline = Pipeline([
            ('fill_missing', fill_missing),
            ('var_thr', var_thr),
            ('normalize', normalize),
            ('regression_model', regression_model),
            # ('rfe', rfe),
        ])

        if use_grid_search:
            params = {
                'regression_model__C': [.00001, .0001, .001, .01, .1, 1, 10, 100, 2000, 14450],
                # 'regression_model__epsilon': [0, .005, .01, .05, .1, 1, 5, 10],
            }
            pipe = GridSearchCV(pipeline, params, cv=5, scoring='mean_absolute_error', n_jobs=2)

        else:
            pipe = pipeline

        pipe.fit(X_train, y_train)
        y_predicted_train = pipe.predict(X_train)
        y_predicted = pipe.predict(X_test)

        df.ix[ind_train, ['pred_age_train']] = y_predicted_train
        df.ix[ind_test, ['pred_age_test']] = y_predicted

        test_mae = mean_absolute_error(y_test, y_predicted)
        test_r2 = r2_score(y_test, y_predicted)

        train_mae = mean_absolute_error(y_train, y_predicted_train)
        train_r2 = r2_score(y_train, y_predicted_train)

        # FIXME
        # {'C':C,'epislon':epsilon,'trainr2':train_r2,'trainmae':train_mae,'testr2':test_r2,'testmae':test_mae}
        # res=res.append({'C':C,'epislon':epsilon,'trainr2':train_r2,'trainmae':train_mae,'testr2':test_r2,'testmae':test_mae, 'alpha':np.max(np.abs(pipe.steps[3][1].dual_coef_))}, ignore_index=True)

        title_str = 'r2: {:.3f} MAE:{:.3f}'.format(test_r2, test_mae)
        scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)

        # performace results df
        df_res_out_file = os.path.abspath(data_str + '_df_results.txt')
        df_res = pd.DataFrame(
            {'FD_res': regress_confounds, 'r2_train': [train_r2], 'MAE_train': [train_mae], 'r2_test': [test_r2],
             'MAE_test': [test_mae]},
            columns=['FD_res', 'r2_train', 'MAE_train', 'r2_test', 'MAE_test'], index=[data_str])
        df_res.to_csv(df_res_out_file)

        # grid scores as textfile
        if use_grid_search:
            sorted_grid_score = sorted(pipe.grid_scores_, key=lambda x: x.mean_validation_score, reverse=True)
            score_str = [str(n) + ': ' + str(g) for n, g in enumerate(sorted_grid_score)]
            gs_text_file = os.path.abspath('gs_txt_' + data_str + '.txt')
            with open(gs_text_file, 'w') as f:
                f.write('\n'.join(score_str))
            scatter_file = [scatter_file, gs_text_file]

        brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)

        model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
        with open(model_out_file, 'w') as f:
            pickle.dump(pipe, f)

        df_use_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
        df.to_pickle(df_use_file)

        return scatter_file, brain_age_scatter_file, df_use_file, model_out_file, df_res_out_file

    prediction_split = Node(
        util.Function(
            input_names=['X_file',
                         'target_name',
                         'selection_criterium',
                         'df_file',
                         'data_str',
                         'regress_confounds',
                         'use_grid_search'],
            output_names=['scatter_file',
                          'brain_age_scatter_file',
                          'df_use_file',
                          'model_out_file',
                          'df_res_out_file'],
            function=run_prediction_split),
        name='prediction_split')
    the_in_node = prediction_split
    the_out_node_str = '01_split_'
    the_in_node.inputs.regress_confounds = False
    the_in_node.inputs.use_grid_search = False

    wf.connect(select_multimodal_X, 'X_multimodal_selected_file', the_in_node, 'X_file')
    wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    wf.connect(subject_selection_infosource, 'selection_criterium', the_in_node, 'selection_criterium')
    wf.connect(select_subjects, 'df_use_pickle_file', the_in_node, 'df_file')
    wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')

    wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    wf.connect(the_in_node, 'df_use_file', ds_pdf, the_out_node_str + 'predicted')
    wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')


    #
    # ###############################################################################################################
    # # BACKPROJECT PREDICTION WEIGHTS
    # # map weights back to single modality original format (e.g., nifti or matrix)
    #
    # def backproject_and_split_weights_fct(trained_model_file, backproject_info, data_str, target_name):
    #     from LeiCA_LIFE.learning.utils import backproject_weights_to_full_space, save_weights
    #
    #     data_str = target_name + '__' + data_str
    #
    #     out_file_list = []
    #     out_file_render_list = []
    #
    #     weights = backproject_weights_to_full_space(trained_model_file)
    #
    #     for m in backproject_info:
    #         out_data = weights[:, backproject_info[m]['start_index']:backproject_info[m]['end_index']]
    #         data_type = backproject_info[m]['data_type']
    #         save_template = backproject_info[m]['save_template']
    #         masker = backproject_info[m]['masker']
    #         out_name = data_str + '__' + m
    #
    #         out_file, out_file_render = save_weights(out_data, data_type[0], save_template[0], out_name, masker[0])
    #         out_file_list.append(out_file)
    #         out_file_render_list.append(out_file_render)
    #
    #     return out_file_list, out_file_render_list
    #
    # backproject_and_split_weights = Node(
    #     util.Function(input_names=['trained_model_file', 'backproject_info', 'data_str', 'target_name'],
    #                   output_names=['out_file_list', 'out_file_render_list'],
    #                   function=backproject_and_split_weights_fct),
    #     name='backproject_and_split_weights')
    #
    # the_from_node = prediction_split
    # the_in_node = backproject_and_split_weights
    # the_out_node_str = '01_split_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(load_info_dict, 'backproject_info', the_in_node, 'backproject_info')
    # wf.connect(load_info_dict, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
    #
    #
    #
    # ###############################################################################################################
    # # RUN PREDICTION FD regressed out
    # #
    #
    # prediction_split_regFD = prediction_split.clone('prediction_split_regFD')
    # the_in_node = prediction_split_regFD
    # the_out_node_str = '03_split_regFD'
    # the_in_node.inputs.regress_confounds = True
    # the_in_node.inputs.use_grid_search = False
    # wf.connect(select_subjects, 'X_selected_file', the_in_node, 'X_file')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(get_subjects_info, 'df_use_pickle_file', the_in_node, 'df_file')
    # wf.connect(load_info_dict, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_use_file', ds_pdf, the_out_node_str + 'predicted')
    # wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')
    #
    # backproject_and_split_regFD_weights = backproject_and_split_weights.clone('backproject_and_split_regFD_weights')
    # the_from_node = prediction_split_regFD
    # the_in_node = backproject_and_split_regFD_weights
    # the_out_node_str = '03_split_regFD_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(load_info_dict, 'backproject_info', the_in_node, 'backproject_info')
    # wf.connect(load_info_dict, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
    #
    #

    # #####################################
    # # TEST TEST TEST TET
    #
    # def run_prediction_strat_split(X_file, target_name, df_file, data_str, regress_confounds=False,
    #                                use_grid_search=False):
    #     import os, pickle
    #     import numpy as np
    #     import pandas as pd
    #     from sklearn.svm import SVR
    #     from sklearn.cross_validation import cross_val_score, cross_val_predict
    #     from sklearn.feature_selection import VarianceThreshold
    #     from sklearn.preprocessing import MinMaxScaler, StandardScaler
    #     from sklearn.preprocessing import Imputer
    #     from sklearn.pipeline import Pipeline
    #     from sklearn.metrics import r2_score, mean_absolute_error
    #     from sklearn.cross_validation import train_test_split
    #     from sklearn.grid_search import GridSearchCV
    #     # for some reason, only full path works
    #     from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age, residualize_group_data
    #     from sklearn.decomposition import PCA
    #     from sklearn.feature_selection import RFE
    #     from sklearn.cross_validation import StratifiedShuffleSplit
    #     data_str = target_name + '__' + data_str
    #
    #     df = pd.read_pickle(df_file)
    #     # add ouput cols to df
    #     df['split_group'] = ''
    #     df['pred_age_train'] = np.nan
    #     df['pred_age_test'] = np.nan
    #
    #     X = np.load(X_file)
    #
    #     y = df[[target_name]].values.squeeze()
    #
    #     confounds = df[['mean_FD_P']].values
    #
    #     ind = range(X.shape[0])
    #     n_bins = 10
    #     df['mean_FD_P_bins'] = pd.cut(df['mean_FD_P'], n_bins, labels=range(n_bins))
    #
    #     cv = StratifiedShuffleSplit(df['mean_FD_P_bins'].values, 5, test_size=0.5, random_state=0)
    #
    #
    #
    #
    #     # REGRESS OUT CONFOUNDS IF NEEDED
    #     if regress_confounds:
    #         X = residualize_group_data(X, confounds)
    #
    #     # PREPROCESSING
    #     fill_missing = Imputer()
    #     var_thr = VarianceThreshold()
    #     normalize = StandardScaler()
    #
    #     regression_model = SVR(kernel='linear')
    #
    #     pipeline = Pipeline([
    #         ('fill_missing', fill_missing),
    #         ('var_thr', var_thr),
    #         ('normalize', normalize),
    #         ('regression_model', regression_model),
    #         # ('rfe', rfe),
    #     ])
    #
    #     pipe = pipeline
    #
    #     #pipe.fit(X, y, cv=cv)
    #
    #     #print("The best parameters are %s with a score of %0.2f" % (pipe.best_params_, pipe.best_score_))
    #
    # prediction_strat_split = Node(
    #     util.Function(
    #         input_names=['X_file',
    #                      'target_name',
    #                      'df_file',
    #                      'data_str',
    #                      'regress_confounds',
    #                      'use_grid_search'],
    #         output_names=['scatter_file',
    #                       'brain_age_scatter_file',
    #                       'df_use_file',
    #                       'model_out_file',
    #                       'df_res_out_file'],
    #         function=run_prediction_strat_split),
    #     name='prediction_strat_split')
    # the_in_node = prediction_strat_split
    # the_out_node_str = '11_strat_split_'
    # the_in_node.inputs.regress_confounds = False
    # the_in_node.inputs.use_grid_search = False
    # wf.connect(select_subjects, 'X_selected_file', the_in_node, 'X_file')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(get_subjects_info, 'df_use_pickle_file', the_in_node, 'df_file')
    # wf.connect(load_info_dict, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_use_file', ds_pdf, the_out_node_str + 'predicted')
    # wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')

    #####################################
    # RUN WF
    #####################################
    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name)
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
