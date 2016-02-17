def learning_prepare_data_wf(working_dir,
                             ds_dir,
                             template_lookup_dict,
                             behav_file,
                             qc_file,
                             in_data_name_list,
                             data_lookup_dict,
                             subjects_selection_crit_dict,
                             subjects_selection_crit_names_list,
                             use_n_procs,
                             plugin_name):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow, MapNode, JoinNode
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from nipype.interfaces.freesurfer.utils import ImageInfo
    from utils import aggregate_data, vectorize_data
    from itertools import chain

    # ensure in_data_name_list is list of lists
    in_data_name_list = [i if type(i) == list else [i] for i in in_data_name_list]
    in_data_name_list_unique = list(set(chain.from_iterable(in_data_name_list)))


    #####################################
    # GENERAL SETTINGS
    #####################################
    wf = Workflow(name='learning_prepare_data_wf')
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

    mulitmodal_in_data_name_infosource = Node(util.IdentityInterface(fields=['multimodal_in_data_name']),
                                              name='mulitmodal_in_data_name_infosource')
    mulitmodal_in_data_name_infosource.iterables = ('multimodal_in_data_name', in_data_name_list)

    subject_selection_infosource = Node(util.IdentityInterface(fields=['selection_criterium']),
                                        name='subject_selection_infosource')
    subject_selection_infosource.iterables = ('selection_criterium', subjects_selection_crit_names_list)

    # fixme check if works, then delete
    # def out_name_str_fct(selection_criterium, in_data_name):
    #     return selection_criterium + '_' + in_data_name
    #
    # out_name_str = Node(util.Function(input_names=['selection_criterium', 'in_data_name'],
    #                                   output_names=['out_name_str'],
    #                                   function=out_name_str_fct),
    #                     name='out_name_str')
    # wf.connect(in_data_name_infosource, 'in_data_name', out_name_str, 'in_data_name')
    # wf.connect(subject_selection_infosource, 'selection_criterium', out_name_str, 'selection_criterium')

    def get_subjects_info_fct(behav_file, qc_file, subjects_selection_crit_dict, selection_criterium):
        import pandas as pd
        import os
        import numpy as np

        df = pd.read_pickle(behav_file)
        qc = pd.read_pickle(qc_file)
        df = qc.join(df, how='inner')

        #   EXCLUSION HERE:
        for eval_str in subjects_selection_crit_dict[selection_criterium]:
            df = eval(eval_str)

        df_out_file = os.path.join(os.getcwd(), 'df_use.csv')
        df.to_csv(df_out_file)
        df_out_pickle_file = os.path.join(os.getcwd(), 'df_use.pkl')
        df.to_pickle(df_out_pickle_file)

        subjects_list = df.index.values

        age = df.age.values
        age_file = os.path.join(os.getcwd(), 'age.npy')
        np.save(age_file, age)

        return age_file, df_out_file, df_out_pickle_file, subjects_list

    get_subjects_info = Node(
        util.Function(input_names=['behav_file', 'qc_file', 'subjects_selection_crit_dict', 'selection_criterium'],
                      output_names=['age_file', 'df_out_file', 'df_out_pickle_file', 'subjects_list'],
                      function=get_subjects_info_fct),
        name='get_subjects_info')
    get_subjects_info.inputs.behav_file = behav_file
    get_subjects_info.inputs.qc_file = qc_file
    get_subjects_info.inputs.subjects_selection_crit_dict = subjects_selection_crit_dict
    wf.connect(subject_selection_infosource, 'selection_criterium', get_subjects_info, 'selection_criterium')
    wf.connect(get_subjects_info, 'df_out_file', ds, 'test')
    wf.connect(get_subjects_info, 'age_file', ds, 'test_age_file')

    def create_file_list_fct(subjects_list, in_data_name, data_lookup_dict, template_lookup_dict):
        file_list = []
        for s in subjects_list:
            file_list.append(data_lookup_dict[in_data_name]['path_str'].format(subject_id=s))

        if 'matrix_name' in data_lookup_dict[in_data_name].keys():
            matrix_name = data_lookup_dict[in_data_name]['matrix_name']
        else:
            matrix_name = None

        if 'parcellation_path' in data_lookup_dict[in_data_name].keys():
            parcellation_path = data_lookup_dict[in_data_name]['parcellation_path']
        else:
            parcellation_path = None

        if 'fwhm' in data_lookup_dict[in_data_name].keys():
            fwhm = data_lookup_dict[in_data_name]['fwhm']
            if fwhm == 0:
                fwhm = None
        else:
            fwhm = None

        if 'mask_name' in data_lookup_dict[in_data_name].keys():
            mask_path = template_lookup_dict[data_lookup_dict[in_data_name]['mask_name']]
        else:
            mask_path = None

        if 'use_diagonal' in data_lookup_dict[in_data_name].keys():
            use_diagonal = data_lookup_dict[in_data_name]['use_diagonal']
        else:
            use_diagonal = False

        if 'use_fishers_z' in data_lookup_dict[in_data_name].keys():
            use_fishers_z = data_lookup_dict[in_data_name]['use_fishers_z']
        else:
            use_fishers_z = False

        return file_list, matrix_name, parcellation_path, fwhm, mask_path, use_diagonal, use_fishers_z

    create_file_list = Node(util.Function(input_names=['subjects_list',
                                                       'in_data_name',
                                                       'data_lookup_dict',
                                                       'template_lookup_dict',
                                                       ],
                                          output_names=['file_list',
                                                        'matrix_name',
                                                        'parcellation_path',
                                                        'fwhm',
                                                        'mask_path',
                                                        'use_diagonal',
                                                        'use_fishers_z'],
                                          function=create_file_list_fct),
                            name='create_file_list')
    wf.connect(get_subjects_info, 'subjects_list', create_file_list, 'subjects_list')
    wf.connect(in_data_name_infosource, 'in_data_name', create_file_list, 'in_data_name')
    create_file_list.inputs.data_lookup_dict = data_lookup_dict
    create_file_list.inputs.template_lookup_dict = template_lookup_dict

    aggregate_subjects = Node(util.Function(input_names=['file_list'],
                                            output_names=['merged_file', 'save_template'],
                                            function=aggregate_data),
                              name='aggregate_subjects')
    wf.connect(create_file_list, 'file_list', aggregate_subjects, 'file_list')

    vectorized_data = Node(
        util.Function(
            input_names=['in_data_file', 'mask_file', 'matrix_name', 'parcellation_path', 'fwhm', 'use_diagonal', 'use_fishers_z'],
            output_names=['vectorized_data', 'vectorized_data_file', 'data_type', 'masker'],
            function=vectorize_data),
        name='vectorized_data')
    wf.connect(aggregate_subjects, 'merged_file', vectorized_data, 'in_data_file')
    wf.connect(create_file_list, 'mask_path', vectorized_data, 'mask_file')
    wf.connect(create_file_list, 'matrix_name', vectorized_data, 'matrix_name')
    wf.connect(create_file_list, 'parcellation_path', vectorized_data, 'parcellation_path')
    wf.connect(create_file_list, 'fwhm', vectorized_data, 'fwhm')
    wf.connect(create_file_list, 'use_diagonal', vectorized_data, 'use_diagonal')
    wf.connect(create_file_list, 'use_fishers_z', vectorized_data, 'use_fishers_z')

    def aggregate_multimodal_metrics_fct(multimodal_list, vectorized_data_file, vectorized_data_names,
                                         selection_criterium, data_type_list, save_template_list, masker_list):
        import numpy as np
        import os
        backproject_info = {}

        metrics_index_list = [vectorized_data_names.index(m) for m in multimodal_list]
        X = None
        for i in metrics_index_list:
            subject_info = {}

            X_file = vectorized_data_file[i]
            X_single = np.load(X_file)
            if X is None:
                subject_info['start_index'] = 0
                X = X_single
            else:
                subject_info['start_index'] = X.shape[1]
                X = np.hstack((X, X_single))
            subject_info['end_index'] = X.shape[1]
            subject_info['data_type'] = data_type_list[i]

            subject_info['save_template'] = save_template_list[i]
            subject_info['masker'] = masker_list[i]

            backproject_info[vectorized_data_names[i]] = subject_info

        multimodal_in_name = '_'.join(multimodal_list)
        multimodal_out_name = selection_criterium + '_' + multimodal_in_name
        X_file = os.path.join(os.getcwd(), multimodal_in_name + '.npy')
        np.save(X_file, X)
        return multimodal_in_name, multimodal_out_name, X_file, backproject_info

    aggregate_multimodal_metrics = JoinNode(util.Function(input_names=['multimodal_list',
                                                                       'vectorized_data_file',
                                                                       'vectorized_data_names',
                                                                       'selection_criterium',
                                                                       'data_type_list',
                                                                       'save_template_list',
                                                                       'masker_list'],
                                                          output_names=['multimodal_in_name', 'multimodal_out_name',
                                                                        'X_file', 'backproject_info'],
                                                          function=aggregate_multimodal_metrics_fct),
                                            joinfield=['vectorized_data_file', 'data_type_list', 'save_template_list',
                                                       'masker_list'],
                                            joinsource=in_data_name_infosource,
                                            name='aggregate_multimodal_metrics')

    wf.connect(mulitmodal_in_data_name_infosource, 'multimodal_in_data_name', aggregate_multimodal_metrics,
               'multimodal_list')
    wf.connect(vectorized_data, 'vectorized_data_file', aggregate_multimodal_metrics, 'vectorized_data_file')
    wf.connect(subject_selection_infosource, 'selection_criterium', aggregate_multimodal_metrics, 'selection_criterium')
    wf.connect(vectorized_data, 'data_type', aggregate_multimodal_metrics, 'data_type_list')
    wf.connect(aggregate_subjects, 'save_template', aggregate_multimodal_metrics, 'save_template_list')
    wf.connect(vectorized_data, 'masker', aggregate_multimodal_metrics, 'masker_list')
    wf.connect(aggregate_multimodal_metrics, 'X_file', ds, 'multimodal_test')
    aggregate_multimodal_metrics.inputs.vectorized_data_names = in_data_name_list_unique







    #####################################
    # RUN PREDICTION
    #####################################

    def run_prediction_split(X_file, df_file, data_str, regress_confounds=False, use_grid_search=False):

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
        from sklearn.feature_selection import RFE

        df = pd.read_pickle(df_file)
        # add ouput cols to df
        df['split_group'] = ''
        df['pred_age_train'] = np.nan
        df['pred_age_test'] = np.nan

        X = np.load(X_file)
        # FIXME
        # X = np.concatenate((X,np.sqrt(X)), axis=1)

        y = df[['age']].values.squeeze()

        confounds = df[['mean_FD_P']].values

        ind = range(X.shape[0])
        X_train, X_test, y_train, y_test, confounds_train, confounds_test, ind_train, ind_test = train_test_split(X, y,
                                                                                                                  confounds,
                                                                                                                  ind,
                                                                                                                  test_size=0.5,
                                                                                                                  random_state=666)
        df['split_group'].iloc[ind_train] = 'train'
        df['split_group'].iloc[ind_test] = 'test'

        # REGRESS OUT CONFOUNDS IF NEEDED
        if regress_confounds:
            X_train = residualize_group_data(X_train, confounds_train)
            X_test = residualize_group_data(X_test, confounds_test)

        # PREPROCESSING
        fill_missing = Imputer()
        var_thr = VarianceThreshold()
        normalize = MinMaxScaler()

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

        df['pred_age_train'].iloc[ind_train] = y_predicted_train
        df['pred_age_test'].iloc[ind_test] = y_predicted

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

        df_out_file = os.path.join(os.getcwd(), data_str + '_df_predicted.pkl')
        df.to_pickle(df_out_file)

        return scatter_file, brain_age_scatter_file, df_out_file, model_out_file, df_res_out_file

    prediction_split = Node(
        util.Function(input_names=['X_file', 'df_file', 'data_str', 'regress_confounds', 'use_grid_search'],
                      output_names=['scatter_file', 'brain_age_scatter_file', 'df_out_file', 'model_out_file',
                                    'df_res_out_file'],
                      function=run_prediction_split),
        name='prediction_split')
    the_in_node = prediction_split
    the_out_node_str = '01_split_'
    the_in_node.inputs.regress_confounds = False
    the_in_node.inputs.use_grid_search = False
    wf.connect(aggregate_multimodal_metrics, 'X_file', the_in_node, 'X_file')
    wf.connect(get_subjects_info, 'df_out_pickle_file', the_in_node, 'df_file')
    wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    wf.connect(the_in_node, 'df_out_file', ds_pdf, the_out_node_str + 'predicted')
    wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')

    #####################################
    # Backproject prediction weights
    #####################################
    def backproject_and_split_weights_fct(trained_model_file, backproject_info, data_str):
        from LeiCA_LIFE.learning.utils import backproject_weights_to_full_space, save_weights

        out_file_list = []
        out_file_render_list = []

        weights = backproject_weights_to_full_space(trained_model_file)

        for m in backproject_info:
            out_data = weights[:, backproject_info[m]['start_index']:backproject_info[m]['end_index']]
            data_type = backproject_info[m]['data_type']
            save_template = backproject_info[m]['save_template']
            masker = backproject_info[m]['masker']
            out_name = data_str + '__' + m

            out_file, out_file_render = save_weights(out_data, data_type, save_template, out_name, masker)
            out_file_list.append(out_file)
            out_file_render_list.append(out_file_render)

        return out_file_list, out_file_render_list

    backproject_and_split_weights = Node(
        util.Function(input_names=['trained_model_file', 'backproject_info', 'data_str'],
                      output_names=['out_file_list', 'out_file_render_list'],
                      function=backproject_and_split_weights_fct),
        name='backproject_and_split_weights')

    the_from_node = prediction_split
    the_in_node = backproject_and_split_weights
    the_out_node_str = '01_split_weights_'
    wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    wf.connect(aggregate_multimodal_metrics, 'backproject_info', the_in_node, 'backproject_info')
    wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')

    #####################################
    # RUN PREDICTION FD regressed out
    #####################################
    prediction_split_regFD = prediction_split.clone('prediction_split_regFD')
    the_in_node = prediction_split_regFD
    the_out_node_str = '03_split_regFD'
    the_in_node.inputs.regress_confounds = True
    the_in_node.inputs.use_grid_search = False
    wf.connect(aggregate_multimodal_metrics, 'X_file', the_in_node, 'X_file')
    wf.connect(get_subjects_info, 'df_out_pickle_file', the_in_node, 'df_file')
    wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    wf.connect(the_in_node, 'df_out_file', ds_pdf, the_out_node_str + 'predicted')
    wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')

    backproject_and_split_regFD_weights = backproject_and_split_weights.clone('backproject_and_split_regFD_weights')

    the_from_node = prediction_split_regFD
    the_in_node = backproject_and_split_regFD_weights
    the_out_node_str = '03_split_regFD_weights_'
    wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    wf.connect(aggregate_multimodal_metrics, 'backproject_info', the_in_node, 'backproject_info')
    wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')



    #
    # #####################################
    # # RUN PREDICTION GS
    # #####################################
    # prediction_split_gs = prediction_split.clone('prediction_split_gs')
    #
    # the_in_node = prediction_split_gs
    # the_out_node_str = '02_split_gs_'
    # the_in_node.inputs.regress_confounds = False
    # the_in_node.inputs.use_grid_search = True
    # wf.connect(aggregate_multimodal_metrics, 'X_file', the_in_node, 'X_file')
    # wf.connect(get_subjects_info, 'df_out_pickle_file', the_in_node, 'df_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_out_file', ds_pdf, the_out_node_str + 'predicted')
    #
    #
    # backproject_and_split_gs_weights = backproject_and_split_weights.clone('backproject_and_split_gs_weights')
    #
    # the_from_node = prediction_split_gs
    # the_in_node = backproject_and_split_gs_weights
    # the_out_node_str = '02_split_gs_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(aggregate_multimodal_metrics, 'backproject_info', the_in_node, 'backproject_info')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
    #
    # #####################################
    # # RUN PREDICTION GS regFD
    # #####################################
    # prediction_split_gs_regFD = prediction_split.clone('prediction_split_gs_regFD')
    #
    # the_in_node = prediction_split_gs_regFD
    # the_out_node_str = '04_split_gs_regFD_'
    # the_in_node.inputs.regress_confounds = True
    # the_in_node.inputs.use_grid_search = True
    # wf.connect(aggregate_multimodal_metrics, 'X_file', the_in_node, 'X_file')
    # wf.connect(get_subjects_info, 'df_out_pickle_file', the_in_node, 'df_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_out_file', ds_pdf, the_out_node_str + 'predicted')
    #
    #
    # backproject_and_split_gs_regFD_weights = backproject_and_split_weights.clone('backproject_and_split_gs_regFD_weights')
    #
    # the_from_node = prediction_split_gs_regFD
    # the_in_node = backproject_and_split_gs_regFD_weights
    # the_out_node_str = '04_split_gs_regFD_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(aggregate_multimodal_metrics, 'backproject_info', the_in_node, 'backproject_info')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', the_in_node, 'data_str')
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
    #














    #
    # def run_prediction_split_gs(X_file, y_file, data_str, regress_confounds=False, confounds=None):
    #
    #     import os, pickle
    #     import numpy as np
    #     from sklearn.svm import SVR
    #     from sklearn.cross_validation import cross_val_score, cross_val_predict
    #     from sklearn.grid_search import GridSearchCV
    #     from sklearn.feature_selection import VarianceThreshold
    #     from sklearn.preprocessing import MinMaxScaler
    #     from sklearn.preprocessing import Imputer
    #     from sklearn.pipeline import Pipeline
    #     from sklearn.metrics import r2_score, mean_absolute_error
    #     from sklearn.cross_validation import train_test_split
    #     from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age, residualize_group_data
    #
    #     X = np.load(X_file)
    #     # FIXME
    #     # X = np.concatenate((X,np.sqrt(X)), axis=1)
    #     y = np.load(y_file)
    #
    #     X_train, X_test, y_train, y_test, confounds_train, confounds_test = train_test_split(X, y, confounds,
    #                                                                                          test_size=0.5,
    #                                                                                          random_state=666)
    #     if regress_confounds:
    #         X_train = residualize_group_data(X_train, confounds_train)
    #         X_test = residualize_group_data(X_test, confounds_test)
    #
    #     fill_missing = Imputer()
    #     var_thr = VarianceThreshold()
    #     normalize = MinMaxScaler()
    #
    #     regression_model = SVR(kernel='linear')
    #
    #     pipe = Pipeline([
    #         ('fill_missing', fill_missing),
    #         ('var_thr', var_thr),
    #         ('normalize', normalize),
    #         ('regression_model', regression_model),
    #     ])
    #
    #     # GRID SEARCH
    #     params = {
    #         'regression_model__C': [1, 98, 106, 110, 130, 150, 170, 200, 2000, 14450],
    #         # [.0001, .001, .01, .1, 1, 10, 20, 30],
    #         'regression_model__epsilon': [0, .005, .01, .05, .1, 1, 5, 10],
    #         # 'regression_model__nu': [0.01, .25, .5, .75, .8, .85, .9, .95, .99],
    #     }
    #     # fixme njobs
    #     gs = GridSearchCV(pipe, params, cv=5, scoring='mean_absolute_error', n_jobs=2)
    #     gs.fit(X_train, y_train)
    #
    #     # pipe.fit(X_train, y_train)
    #
    #     y_predicted = gs.predict(X_test)
    #
    #     error_mae = mean_absolute_error(y_test, y_predicted)
    #     error_r2 = r2_score(y_test, y_predicted)
    #     title_str = 'r2: {:.3f} MAE:{:.3f}'.format(error_r2, error_mae)
    #     scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)
    #
    #     brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)
    #
    #     model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
    #     with open(model_out_file, 'w') as f:
    #         pickle.dump(pipe, f)
    #
    #     y_pred_file = os.path.join(os.getcwd(), data_str + '_y_pred.npy')
    #     np.save(y_pred_file, y_predicted)
    #
    #     return scatter_file, brain_age_scatter_file, y_pred_file, model_out_file

    # prediction_split_gs = Node(
    #     util.Function(input_names=['X_file', 'y_file', 'data_str', 'regress_confounds', 'confounds'],
    #                   output_names=['scatter_file', 'brain_age_scatter_file', 'y_pred_file', 'model_out_file'],
    #                   function=run_prediction_split_gs),
    #     name='prediction_split_gs')
    # prediction_split_gs.inputs.regress_confounds = False
    # wf.connect(aggregate_multimodal_metrics, 'X_file', prediction_split_gs, 'X_file')
    # wf.connect(get_subjects_info, 'age_file', prediction_split_gs, 'y_file')
    # wf.connect(get_subjects_info, 'confounds', prediction_split_gs, 'confounds')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', prediction_split_gs, 'data_str')
    # wf.connect(prediction_split_gs, 'model_out_file', ds, 'test_split_trained_model_gs')
    # wf.connect(prediction_split_gs, 'scatter_file', ds_pdf, 'test_split_scatter_gs')
    # wf.connect(prediction_split_gs, 'brain_age_scatter_file', ds_pdf, 'test_split_brain_age_scatter_gs')
    # wf.connect(prediction_split_gs, 'y_pred_file', ds_pdf, 'test_split_predicted_gs')
    #

    #####################################
    # RUN PREDICTION GS FD regressed out
    #####################################
    # prediction_split_gs_regFD = Node(
    #     util.Function(input_names=['X_file', 'y_file', 'data_str', 'regress_confounds', 'confounds'],
    #                   output_names=['scatter_file', 'brain_age_scatter_file', 'y_pred_file', 'model_out_file'],
    #                   function=run_prediction_split_gs),
    #     name='prediction_split_gs_regFD')
    # prediction_split_gs_regFD.inputs.regress_confounds = True
    # wf.connect(aggregate_multimodal_metrics, 'X_file', prediction_split_gs_regFD, 'X_file')
    # wf.connect(get_subjects_info, 'age_file', prediction_split_gs_regFD, 'y_file')
    # wf.connect(get_subjects_info, 'confounds', prediction_split_gs_regFD, 'confounds')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', prediction_split_gs_regFD, 'data_str')
    # wf.connect(prediction_split_gs_regFD, 'model_out_file', ds, 'test_split_trained_model_gs_regFD')
    # wf.connect(prediction_split_gs_regFD, 'scatter_file', ds_pdf, 'test_split_scatter_gs_regFD')
    # wf.connect(prediction_split_gs_regFD, 'brain_age_scatter_file', ds_pdf, 'test_split_brain_age_scatter_gs_regFD')
    # wf.connect(prediction_split_gs_regFD, 'y_pred_file', ds_pdf, 'test_split_predicted_gs_regFD')
    #

    #####################################
    # RUN PREDICTION ENET
    #####################################
    #
    # def run_prediction_split_enet(X_file, y_file, data_str):
    #
    #     import os, pickle
    #     import numpy as np
    #     from sklearn.svm import SVR
    #     from sklearn.linear_model import ElasticNetCV
    #     from sklearn.cross_validation import cross_val_score, cross_val_predict
    #     from sklearn.feature_selection import VarianceThreshold
    #     from sklearn.preprocessing import MinMaxScaler
    #     from sklearn.preprocessing import Imputer
    #     from sklearn.pipeline import Pipeline
    #     from sklearn.metrics import r2_score, mean_absolute_error
    #     from sklearn.cross_validation import train_test_split
    #     from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age
    #
    #     X = np.load(X_file)
    #     # FIXME
    #     # X = np.concatenate((X,np.sqrt(X)), axis=1)
    #     y = np.load(y_file)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=666)
    #
    #     fill_missing = Imputer()
    #     var_thr = VarianceThreshold()
    #     normalize = MinMaxScaler()
    #
    #     regression_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], n_jobs=3)
    #     pipe = Pipeline([
    #         ('fill_missing', fill_missing),
    #         ('var_thr', var_thr),
    #         ('normalize', normalize),
    #         ('regression_model', regression_model),
    #     ])
    #
    #     pipe.fit(X_train, y_train)
    #
    #     y_predicted = pipe.predict(X_test)
    #
    #     error_mae = mean_absolute_error(y_test, y_predicted)
    #     error_r2 = r2_score(y_test, y_predicted)
    #     title_str = 'r2: {:.3f} MAE:{:.3f}'.format(error_r2, error_mae)
    #     scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)
    #
    #     brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)
    #
    #     model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
    #     with open(model_out_file, 'w') as f:
    #         pickle.dump(pipe, f)
    #
    #     y_pred_file = os.path.join(os.getcwd(), data_str + '_y_pred.npy')
    #     np.save(y_pred_file, y_predicted)
    #
    #     return scatter_file, brain_age_scatter_file, y_pred_file, model_out_file
    #
    # prediction_split_enet = Node(util.Function(input_names=['X_file', 'y_file', 'data_str'],
    #                                            output_names=['scatter_file', 'brain_age_scatter_file', 'y_pred_file',
    #                                                          'model_out_file'],
    #                                            function=run_prediction_split_enet),
    #                              name='prediction_split_enet')
    # #
    # wf.connect(aggregate_multimodal_metrics, 'X_file', prediction_split_enet, 'X_file')
    # wf.connect(get_subjects_info, 'age_file', prediction_split_enet, 'y_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', prediction_split_enet, 'data_str')
    # wf.connect(prediction_split_enet, 'model_out_file', ds, 'test_split_trained_model_enet')
    # wf.connect(prediction_split_enet, 'scatter_file', ds_pdf, 'test_split_scatter_enet')
    # wf.connect(prediction_split_enet, 'brain_age_scatter_file', ds_pdf, 'test_split_brain_age_scatter_enet')
    # wf.connect(prediction_split_enet, 'y_pred_file', ds_pdf, 'test_split_predicted_enet')
    #
    #


    #####################################
    # RUN PREDICTION ENET
    #####################################
    #
    # def run_prediction_split_gbr(X_file, y_file, data_str):
    #
    #     import os, pickle
    #     import numpy as np
    #     from sklearn.svm import SVR
    #     from sklearn.ensemble import GradientBoostingRegressor
    #     from sklearn.cross_validation import cross_val_score, cross_val_predict
    #     from sklearn.feature_selection import VarianceThreshold
    #     from sklearn.preprocessing import MinMaxScaler
    #     from sklearn.preprocessing import Imputer
    #     from sklearn.pipeline import Pipeline
    #     from sklearn.metrics import r2_score, mean_absolute_error
    #     from sklearn.cross_validation import train_test_split
    #     from LeiCA_LIFE.learning.utils import pred_real_scatter, plot_brain_age
    #
    #     X = np.load(X_file)
    #     # FIXME
    #     # X = np.concatenate((X,np.sqrt(X)), axis=1)
    #     y = np.load(y_file)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=666)
    #
    #     fill_missing = Imputer()
    #     var_thr = VarianceThreshold()
    #     normalize = MinMaxScaler()
    #
    #     regression_model = GradientBoostingRegressor()
    #     pipe = Pipeline([
    #         ('fill_missing', fill_missing),
    #         ('var_thr', var_thr),
    #         ('normalize', normalize),
    #         ('regression_model', regression_model),
    #     ])
    #
    #     pipe.fit(X_train, y_train)
    #
    #     y_predicted = pipe.predict(X_test)
    #
    #     error_mae = mean_absolute_error(y_test, y_predicted)
    #     error_r2 = r2_score(y_test, y_predicted)
    #     title_str = 'r2: {:.3f} MAE:{:.3f}'.format(error_r2, error_mae)
    #     scatter_file = pred_real_scatter(y_test, y_predicted, title_str, data_str)
    #
    #     brain_age_scatter_file = plot_brain_age(y_test, y_predicted, data_str)
    #
    #     model_out_file = os.path.join(os.getcwd(), 'trained_model.pkl')
    #     with open(model_out_file, 'w') as f:
    #         pickle.dump(pipe, f)
    #
    #     y_pred_file = os.path.join(os.getcwd(), data_str + '_y_pred.npy')
    #     np.save(y_pred_file, y_predicted)
    #
    #     return scatter_file, brain_age_scatter_file, y_pred_file, model_out_file
    #
    # prediction_split_bgr = Node(util.Function(input_names=['X_file', 'y_file', 'data_str'],
    #                                           output_names=['scatter_file', 'brain_age_scatter_file', 'y_pred_file',
    #                                                         'model_out_file'],
    #                                           function=run_prediction_split_gbr),
    #                             name='prediction_split_bgr')

    # wf.connect(aggregate_multimodal_metrics, 'X_file', prediction_split_bgr, 'X_file')
    # wf.connect(get_subjects_info, 'age_file', prediction_split_bgr, 'y_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', prediction_split_bgr, 'data_str')
    # wf.connect(prediction_split_bgr, 'model_out_file', ds, 'test_split_trained_model_gbr')
    # wf.connect(prediction_split_bgr, 'scatter_file', ds_pdf, 'test_split_scatter_gbr')
    # wf.connect(prediction_split_bgr, 'brain_age_scatter_file', ds_pdf, 'test_split_brain_age_scatter_gbr')
    # wf.connect(prediction_split_bgr, 'y_pred_file', ds_pdf, 'test_split_predicted_gbr')
    #
    #

    # #####################################
    # # LEARNING CURVE
    # #####################################
    # from learning_curve import learning_curve_fct
    # learning_curve_svr = Node(util.Function(input_names=['X_file', 'y_file', 'out_name', 'model_file'],
    #                                         output_names=['curve_file'],
    #                                         function=learning_curve_fct),
    #                           name='learning_curve_svr')
    #
    # wf.connect(aggregate_multimodal_metrics, 'X_file', learning_curve_svr, 'X_file')
    # wf.connect(get_subjects_info, 'age_file', learning_curve_svr, 'y_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_out_name', learning_curve_svr, 'out_name')
    # wf.connect(prediction_split, 'model_out_file', learning_curve_svr, 'model_file')
    # wf.connect(learning_curve_svr, 'curve_file', ds_pdf, 'learning_curve_svr')
    #


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
