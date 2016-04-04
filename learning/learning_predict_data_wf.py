def learning_predict_data_wf(working_dir,
                             ds_dir,
                             in_data_name_list,
                             subjects_selection_crit_dict,
                             subjects_selection_crit_names_list,
                             aggregated_subjects_dir,
                             target_list,
                             use_n_procs,
                             plugin_name,
                             scaler=['standard'],
                             rfe=[False, True],
                             strat_split=[False],
                             confound_regression=[False, True]):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from itertools import chain
    from learning_utils import aggregate_multimodal_metrics_fct, run_prediction_split_fct, \
        backproject_and_split_weights_fct, select_subjects_fct, select_multimodal_X_fct
    import pandas as pd

    ###############################################################################################################
    # GENERAL SETTINGS

    wf = Workflow(name='learning_predict_data_wf')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'),
                      execution={'stop_on_first_crash': False,
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



    ###############################################################################################################
    # ensure in_data_name_list is list of lists
    in_data_name_list = [i if type(i) == list else [i] for i in in_data_name_list]
    in_data_name_list_unique = list(set(chain.from_iterable(in_data_name_list)))



    ###############################################################################################################
    # SET ITERATORS

    in_data_name_infosource = Node(util.IdentityInterface(fields=['in_data_name']), name='in_data_name_infosource')
    in_data_name_infosource.iterables = ('in_data_name', in_data_name_list_unique)

    multimodal_in_data_name_infosource = Node(util.IdentityInterface(fields=['multimodal_in_data_name']),
                                              name='multimodal_in_data_name_infosource')
    multimodal_in_data_name_infosource.iterables = ('multimodal_in_data_name', in_data_name_list)

    subject_selection_infosource = Node(util.IdentityInterface(fields=['selection_criterium']),
                                        name='subject_selection_infosource')
    subject_selection_infosource.iterables = ('selection_criterium', subjects_selection_crit_names_list)

    target_infosource = Node(util.IdentityInterface(fields=['target_name']), name='target_infosource')
    target_infosource.iterables = ('target_name', target_list)



    ###############################################################################################################
    # GET INFO AND SELECT FILES
    df_all_subjects_pickle_file = os.path.join(aggregated_subjects_dir, 'df_all_subjects_pickle_file/df_all.pkl')
    df = pd.read_pickle(df_all_subjects_pickle_file)

    # build lookup dict for unimodal data
    X_file_template = 'X_file/_in_data_name_{in_data_name}/vectorized_aggregated_data.npy'
    info_file_template = 'unimodal_backprojection_info_file/_in_data_name_{in_data_name}/unimodal_backprojection_info.pkl'
    unimodal_lookup_dict = {}
    for k in in_data_name_list_unique:
        unimodal_lookup_dict[k] = {'X_file': os.path.join(aggregated_subjects_dir, X_file_template.format(
            in_data_name=k)),
                                   'unimodal_backprojection_info_file': os.path.join(aggregated_subjects_dir,
                                                                                     info_file_template.format(
                                                                                         in_data_name=k))
                                   }



    ###############################################################################################################
    # AGGREGATE MULTIMODAL METRICS
    # stack single modality arrays horizontally
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
    # GET INDEXER FOR SUBJECTS OF INTEREST (as defined by selection criterium)
    select_subjects = Node(util.Function(input_names=['df_all_subjects_pickle_file',
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



    ###############################################################################################################
    # SELECT MULITMODAL X
    # select subjects (rows) from multimodal X according indexer
    select_multimodal_X = Node(util.Function(input_names=['X_multimodal_file', 'subjects_selection_index',
                                                          'selection_criterium'],
                                             output_names=['X_multimodal_selected_file'],
                                             function=select_multimodal_X_fct),
                               name='select_multimodal_X')
    wf.connect(aggregate_multimodal_metrics, 'X_multimodal_file', select_multimodal_X, 'X_multimodal_file')
    wf.connect(select_subjects, 'subjects_selection_index', select_multimodal_X, 'subjects_selection_index')



    ###############################################################################################################
    # RUN PREDICTION
    #
    prediction_node_dict = {}
    backprojection_node_dict = {}

    prediction_split = Node(util.Function(input_names=['X_file',
                                                       'target_name',
                                                       'selection_criterium',
                                                       'df_file',
                                                       'data_str',
                                                       'regress_confounds',
                                                       'use_grid_search',
                                                       'scaler',
                                                       'rfe',
                                                       'strat_split'],
                                          output_names=['scatter_file',
                                                        'brain_age_scatter_file',
                                                        'df_use_file',
                                                        'model_out_file',
                                                        'df_res_out_file'],
                                          function=run_prediction_split_fct),
                            name='prediction_split')

    backproject_and_split_weights = Node(util.Function(input_names=['trained_model_file',
                                                                    'multimodal_backprojection_info',
                                                                    'data_str',
                                                                    'target_name'],
                                                       output_names=['out_file_list',
                                                                     'out_file_render_list'],
                                                       function=backproject_and_split_weights_fct),
                                         name='backproject_and_split_weights')

    i = 0
    for s in scaler:
        for r in rfe:
            for strat in strat_split:
                for reg in confound_regression:
                    the_out_node_str = '%02d_scaler_%s_rfe_%s_strat_%s_reg_%s_' % (i, s, r, strat, reg)
                    prediction_node_dict[i] = prediction_split.clone(the_out_node_str)
                    the_in_node = prediction_node_dict[i]
                    the_in_node.inputs.use_grid_search = False
                    the_in_node.inputs.regress_confounds = reg
                    the_in_node.inputs.scaler = s
                    the_in_node.inputs.rfe = r
                    the_in_node.inputs.strat_split = strat

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

                    if not strat:  # backprojection with strat split is not possible, becaus no estimator is estimated
                        # BACKPROJECT PREDICTION WEIGHTS
                        # map weights back to single modality original format (e.g., nifti or matrix)
                        the_out_node_str = 'backprojection_%02d_scaler_%s_rfe_%s_strat_%s_reg_%s_' % (
                            i, s, r, strat, reg)
                        backprojection_node_dict[i] = backproject_and_split_weights.clone(the_out_node_str)
                        the_from_node = prediction_node_dict[i]
                        the_in_node = backprojection_node_dict[i]
                        wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
                        wf.connect(aggregate_multimodal_metrics, 'multimodal_backprojection_info', the_in_node,
                                   'multimodal_backprojection_info')
                        wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')
                        wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')

                        wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
                        wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')

                    i += 1


    # ###############################################################################################################
    # # BACKPROJECT PREDICTION WEIGHTS
    # # map weights back to single modality original format (e.g., nifti or matrix)
    # backproject_and_split_weights = Node(util.Function(input_names=['trained_model_file',
    #                                                                 'multimodal_backprojection_info', 'data_str',
    #                                                                 'target_name'],
    #                                                    output_names=['out_file_list', 'out_file_render_list'],
    #                                                    function=backproject_and_split_weights_fct),
    #                                      name='backproject_and_split_weights')
    #
    # the_from_node = prediction_split
    # the_in_node = backproject_and_split_weights
    # the_out_node_str = '01_split_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_backprojection_info', the_in_node,
    #            'multimodal_backprojection_info')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    #
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
    #
    #
    #
    # ###############################################################################################################
    # # RUN PREDICTION FD regressed out
    # prediction_split_regFD = prediction_split.clone('prediction_split_regFD')
    # the_in_node = prediction_split_regFD
    # the_out_node_str = '03_split_regFD_'
    # the_in_node.inputs.regress_confounds = True
    # the_in_node.inputs.use_grid_search = False
    #
    # wf.connect(select_multimodal_X, 'X_multimodal_selected_file', the_in_node, 'X_file')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(subject_selection_infosource, 'selection_criterium', the_in_node, 'selection_criterium')
    # wf.connect(select_subjects, 'df_use_pickle_file', the_in_node, 'df_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')
    #
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_use_file', ds_pdf, the_out_node_str + 'predicted')
    # wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')
    #
    #
    # ###############################################################################################################
    # # BACKPROJECT PREDICTION WEIGHTS FD regressed out
    # backproject_and_split_regFD_weights = backproject_and_split_weights.clone('backproject_and_split_regFD_weights')
    # the_from_node = prediction_split_regFD
    # the_in_node = backproject_and_split_regFD_weights
    # the_out_node_str = '03_split_regFD_weights_'
    # wf.connect(the_from_node, 'model_out_file', the_in_node, 'trained_model_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_backprojection_info', the_in_node,
    #            'multimodal_backprojection_info')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    #
    # wf.connect(the_in_node, 'out_file_list', ds_pdf, the_out_node_str + '.@weights')
    # wf.connect(the_in_node, 'out_file_render_list', ds_pdf, the_out_node_str + 'renders.@renders')
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

    # wf.connect(select_multimodal_X, 'X_multimodal_selected_file', the_in_node, 'X_file')
    # wf.connect(target_infosource, 'target_name', the_in_node, 'target_name')
    # wf.connect(subject_selection_infosource, 'selection_criterium', the_in_node, 'selection_criterium')
    # wf.connect(select_subjects, 'df_use_pickle_file', the_in_node, 'df_file')
    # wf.connect(aggregate_multimodal_metrics, 'multimodal_name', the_in_node, 'data_str')
    #
    # wf.connect(the_in_node, 'model_out_file', ds, the_out_node_str + 'trained_model')
    # wf.connect(the_in_node, 'scatter_file', ds_pdf, the_out_node_str + 'scatter')
    # wf.connect(the_in_node, 'brain_age_scatter_file', ds_pdf, the_out_node_str + 'brain_age_scatter')
    # wf.connect(the_in_node, 'df_use_file', ds_pdf, the_out_node_str + 'predicted')
    # wf.connect(the_in_node, 'df_res_out_file', ds_pdf, the_out_node_str + 'results_error')



    ###############################################################################################################
    #  RUN WF
    wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name)
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
