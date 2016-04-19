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
                             confound_regression=[False, True],
                             run_cv=False,
                             n_jobs_cv=1,
                             run_tuning=False):
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
                                                       'strat_split',
                                                       'run_cv',
                                                       'n_jobs_cv'],
                                          output_names=['scatter_file',
                                                        'brain_age_scatter_file',
                                                        'df_use_file',
                                                        'model_out_file',
                                                        'df_res_out_file',
                                                        'scatter_file_no_motion',
                                                        'scatter_file_random_motion',
                                                        'tuning_curve_file',
                                                        'scatter_file_cv'],
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
                    the_out_node_str = 'scaler_%s_rfe_%s_strat_%s_reg_%s_' % (s, r, strat, reg)
                    prediction_node_dict[i] = prediction_split.clone(the_out_node_str)
                    the_in_node = prediction_node_dict[i]
                    the_in_node.inputs.use_grid_search = False
                    the_in_node.inputs.regress_confounds = reg
                    the_in_node.inputs.scaler = s
                    the_in_node.inputs.rfe = r
                    the_in_node.inputs.strat_split = strat
                    the_in_node.inputs.run_cv = run_cv
                    the_in_node.inputs.n_jobs_cv = n_jobs_cv
                    the_in_node.inputs.run_tuning = run_tuning

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
                    wf.connect(the_in_node, 'scatter_file_no_motion', ds_pdf, the_out_node_str + 'scatter_motion.@no_mo')
                    wf.connect(the_in_node, 'scatter_file_random_motion', ds_pdf, the_out_node_str + 'scatter_motion.@rand_mo')
                    wf.connect(the_in_node, 'tuning_curve_file', ds_pdf, the_out_node_str + 'tuning_curve')
                    wf.connect(the_in_node, 'scatter_file_cv', ds_pdf, the_out_node_str + 'scatter_cv')

                    if not strat:  # backprojection with strat split is not possible, becaus no estimator is estimated
                        # BACKPROJECT PREDICTION WEIGHTS
                        # map weights back to single modality original format (e.g., nifti or matrix)
                        the_out_node_str = 'backprojection_scaler_%s_rfe_%s_strat_%s_reg_%s_' % (
                            s, r, strat, reg)
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



    ###############################################################################################################
    # #  RUN WF
    # wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    # wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    # wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name)
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})
