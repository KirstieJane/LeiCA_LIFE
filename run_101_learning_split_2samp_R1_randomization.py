# run with python run_101_learning_split_2samp_R1_randomization.py i_randomization

import os, sys

# # LeiCA modules
from learning.learning_predict_data_wf import learning_predict_data_2samp_wf
from learning.learning_variables import in_data_name_list, subjects_selection_crit_dict, \
    subjects_selection_crit_names_list, target_list

i_randomization = int(sys.argv[1])


use_n_procs = 3
plugin_name = 'MultiProc'
#n_randomizations = 2

working_dir = '/data/liem-3/LeiCA_LIFE_learning_wd/wd_learning'
ds_dir = '/scr/adenauer2/Franz/LeiCA_LIFE/learning_out/training_life_only'
aggregated_subjects_dir = '/scr/adenauer2/Franz/LeiCA_LIFE/learning_out/vectorized_aggregated_data'

aggregated_subjects_dir_nki = '/scr/adenauer2/Franz/LeiCA_NKI/learning_out/vectorized_aggregated_data'
subjects_selection_crit_dict_nki = {}
subjects_selection_crit_dict_nki['adult'] = ["age >= 18", "n_TRs > 890"]
subjects_selection_crit_name_nki = 'adult'
#
# # LIFE only training
# run_2sample_training = False
# learning_predict_data_2samp_wf(working_dir=working_dir,
#                                ds_dir=ds_dir,
#                                in_data_name_list=in_data_name_list,
#                                subjects_selection_crit_dict=subjects_selection_crit_dict,
#                                subjects_selection_crit_names_list=subjects_selection_crit_names_list,
#                                aggregated_subjects_dir=aggregated_subjects_dir,
#                                target_list=target_list,
#                                use_n_procs=use_n_procs,
#                                plugin_name=plugin_name,
#                                confound_regression=[False, True],
#                                run_cv=True,
#                                n_jobs_cv=5,
#                                run_tuning=False,
#                                run_2sample_training=run_2sample_training,
#                                aggregated_subjects_dir_nki=aggregated_subjects_dir_nki,
#                                subjects_selection_crit_dict_nki=subjects_selection_crit_dict_nki,
#                                subjects_selection_crit_name_nki=subjects_selection_crit_name_nki)
#
#


# LIFE + NKI training

#for i_randomization in range(n_randomizations):
working_dir = '/data/liem-3/LeiCA_LIFE_learning_wd/wd_learning_2samp_randomizations/rand_' + str(i_randomization)
ds_dir = '/scr/adenauer2/Franz/LeiCA_LIFE/learning_out/training_2samp_randomizations/rand_' + str(i_randomization)
run_2sample_training = True
learning_predict_data_2samp_wf(working_dir=working_dir,
                               ds_dir=ds_dir,
                               in_data_name_list=in_data_name_list,
                               subjects_selection_crit_dict=subjects_selection_crit_dict,
                               subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                               aggregated_subjects_dir=aggregated_subjects_dir,
                               target_list=target_list,
                               use_n_procs=use_n_procs,
                               plugin_name=plugin_name,
                               confound_regression=[False], #[False, True],
                               run_cv=False, #True
                               n_jobs_cv=5,
                               run_tuning=False,
                               run_2sample_training=run_2sample_training,
                               aggregated_subjects_dir_nki=aggregated_subjects_dir_nki,
                               subjects_selection_crit_dict_nki=subjects_selection_crit_dict_nki,
                               subjects_selection_crit_name_nki=subjects_selection_crit_name_nki,
                               random_state_nki=i_randomization)
