import os
import pandas as pd

# # LeiCA modules
from LeiCA_LIFE.learning.learning_predict_data_from_trained_model_wf import learning_predict_data_wf

from learning.learning_variables import in_data_name_list, subjects_selection_crit_dict, target_list
from variables import wd_root_path, ds_root_path

wd_root_path = '/data/liem-3/LeiCA_LIFE_learning_nonorm_wd'
ds_root_path = '/scr/adenauer2/Franz/LeiCA_LIFE/'


subjects_selection_crit_names_list =['bothSexes_neuH_FD06_nonorm']


working_dir = os.path.join(wd_root_path, 'wd_learning')
ds_dir = os.path.join(ds_root_path, 'learning_out_nonorm')
aggregated_subjects_dir = os.path.join('/scr/adenauer2/Franz/LeiCA_LIFE/learning_out', 'vectorized_aggregated_data')
trained_model_dir = '/scr/adenauer2/Franz/LeiCA_LIFE'

use_n_procs = 50
plugin_name = 'MultiProc'


trained_model_template = {
    'trained_model': 'learning_out/group_learning_prepare_data/{ana_stream}trained_model/' +
                     '_multimodal_in_data_name_{multimodal_in_data_name}/_selection_criterium_bothSexes_neuH_FD06_norm/' +
                     '_target_name_{target_name}/trained_model.pkl'}

learning_predict_data_wf(working_dir=working_dir,
                         ds_dir=ds_dir,
                         trained_model_dir=trained_model_dir,
                         in_data_name_list=in_data_name_list,
                         subjects_selection_crit_dict=subjects_selection_crit_dict,
                         subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                         aggregated_subjects_dir=aggregated_subjects_dir,
                         target_list=target_list,
                         trained_model_template=trained_model_template,
                         use_n_procs=use_n_procs,
                         plugin_name=plugin_name,
                         confound_regression=[False])
