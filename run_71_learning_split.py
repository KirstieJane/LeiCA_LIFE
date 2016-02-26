import os
import pandas as pd

# # LeiCA modules
from learning.learning_predict_data_wf import learning_predict_data_wf
from learning.learning_variables import in_data_name_list, subjects_selection_crit_dict, \
    subjects_selection_crit_names_list, target_list
from variables import wd_root_path, ds_root_path, data_lookup_dict, template_lookup_dict
from variables import behav_file, qc_file

wd_root_path = '/scr/adenauer1/Franz/LeiCA_LIFE_learning_wd'
ds_root_path = '/scr/adenauer2/Franz/LeiCA_LIFE'

working_dir = os.path.join(wd_root_path, 'wd_learning')
ds_dir = os.path.join(ds_root_path, 'learning_out')
aggregated_subjects_dir = os.path.join(ds_dir, 'vectorized_aggregated_data')


use_n_procs = 10
plugin_name = 'MultiProc'


# fixme
in_data_name_list = [['lh_ct_fsav4_sm0'], ['aseg']]
import warnings

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
learning_predict_data_wf(working_dir=working_dir,
                         ds_dir=ds_dir,
                         in_data_name_list=in_data_name_list,
                         subjects_selection_crit_dict=subjects_selection_crit_dict,
                         subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                         aggregated_subjects_dir=aggregated_subjects_dir,
                         target_list=target_list,
                         use_n_procs=use_n_procs,
                         plugin_name=plugin_name)
