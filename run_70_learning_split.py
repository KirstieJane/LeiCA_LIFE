__author__ = 'franzliem'

import os
import pandas as pd

# # LeiCA modules
from learning.learning_prepare_data_wf import learning_prepare_data_wf

from variables import wd_root_path, ds_root_path, data_lookup_dict, template_lookup_dict
from variables import behav_file, qc_file

working_dir = os.path.join(wd_root_path, 'wd_learning_test')
ds_dir = os.path.join(ds_root_path, 'learning_out_test')

use_n_procs = 25
plugin_name = 'MultiProc'



# # # # # # # # # #
# subject selection criteria
# # # # # # # # # #
subjects_selection_crit_dict = {}
# subjects_selection_crit_dict['adult_healthy_F'] = ["df[df.sex == \'F\']", 'df[df.no_axis_1]', 'df[df.age >= 18]']
# subjects_selection_crit_dict['adult_F'] = ["df[df.sex == \'F\']", 'df[df.age >= 18]']
subjects_selection_crit_dict['bothSexes'] = ["df[df.n_TRs>294]"]
subjects_selection_crit_dict['bothSexes_FD01_035'] = ["df[df.n_TRs>294]", "df[df.mean_FD_P<.35]", "df[df.mean_FD_P>.1]"]
subjects_selection_crit_dict['bothSexes_FD01_035_age50plus'] = ["df[df.n_TRs>294]", "df[df.mean_FD_P<.35]",
                                                                "df[df.mean_FD_P>.1]", "df[df.age>50]"]

subjects_selection_crit_dict['F'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]"]
subjects_selection_crit_dict['F_fd04'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.mean_FD_P<.4]"]
subjects_selection_crit_dict['F_fd01'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.mean_FD_P<.1]"]
subjects_selection_crit_dict['F_fd_med_02_1'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.mean_FD_P<1]",
                                                 "df[df.mean_FD_P>.2]"]
subjects_selection_crit_dict['F_fd_med_03_1'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.mean_FD_P<1]",
                                                 "df[df.mean_FD_P>.3]"]
subjects_selection_crit_dict['F_random_sample'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.random_sample]"]
subjects_selection_crit_dict['F_uniform_sample'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.uniform_sample]"]

subjects_selection_crit_dict['F_young'] = ["df[df.sex == \'F\']", "df[df.n_TRs>294]", "df[df.age<55]"]

subjects_selection_crit_dict['M'] = ["df[df.sex == \'M\']", "df[df.n_TRs>294]"]
subjects_selection_crit_dict['M_fd04'] = ["df[df.sex == \'M\']", "df[df.n_TRs>294]", "df[df.mean_FD_P<.4]"]
subjects_selection_crit_dict['M_young'] = ["df[df.sex == \'M\']", "df[df.n_TRs>294]", "df[df.age<55]"]

subjects_selection_crit_dict['bothSexes_young'] = ["df[df.n_TRs>294]", "df[df.age<55]"]

subjects_selection_crit_dict['random_sample'] = ["df[df.n_TRs>294]", "df[df.random_sample]"]
subjects_selection_crit_dict['uniform_sample'] = ["df[df.n_TRs>294]", "df[df.uniform_sample]"]

# subjects_selection_crit_names_list = ['F', 'F_fd04', 'F_fd01', 'F_fd_med_02_1', 'F_fd_med_03_1']#, 'M', 'F_young', 'M_young', 'bothSexes_young']
# subjects_selection_crit_names_list = [
#     'bothSexes_FD01_035', 'bothSexes_FD01_035_age50plus']  # ['F', 'random_sample', 'uniform_sample']#, 'M', 'F_young', 'M_young', 'bothSexes_young']

# in_data_name_list = [['lh_ct', 'rh_ct'], ['lh_csa', 'rh_csa'], ['aseg'], ['alff'], ['falff'], ['reho'], ['variability_std'],
#                      ['craddock_788_noBP'], ['craddock_788_BP'], ['gordon_noBP'], ['gordon_788_BP'],
#                      # ['falff', 'alff'],
#                      ['lh_ct', 'rh_ct', 'lh_csa', 'rh_csa', 'aseg'],
#                      ['lh_ct', 'rh_ct', 'lh_csa', 'rh_csa', 'aseg', 'alff'],
#                      ['lh_ct', 'rh_ct', 'lh_csa', 'rh_csa', 'aseg', 'falff', 'alff', 'reho', 'variability_std']]  # , ['falff'], ['alff_craddock_788']]
# in_data_name_list = [['alff_GM_3mm'], ['alff_GM_4mm'], ['alff_GM_8mm'],['alff_GM_WM_3mm'], ['alff_GM_WM_4mm'], ['alff_GM_WM_8mm'], ['alff_brain_mask_3mm'], ['alff_brain_mask_4mm'], ['alff_brain_mask_8mm'], ['alff_brain_mask_3mm_frauke']]
in_data_name_list = [
    ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0'], ['lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0'], ['aseg'],
    ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg'],
    ['lh_ct_fsav5_sm10', 'rh_ct_fsav5_sm10'], ['lh_csa_fsav5_sm10', 'rh_csa_fsav5_sm10'],
    ['lh_ct_fsav5_sm10', 'rh_ct_fsav5_sm10', 'lh_csa_fsav5_sm10', 'rh_csa_fsav5_sm10', 'aseg'],
    ['alff_GM_WM_4mm_sm0'], ['falff_GM_WM_4mm_sm0'], ['alff_z_GM_WM_4mm_sm0'], ['falff_z_GM_WM_4mm_sm0'],
    ['alff_GM_WM_4mm_sm8'], ['falff_GM_WM_4mm_sm8'], ['alff_z_GM_WM_4mm_sm8'], ['falff_z_GM_WM_4mm_sm8'],
    ['reho_GM_WM_4mm_sm0'], ['reho_GM_WM_4mm_sm8'],
    ['craddock_205_BP'], ['craddock_788_BP'], ['gordon_BP'], ['gordon_BP_ds'],

    #                    ['aseg'], ['aparc_lh_thickness', 'aparc_rh_thickness'], ['aparc_lh_area', 'aparc_rh_area'],
    #                    ['aseg', 'aparc_lh_thickness', 'aparc_rh_thickness', 'aparc_lh_area', 'aparc_rh_area'],
    #                    # ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0'], ['lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0'], ['aseg'],
    #                    # ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0', 'lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0', 'aseg'],
    #                    # ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0', 'lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0', 'aseg', 'alff_z_GM_WM_4mm'],
]

subjects_selection_crit_names_list = ['bothSexes']
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
learning_prepare_data_wf(working_dir=working_dir,
                         ds_dir=ds_dir,
                         template_lookup_dict=template_lookup_dict,
                         behav_file=behav_file,
                         qc_file=qc_file,
                         in_data_name_list=in_data_name_list,
                         data_lookup_dict=data_lookup_dict,
                         subjects_selection_crit_dict=subjects_selection_crit_dict,
                         subjects_selection_crit_names_list=subjects_selection_crit_names_list,
                         use_n_procs=use_n_procs,
                         plugin_name=plugin_name)
