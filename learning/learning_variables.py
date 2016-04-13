# # # # # # # # # #
# subject selection criteria
# # # # # # # # # #
subjects_selection_crit_dict = {}
# d['a'] = ["a < 3", "b > 2"]
# NEW Style!
subjects_selection_crit_dict['bothSexes'] = ["n_TRs > 294"]
subjects_selection_crit_dict['bothSexes_neuH'] = ["n_TRs > 294", "neurol_healthy==True"]

# full list
# in_data_name_list = [
#     ['aseg'],
#
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0'], ['lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg'],
#     ['lh_ct_fsav5_sm10', 'rh_ct_fsav5_sm10'], ['lh_csa_fsav5_sm10', 'rh_csa_fsav5_sm10'],
#     ['lh_ct_fsav5_sm10', 'rh_ct_fsav5_sm10', 'lh_csa_fsav5_sm10', 'rh_csa_fsav5_sm10', 'aseg'],
#     #
#     ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0'], ['lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0'],
#     ['lh_ct_fsav4_sm0', 'rh_ct_fsav4_sm0', 'lh_csa_fsav4_sm0', 'rh_csa_fsav4_sm0', 'aseg'],
#     ['lh_ct_fsav4_sm10', 'rh_ct_fsav4_sm10'], ['lh_csa_fsav4_sm10', 'rh_csa_fsav4_sm10'],
#     ['lh_ct_fsav4_sm10', 'rh_ct_fsav4_sm10', 'lh_csa_fsav4_sm10', 'rh_csa_fsav4_sm10', 'aseg'],
#     #
#     ['lh_ct_fsav3_sm0', 'rh_ct_fsav3_sm0'], ['lh_csa_fsav3_sm0', 'rh_csa_fsav3_sm0'],
#     ['lh_ct_fsav3_sm0', 'rh_ct_fsav3_sm0', 'lh_csa_fsav3_sm0', 'rh_csa_fsav3_sm0', 'aseg'],
#     ['lh_ct_fsav3_sm10', 'rh_ct_fsav3_sm10'], ['lh_csa_fsav3_sm10', 'rh_csa_fsav3_sm10'],
#     ['lh_ct_fsav3_sm10', 'rh_ct_fsav3_sm10', 'lh_csa_fsav3_sm10', 'rh_csa_fsav3_sm10', 'aseg'],
#     #
#     ['alff_z_GM_WM_3mm_sm0'], ['alff_z_GM_WM_4mm_sm0'], ['alff_z_GM_WM_8mm_sm0'],
#     ['alff_z_GM_WM_3mm_sm8'], ['alff_z_GM_WM_4mm_sm8'], ['alff_z_GM_WM_8mm_sm8'],
#     ['alff_z_GM_3mm_sm0'], ['alff_z_GM_4mm_sm0'], ['alff_z_GM_8mm_sm0'],
#     ['alff_z_GM_3mm_sm8'], ['alff_z_GM_4mm_sm8'], ['alff_z_GM_8mm_sm8'],
#     #
#     ['falff_z_GM_WM_3mm_sm0'], ['falff_z_GM_WM_4mm_sm0'], ['falff_z_GM_WM_8mm_sm0'],
#     ['falff_z_GM_WM_3mm_sm8'], ['falff_z_GM_WM_4mm_sm8'], ['falff_z_GM_WM_8mm_sm8'],
#     ['falff_z_GM_3mm_sm0'], ['falff_z_GM_4mm_sm0'], ['falff_z_GM_8mm_sm0'],
#     ['falff_z_GM_3mm_sm8'], ['falff_z_GM_4mm_sm8'], ['falff_z_GM_8mm_sm8'],
#     #
#     ['reho_GM_WM_3mm_sm0'], ['reho_GM_WM_4mm_sm0'], ['reho_GM_WM_8mm_sm0'],
#     ['reho_GM_WM_3mm_sm8'], ['reho_GM_WM_4mm_sm8'], ['reho_GM_WM_8mm_sm8'],
#     ['reho_GM_3mm_sm0'], ['reho_GM_4mm_sm0'], ['reho_GM_8mm_sm0'],
#     ['reho_GM_3mm_sm8'], ['reho_GM_4mm_sm8'], ['reho_GM_8mm_sm8'],
#     #
#     ['variability_std_z_GM_WM_3mm_sm0'], ['variability_std_z_GM_WM_4mm_sm0'], ['variability_std_z_GM_WM_8mm_sm0'],
#     ['variability_std_z_GM_WM_3mm_sm8'], ['variability_std_z_GM_WM_4mm_sm8'], ['variability_std_z_GM_WM_8mm_sm8'],
#     ['variability_std_z_GM_3mm_sm0'], ['variability_std_z_GM_4mm_sm0'], ['variability_std_z_GM_8mm_sm0'],
#     ['variability_std_z_GM_3mm_sm8'], ['variability_std_z_GM_4mm_sm8'], ['variability_std_z_GM_8mm_sm8'],
#     #
#     ['craddock_205_BP'], ['craddock_788_BP'],
#     ['gordon_BP'], ['gordon_BP_ds'],
#     ['craddock_205_BP_scr05'], ['craddock_788_BP_scr05'], ['gordon_BP_scr05'],
#     ['msdl_abide_BP'],
#     #
#     ['behav_wml_wm_total'], ['behav_wml_wmh_norm'], ['behav_wml_wmh_norm_ln'], ['behav_wml_wmh_ln'],
#     ['behav_wml_fazekas']
# ]
#
# in_data_name_list = [
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg'],
#
#     ['alff_z_GM_WM_3mm_sm0'],
#     ['falff_z_GM_WM_3mm_sm0'],
#     ['reho_GM_WM_3mm_sm0'],
#     ['variability_std_z_GM_WM_3mm_sm0'],
#
#     ['alff_z_GM_3mm_sm0'],
#     ['falff_z_GM_3mm_sm0'],
#     ['reho_GM_3mm_sm0'],
#     ['variability_std_z_GM_3mm_sm0'],
#     ['craddock_205_BP'],
#     ['craddock_788_BP'],
#     #
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'alff_z_GM_3mm_sm0'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'falff_z_GM_3mm_sm0'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'reho_GM_3mm_sm0'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg',
#      'variability_std_z_GM_3mm_sm0'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'craddock_205_BP'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'craddock_788_BP'],
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg', 'behav_wml_wmh_norm_ln'],
#     #
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg',
#      'alff_z_GM_3mm_sm0',
#      'falff_z_GM_3mm_sm0',
#      'reho_GM_3mm_sm0',
#      'variability_std_z_GM_3mm_sm0',
#      'craddock_788_BP'],
#     #
#     ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg',
#      'alff_z_GM_3mm_sm0',
#      'craddock_788_BP'],
#     # # ['craddock_205_BP_scr05'], ['craddock_788_BP_scr05'],
#     # #
#     ['behav_wml_wm_total'], ['behav_wml_wmh_norm'], ['behav_wml_wmh_norm_ln'], ['behav_wml_wmh_ln'],
#     ['behav_wml_fazekas'],
#     ['behav_wml_wmh_norm_ln', 'behav_wml_fazekas'],
# ]
#
# in_data_name_list = [['gordon_BP'], ['gordon_BP_ds'],
#                      ['craddock_205_BP_scr05'], ['craddock_788_BP_scr05'], ['gordon_BP_scr05'],
#                      ['msdl_abide_BP']]

in_data_name_list = [
    ['dosenbach'],
    ['basc_197'], ['basc_444'],
    ['craddock_205_BP'],
    ['craddock_788_BP'],
    ['craddock_205_BP_scr05'], ['craddock_788_BP_scr05'],
    ['msdl_abide_BP'],

    ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0'],
    ['lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0'],
    ['aseg'],
    ['lh_ct_fsav5_sm0', 'rh_ct_fsav5_sm0', 'lh_csa_fsav5_sm0', 'rh_csa_fsav5_sm0', 'aseg'],

]

subjects_selection_crit_names_list = ['bothSexes_neuH']

target_list = ['age']


# check for duplicates in in_data_file_list
seen = []
for i in in_data_name_list:
    if i in seen:
        raise Exception('duplicate in in_data_name_list: %s' % i)
    else:
        seen.append(i)
