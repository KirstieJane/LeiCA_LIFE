import os
from learning.learning_stacking_utils import stacking


target = 'age'
selection_crit_list = ['bothSexes_neuH_FD06'] #'bothSexes_neuH_FD06_norm',


# root_path_template = '/scr/adenauer2/Franz/LeiCA_LIFE/learning_out_20160425/pdfs/single_source_model_reg_{reg}_predicted'
# out_path_template =  '/scr/adenauer2/Franz/LeiCA_LIFE/learning_out_20160425/stacking/stacking_out_reg_{reg}/'


root_path_template = '/Users/franzliem/Dropbox/LeiCA/MS/in_data/learning_out_20160427/pdfs/single_source_model_reg_{reg}_predicted'
out_path_template =  '/Users/franzliem/Dropbox/LeiCA/MS/in_data/learning_out_20160427/stacking/stacking_out_reg_{reg}/'




for reg in [True, False]:
    for selection_crit in selection_crit_list:
        root_path = root_path_template.format(reg=reg)
        out_path = out_path_template.format(reg=reg)

        file_pref = target + '__' + selection_crit + '__'
        source_dict = {
            'aseg': os.path.join(root_path, file_pref + 'aseg_df_predicted.pkl'),
            'ct': os.path.join(root_path, file_pref + 'lh_ct_fsav4_sm0__rh_ct_fsav4_sm0_df_predicted.pkl'),
            'csa': os.path.join(root_path, file_pref + 'lh_csa_fsav4_sm0__rh_csa_fsav4_sm0_df_predicted.pkl'),
            'basc197': os.path.join(root_path, file_pref + 'basc_197_df_predicted.pkl'),
            'basc444': os.path.join(root_path, file_pref + 'basc_444_df_predicted.pkl'),
        }

        source_selection_dict = {'all': ['basc197', 'basc444', 'aseg', 'csa', 'ct'],
                                 'rs': ['basc197', 'basc444'],
                                 'fs': ['aseg', 'csa', 'ct'],
                                 }



        stacking(out_path, target, selection_crit, source_dict, source_selection_dict, rf=None)



