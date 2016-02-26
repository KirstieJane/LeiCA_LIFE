'''
wd ca 170MB/subj
'''
import os
from metrics.calc_metrics_redo import calc_local_metrics
from variables import template_dir, subjects_list, metrics_root_path, wd_root_path, selectfiles_templates
from utils import load_subjects_list


subject_file = '/scr/adenauer2/Franz/LIFE16/LIFE16_preprocessed_subjects_list_n2557.txt'
subjects_list = load_subjects_list(subject_file)

#
# # CHANCE FOLD HERE
# fold = 0
#
# # DONT CHANGE BELOW
# ######
# fold_size = 700
# offset = 1979
# i1 = offset + fold*fold_size
# i2 = i1+fold_size
# if not i2 < len(subjects_list):
#     i2 = len(subjects_list)-1
#
# print (i1,i2)
# subjects_list = subjects_list[i1:i2]


working_dir_base = os.path.join(wd_root_path, 'wd_metrics')
ds_dir_base = os.path.join(metrics_root_path, 'metrics')

brain_mask = '/scr/adenauer2/Franz/LIFE16/Templates/MNI_resampled_brain_mask.nii'
gm_wm_mask = '/home/raid2/liem/PowerFolders/Workspace/LeiCA_LIFE/templates/Templates/MNI152_T1_3mm_GM_WM.nii.gz'

template_dir = os.path.join(template_dir, 'parcellations')

selectfiles_templates = {
    'alff': '{subject_id}/alff/alff.nii.gz',
    'falff': '{subject_id}/alff/falff.nii.gz',
    'variability': '{subject_id}/variability/ts_std.nii.gz',
}
use_n_procs = 5
#plugin_name = 'MultiProc'
plugin_name = 'CondorDAGMan'


for subject_id in subjects_list:
    working_dir = os.path.join(working_dir_base, subject_id)
    ds_dir = os.path.join(ds_dir_base, subject_id)

    print('\n\nsubmitting %s'%subject_id)
    calc_local_metrics(gm_wm_mask=gm_wm_mask,
                       preprocessed_data_dir=ds_dir_base,
                       subject_id=subject_id,
                       selectfiles_templates=selectfiles_templates,
                       working_dir=working_dir,
                       ds_dir=ds_dir,
                       use_n_procs=use_n_procs,
                       plugin_name=plugin_name)
