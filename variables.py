import os
from utils import load_subjects_list

script_dir = os.path.dirname(os.path.realpath(__file__))
template_dir = os.path.join(script_dir, 'templates')

wd_root_path = '/data/liem-3/LIFE/'
metrics_root_path = '/scr/adenauer2/Franz/LeiCA_LIFE/'

ds_root_path = '/scr/adenauer2/Franz/LeiCA_LIFE'

subjects_list_folder = '/home/raid2/liem/data/LIFE/behavioral'

subject_file = '/home/raid2/liem/data/LIFE/LIFE16_preprocessed_subjects_list_n2557.txt'


behav_file = '/home/raid2/liem/data/LIFE/behavioral/LIFE_subjects_behav_n2648.pkl'
qc_file = '/home/raid2/liem/data/LIFE/behavioral/LIFE_subjects_QC_n2557.pkl'

subjects_list = load_subjects_list(subject_file)



in_data_root_path = '/data/liem-1/LIFE/preprocessed'

selectfiles_templates = {
    'epi_MNI_bp': '{subject_id}/resting_state/ants/rest_mni_unsmoothed.nii.gz',
    'epi_MNI_fullspectrum': '{subject_id}/resting_state/ants/rest_mni_unsmoothed_fullspectrum.nii.gz',
    'moco_parms_file': '{subject_id}/resting_state/realign/rest_realigned.par',
    'jenkinson_file': '{subject_id}/resting_state/realign/rest_realigned_rel.rms',
    'rest2anat_cost_file': '{subject_id}/resting_state/coregister/rest2anat.dat.mincost',
}





# # # # # # # # # #
# LEARNING data_lookup_dict
# # # # # # # # # #
gordon_path = os.path.join(template_dir, 'parcellations/Gordon_2014_Parcels/Parcels_MNI_111_sorted.nii.gz')
craddock_788_path = os.path.join(template_dir,
                                 'parcellations/craddock_2012/scorr_mean_single_resolution/scorr_mean_parc_n_43_k_788_rois.nii.gz')

data_lookup_dict = {}
# fixme add mask name

metrics = {'alff': 'alff/alff.nii.gz',
           'falff': 'alff/falff.nii.gz',
           'alff_z': 'alff_z/alff_zstd.nii.gz',
           'falff_z': 'alff_z/falff_zstd.nii.gz',
           'reho': 'reho/ReHo.nii.gz'}
masks = ['GM', 'WM', 'GM_WM', 'brain_mask']
resolutions = [3, 4, 8]

for m in metrics.keys():
    for ma in masks:
        for r in resolutions:
            m_str = '%s_%s_%smm' % (m, ma, r)
            ma_str = '%s_MNI_%smm' % (ma, r)
            data_lookup_dict[m_str] = {'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', metrics[m]),
                                       'mask_name': ma_str}

resolutions = [5, 4, 3]
hemis = ['lh', 'rh']
metrics = {'ct': 'thickness', 'csa': 'area'}
smoothing = [0, 5, 10, 20]
for h in hemis:
    for m in metrics.keys():
        for r in resolutions:
            for s in smoothing:
                m_str = '%s_%s_fsav%s_%smm' % (h, m, r, s)
                surf_str = 'surfs/%s.%s.fsaverage%s.%smm.mgz' % (h, metrics[m], r, s)
                data_lookup_dict[m_str] = {'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', surf_str)}

data_lookup_dict['aseg'] = {'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', 'parcstats/aseg')}

data_lookup_dict['variability_std'] = {
    'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}', 'variability/ts_std.nii.gz')}

data_lookup_dict['craddock_205_noBP'] = {'matrix_name': 'correlation',
                                         'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                  'con_mat/matrix/bp_None.None/craddock_205/matrix.pkl')}
data_lookup_dict['craddock_205_BP'] = {'matrix_name': 'correlation',
                                       'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                'con_mat/matrix/bp_0.01.0.1/craddock_205/matrix.pkl')}

data_lookup_dict['craddock_788_noBP'] = {'matrix_name': 'correlation',
                                         'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                  'con_mat/matrix/bp_None.None/craddock_788/matrix.pkl')}
data_lookup_dict['craddock_788_BP'] = {'matrix_name': 'correlation',
                                       'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                                'con_mat/matrix/bp_0.01.0.1/craddock_788/matrix.pkl')}
data_lookup_dict['gordon_noBP'] = {'matrix_name': 'correlation',
                                   'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                            'con_mat/matrix/bp_None.None/gordon/matrix.pkl')}
data_lookup_dict['gordon_BP'] = {'matrix_name': 'correlation',
                                     'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                              'con_mat/matrix/bp_0.01.0.1/gordon/matrix.pkl')}

data_lookup_dict['gordon_BP_ds'] = {'matrix_name': 'correlation',
                                     'path_str': os.path.join(metrics_root_path, 'metrics/{subject_id}',
                                                              'con_mat/matrix/bp_0.01.0.1/gordon/matrix_downsampled.pkl'),
                                    'use_diagonal': True}



# # # # # # # # # #
# template_lookup_dict
# # # # # # # # # #
# get atlas data
template_lookup_dict = {
    'brain_mask_MNI_3mm_frauke': 'Templates/Frauke_Templates/MNI_resampled_brain_mask.nii'
}

for res in [3, 4, 8]:
    for img in ['GM', 'WM', 'GM_WM', 'brain_mask']:
        k = img + '_MNI_' + str(res) + 'mm'
        v = 'MNI152_T1_' + str(res) + 'mm_' + img + '.nii.gz'
        template_lookup_dict[k] = 'Templates/' + v

template_lookup_dict = {k: os.path.join(template_dir, v) for k, v in template_lookup_dict.items()}