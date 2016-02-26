import pandas as pd
import numpy as np
import pylab as plt
import glob, os
import seaborn as sns
from collections import OrderedDict

df = pd.read_pickle('/home/raid2/liem/data/LIFE/behavioral/LIFE_subjects_behav_n2636.pkl')
qc = pd.read_pickle('/home/raid2/liem/data/LIFE/behavioral/LIFE_subjects_QC_n2557.pkl')

df = df.join(qc, how='right')

df_missing = df[df['WML_lesionload'].isnull()]
df_missing['t1_path'] = ''
df_missing['flair_path'] = ''

for s in df_missing.index.values:
    txt_file = '/data/liem-1/LIFE/raw_data/{s}/check/images_used.txt'.format(s=s)

    with open(txt_file) as fi:
        txt_data = fi.readlines()[0].strip()

    data_root_path = txt_data.strip()
    i = glob.glob(os.path.join(data_root_path, 'NIFTI', '*MPRAGE_ADNI_32Ch_*'))
    if i:
        t1_path = i[0]
    else:
        t1_path = np.nan

    i = glob.glob(os.path.join(data_root_path, 'NIFTI', '*t2_spc_da-fl_irprep_sag_p2_iso_395.nii.gz'))
    if i:
        flair_path = i[0]
    else:
        flair_path = np.nan
    print t1_path
    print flair_path
    df_missing.loc[s, ['t1_path', 'flair_path']] = [t1_path, flair_path]  #

fold_size = 100
n_folds = len(df_missing) / 100 + 1
for f in range(n_folds):
    ind_start = f * fold_size
    ind_end = ind_start + fold_size
    if ind_end > len(df_missing):
        ind_end = len(df_missing)

    print ind_start, ind_end
    t1 = df_missing.iloc[ind_start:ind_end]['t1_path'].values
    np.savetxt('/home/raid2/liem/data/LIFE//wml_redo/list_t1_%s_%s.txt' % (ind_start, ind_end), t1, fmt="%s")
    flair = df_missing.iloc[ind_start:ind_end]['flair_path'].values
    np.savetxt('/home/raid2/liem/data/LIFE//wml_redo/list_flair_%s_%s.txt' % (ind_start, ind_end), flair, fmt="%s")


# df.to_pickle(os.path.join(behav_out_folder, 'LIFE_subjects_behav_n%s.pkl' % str(len(df))))
df_missing.to_excel(os.path.join('/home/raid2/liem/data/LIFE/behavioral/', 'wml_missing.xlsx'))
# df.to_pickle('../LIFE_subjects_behav_n%s.pkl' % str(len(df)))
# df.to_excel('../LIFE_subjects_behav_n%s.xlsx' % str(len(df)))
#
