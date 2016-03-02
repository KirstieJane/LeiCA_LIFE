'''
Imports and concatenates behav files
1. Remove duplicate subjects (if 2 MR tps, take second)
2. Add Cerad (no exclusion)
3. Add MR diagnostics
4. Exclude subjects without diagnostics (mri_lesion_num=999999)
5. Create 'neurol_healthy': subjects without ischemic, hemorraghic, both, traumatic diagnostic (mri_lesion_num 2-5)
(subjects with diagnostics but without Fazekas are not excluded)

'''

import pandas as pd
import numpy as np
import pylab as plt
import glob, os
import seaborn as sns
from collections import OrderedDict

behav_out_folder = '/Users/franzliem/PowerFolders/LIFE/behavioral'
os.chdir('/Users/franzliem/Dropbox/LeiCA/LIFE/behavioral/data')
n = OrderedDict()

prov = pd.DataFrame([], columns=['from_file'])
def add_prov(df, f_name):
    for c in df.columns.values:
        prov.loc[c] = f_name

def read_life_excel(xls):
    df = pd.read_excel(xls)
    df.set_index(df.ix[:, 0], inplace=True)
    return df


def create_multi_index(df):
    # creates unique index from sic and study group (pilot, haupt)
    def get_grp_col(df):
        import fnmatch
        for i in df.columns.values:
            if fnmatch.fnmatch(i.lower(), '*_grp'):
                return i
            if fnmatch.fnmatch(i.lower(), '*_gruppe'):
                return i

    grp_col = get_grp_col(df)
    assert (grp_col is not None), 'Could not find group column'
    multi_index = pd.MultiIndex.from_tuples(zip(df.index, df[grp_col]), names=['SIC', 'study'])
    df.set_index(multi_index, inplace=True)
    return df


# read basic df
df = read_life_excel('PV0250_R00001.xlsx')[['R00001_PBD_GESCHLECHT', 'R00001_PBD_GEBJAHRMON']]
df['dob'] = pd.to_datetime(df.R00001_PBD_GEBJAHRMON, format='%Y%m')
# read MR df
df_mr_orig = read_life_excel('PV0250_T00197.xlsx')[['MRT_SIC', 'MRT_DATUM', 'MRT_GRUPPE']]



# for duplicates, take more recent MR date
df_mr_orig.sort_values(by='MRT_DATUM', inplace=True)
df_mr_dup = df_mr_orig[df_mr_orig.index.duplicated(keep=False)]
df_mr = df_mr_orig.drop_duplicates(subset='MRT_SIC', keep='last')

# Deal with special duplicate case: data missing on second time point -> use first
df_mr.drop('LI00002611', axis=0, inplace=True)
df_mr.loc['LI00002611'] = df_mr_dup.loc['LI00002611'].ix[0, :]

df = df.join(df_mr, how='outer')
df['t_days'] = (df.MRT_DATUM - df.MRT_DATUM.min()).dt.days

# calc age and sex
df['age'] = (df_mr.MRT_DATUM - df.dob).dt.days / 365.25
df['sex'] = df['R00001_PBD_GESCHLECHT'].replace({1: 'M', 2: 'F'})
df_all = df.copy()
df = create_multi_index(df)
df_big = df.copy()

# add CERAD
cerad_list = glob.glob('*_D000[456]*.xlsx')
df_cerad_big = None
for c in cerad_list:
    df_in = read_life_excel(c)
    df_in = create_multi_index(df_in)
    if df_cerad_big is None:
        df_cerad_big = df_in
    else:
        df_cerad_big = df_cerad_big.join(df_in, how='outer')
    add_prov(df_in, c)

ren = {'CERAD_TOTAL_CERAD_WL_TOTAL': 'CERAD_LEA_O',
       'CERAD_WL_LERN_WORTLISTE_TOTAL': 'CERAD_LEA_Y',
       'CERAD_WL_ABRUF_CERAD_WL_ABRUF': 'CERAD_RECALL_O',
       'CERAD_WL_AB_WORTLISTE_WL_ABR': 'CERAD_RECALL_Y',
       'CERAD_WL_ERKENN_CERAD_WL_ERK': 'CERAD_RECOG_O',
       'CERAD_WL_ERKENN_J_WORTL_WL_ERK': 'CERAD_RECOG_Y'}
df_cerad_big.rename(columns=ren, inplace=True)
# nanmean for _Y and _O-> if both are nan: nan; if 1 is nan: pick other one; if none is nan: mean
df_cerad_big['CERAD_LEA'] = df_cerad_big[['CERAD_LEA_O', 'CERAD_LEA_Y']].apply(np.nanmean, axis=1)
df_cerad_big['CERAD_RECALL'] = df_cerad_big[['CERAD_RECALL_O', 'CERAD_RECALL_Y']].apply(np.nanmean, axis=1)
df_cerad_big['CERAD_RECOG'] = df_cerad_big[['CERAD_RECOG_O', 'CERAD_RECOG_Y']].apply(np.nanmean, axis=1)
df_cerad = df_cerad_big[['CERAD_LEA', 'CERAD_RECALL', 'CERAD_RECOG']]

df = df.join(df_cerad, how='left')
df_big = df_big.join(df_cerad_big, how='left')

# add TMT
df_tmt = read_life_excel('PV0250_T00041.xlsx')
add_prov(df_tmt, 'PV0250_T00041.xlsx')
df_tmt = create_multi_index(df_tmt)
df_tmt.dropna(axis=0, subset=['TMT_TIMEA', 'TMT_ERRORSA', 'TMT_TIMEB', 'TMT_ERRORSB'], inplace=True)
df_big = df_big.join(df_tmt, how='left')
df_tmt['TMT_task_switching'] = (df_tmt['TMT_TIMEB'] - df_tmt['TMT_TIMEA']) / df_tmt['TMT_TIMEA']
df = df.join(df_tmt[['TMT_task_switching', 'TMT_TIMEA', 'TMT_TIMEB']], how='left')


# add VF
df_vf_1 = read_life_excel('PV0250_D00046.xlsx')
add_prov(df_vf_1, 'PV0250_D00046.xlsx')
df_vf_1 = create_multi_index(df_vf_1)
df_vf_2 = read_life_excel('PV0250_D00061.xlsx')
add_prov(df_vf_2, 'PV0250_D00061.xlsx')
df_vf_2 = create_multi_index(df_vf_2)
df_vf = df_vf_1.join(df_vf_2, how="outer")
df_vf.rename(columns={'CERAD_S_SUM_CERAD_S': 'VF_phon', 'SUM_CERADVF_SUM_CERADVF': 'VF_sem'}, inplace=True)
df_big = df_big.join(df_vf, how='left', lsuffix='1', rsuffix='2')
df = df.join(df_vf[['VF_phon', 'VF_sem']], how='left')



# remove duplicates and set index back to sic
df.set_index(df.MRT_SIC, inplace=True)
df.drop(labels=['R00001_PBD_GEBJAHRMON', 'dob', 'R00001_PBD_GESCHLECHT', 'MRT_SIC'], axis=1, inplace=True)
n['pre befund'] = len(df)

# add diagnostics
df_befund = pd.read_csv('MRT_befund_daten.csv', na_values=999999)
df_befund.dropna(axis=0, subset=['mri_lesion_num'], inplace=True)
df_befund.rename(columns={'\xef\xbb\xbfSIC': 'SIC'}, inplace=True)
df_befund.set_index('SIC', inplace=True)
# Frauke: mri_lesion_num. Du solltest 2-5 und 999999 ausschließen, 0 und 1 sind ok
#  ischemic, hemorraghic, both, traumatic (coded as 2-5), +missing (999999) lesions
df_befund['neurol_healthy'] = False
df_befund.loc[df_befund['mri_lesion_num'] <= 2, ['neurol_healthy']] = True
df_befund['MRT_BefundFazekas'] = pd.to_numeric(df_befund.MRT_BefundFazekas, 'coerce')

df = df.join(df_befund[['MRT_BefundFazekas', 'mri_lesion_num', 'mri_tumors_num', 'neurol_healthy']], how='inner')
# df.dropna(axis=0, subset=['MRT_BefundFazekas'], inplace=True)

# add lesion volume
# WML_lesionload (gen_lesionload) ist der ausgegebene Wert von LesionTOADS
# WML_lesionload_norm_tiv (gen_n) ist Läsionsvolumen normalisiert mit TIV
# WML_lesionload_norm_tiv_ln (gen_n_ln) ist das transformierte Läsionsvolumen, damit es normalverteilt ist (falls du mit parametrischen Tests arbeitest)

df_lesvol = pd.read_csv('lesions_franz.csv', )
df_lesvol.set_index('SIC', inplace=True)
df_lesvol = df_lesvol.ix[:, 1:]
ren = {'gen_lesionload': 'WML_lesionload', 'gen_n': 'WML_lesionload_norm_tiv', 'gen_n_ln': 'WML_lesionload_norm_tiv_ln'}
df_lesvol.rename(columns=ren, inplace=True)

df = df.join(df_lesvol, how='left')

n['post befund'] = len(df)
n['neurol healthy'] = len(df.query('neurol_healthy'))

print('N SUBJECTS')
print(n)


# add more to big
tests = ['PV0250_D00030.xlsx', 'PV0250_T00041.xlsx', 'PV0250_T00043.xlsx', 'PV0250_T00044.xlsx', 'PV0250_T00083.xlsx',
         'PV0250_T00084.xlsx']
for f in tests:
    df_in = read_life_excel(f)
    df_in = create_multi_index(df_in)
    add_prov(df_in, f)
    df_big = df_big.join(df_in, how='left', lsuffix='1', rsuffix='2')

df_big.set_index(df_big.MRT_SIC, inplace=True)
df_big.drop(labels=['R00001_PBD_GEBJAHRMON', 'dob', 'R00001_PBD_GESCHLECHT', 'MRT_SIC'], axis=1, inplace=True)


df.to_pickle(os.path.join(behav_out_folder, 'LIFE_subjects_behav_n%s.pkl' % str(len(df))))
df.to_excel(os.path.join(behav_out_folder, 'LIFE_subjects_behav_n%s.xlsx' % str(len(df))))

df_big.to_excel(os.path.join(behav_out_folder, 'LIFE_subjects_behav_n%s_big.xlsx' % str(len(df_big))))
prov.to_excel(os.path.join(behav_out_folder,'LIFE_subjects_behav_n%s_big_prov.xlsx' % str(len(df_big))))
# df.to_pickle('../LIFE_subjects_behav_n%s.pkl' % str(len(df)))
# df.to_excel('../LIFE_subjects_behav_n%s.xlsx' % str(len(df)))


df_excluded = df_all.drop(labels=df.index, axis=0)




# plt.figure()
# sns.regplot('t_days', 'age', data=df, fit_reg=False);
# plt.savefig('../LIFE_sampling.pdf')
#
# df_old = pd.read_pickle('../120_all_available_subjects_n2559.pkl')[[u'mean_FD_P', u'max_FD_P']]
#
# df = df.join(df_old, how='right')
#
# corcol=['age', 'm', 'mean_FD_P']
# df_ = df[corcol]
# from partial_corr import partial_corr
# df_.corr()
#
# df_.dropna(inplace=True)
# p = pd.DataFrame(partial_corr(df_), index=corcol, columns=corcol)
