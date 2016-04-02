import os, glob
import seaborn as sns
import pylab as plt
import numpy as np
import pandas as pd

ana_stream_root = '*_scaler_standard_rfe_False_strat_False_reg_*_results_error'
metrics_sel_list = ['lh_ct_fsav5_sm0__rh_ct_fsav5_sm0__lh_csa_fsav5_sm0__rh_csa_fsav5_sm0__aseg',
                    'GM_3mm_sm0',
                    'craddock_788_BP']
metrics_sel_list = ['lh_ct_fsav5_sm0__rh_ct_fsav5_sm0__lh_csa_fsav5_sm0__rh_csa_fsav5_sm0__aseg']

# metrics_sel_list = ['*']
ds_root_path = '/Users/franzliem/PowerFolders/00_Print/tmp/learning_out_partial_20160401'
fig_path = '/Users/franzliem/Dropbox/LeiCA/PPT/20160321_results/figs'
if not os.path.exists(fig_path):
    os.mkdir(fig_path)

ds_dir = os.path.join(ds_root_path, 'pdfs')

os.chdir(ds_dir)
df = None
dd = []
for ana_stream in glob.glob(ana_stream_root):
    for metrics_sel in metrics_sel_list:
        df_list = glob.glob(os.path.join(ana_stream, '*' + metrics_sel + '*' + '*.pkl'))
        for df_file in df_list:
            dd.append(df_file)
            df_in = pd.read_pickle(df_file)
            df_in['ana_stream'] = ana_stream
            s = df_in.index.values[0].split('__')
            target, selection = s[0:2]
            metrics = '__'.join(s[2:])
            _, _, scaler, _, rfe, _, strat, _, reg, _, _ = ana_stream.split('_')
            #
            # try:
            #     fsav = metrics.split('_')[2]
            #     if not fsav.startswith('fsav'):
            #         fsav = 'na'
            # except:
            #     fsav = 'na'
            #
            # try:
            #     sm = metrics.split('_')[3]
            #     if not sm.startswith('sm'):
            #         sm = 'na'
            # except:
            #     sm = 'na'
            # m = metrics.split('_')[0]
            # res = metrics.split('_')[-2]
            # sm = metrics.split('_')[-1]
            # if metrics.split('_')[-3] == 'WM':
            #     mask = 'GM_WM'
            # else:
            #     mask = 'GM'

            df_in['scaler'] = scaler
            df_in['rfe'] = rfe
            df_in['strat'] = strat
            df_in['reg'] = reg
            df_in['target'] = target
            df_in['selection'] = selection
            df_in['metrics'] = metrics
            # df_in['fsav'] = fsav
            # df_in['m'] = m
            # df_in['res'] = res
            # df_in['sm'] = sm
            # df_in['mask'] = mask
            df_in['parc'] = metrics.split('_')[0]

            df_in = df_in.replace({'False': False, 'True': True})
            if df is None:
                df = df_in
            else:
                df = df.append(df_in)
df = df.drop_duplicates()
df['index_name'] = df.index
multi_index = pd.MultiIndex.from_tuples(zip(df.index, df['reg']), names=['index_name', 'reg'])
df.set_index(multi_index, inplace=True)

##### mean of sep. pred
scatter_df = {}
for i in range(len(df)):
    metric = df.iloc[i].name[0]
    ana_stream = df.iloc[i]['ana_stream']
    scatter_path = '_'.join(ana_stream.split('_')[:-2] + ['predicted'])
    scatter_filename = metric + '_df_predicted.pkl'
    scatter_file = os.path.join(scatter_path, scatter_filename)

    if ana_stream in scatter_df:
        d = scatter_df[ana_stream]
    else:
        d = dict()
    d.update({metric: pd.read_pickle(scatter_file)})
    scatter_df.update({ana_stream: d})
#####
reg = {False: '00_scaler_standard_rfe_False_strat_False_reg_False_results_error',
      True: '01_scaler_standard_rfe_False_strat_False_reg_True_results_error'}
df_single_pred_age = {}
r2_mean = {}
for reg_k, reg_v in reg.items():
    sc = scatter_df[reg_v]

    df_single_pred_age[reg_k] = pd.DataFrame([], columns=['age'])
    df_single_pred_age[reg_k].age = sc[sc.keys()[0]].age
    try:
        df_single_pred_age[reg_k]['split_group'] = sc[sc.keys()[0]].split_group
    except:
        df_single_pred_age[reg_k]['split_group'] = 'test'

    for m in sc.keys():
        df_single_pred_age[reg_k][m] = sc[m].pred_age_test
        tmp = df_single_pred_age[reg_k][df_single_pred_age[reg_k].split_group == 'test']
        r_pears_2 = np.corrcoef(tmp[m].values,tmp['age'].values)[0,1] ** 2
        df.ix[(m, reg_k), 'r_pears_2'] = r_pears_2
    df_single_pred_age[reg_k]['mean_pred_age'] = df_single_pred_age[reg_k].ix[:, 2:].mean(1)

    from sklearn.metrics import r2_score

    r2_mean[reg_k] = r2_score(df_single_pred_age[reg_k][df_single_pred_age[reg_k].split_group == 'test'].age,
                       df_single_pred_age[reg_k][df_single_pred_age[reg_k].split_group == 'test'].mean_pred_age)



#####
# FILTER
# df = df[df.m=='falff']

# df = df[df.r2_test > 0]
df.r2_test[df.r2_test < 0] = -.1
df['n_metrics'] = df['index_name'].str.count('__') - 1
df.sort_values(['n_metrics', 'index_name'], inplace=True)


def break_text(s, n):
    o = []
    while s:
        o.append(s[:n])
        s = s[n:]
    o = '\n'.join(o)
    return o


plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="r2_test", y="metrics", hue="reg", data=df[~df.reg])
ax = plt.axes()
li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
ax.set_yticklabels(li)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '06_combined_1_r2.pdf'))
# plt.show()
plt.close()

plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="MAE_test", y="metrics", hue="reg", data=df[~df.reg])
ax = plt.axes()
li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
ax.set_yticklabels(li)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '06_combined_2_mae.pdf'))
# plt.show()
plt.close()
#
# df_plt = df[~df.reg]
# plt.figure(figsize=(11.69, 8.27))
# sns.regplot(x="MAE_test", y="r2_test", data=df_plt, ci=0)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_path, 'x.pdf'))
# # plt.show()
# plt.close()
#
# plt.figure(figsize=(11.69, 8.27))
# sns.regplot(x="MAE_test", y="r2_test", data=df_plt, ci=0)
# ax = plt.axes()
# for i in range(len(df_plt)):
#     ax.annotate(df_plt.iloc[i].metrics, (df_plt.iloc[i].MAE_test, df_plt.iloc[i].r2_test))
# plt.tight_layout()
# plt.savefig(os.path.join(fig_path, 'x.pdf'))
# # plt.show()
# plt.close()
#
# plt.figure(figsize=(11.69, 8.27))
# sns.barplot(x="r2_test", y="metrics", hue="reg", data=df)
# ax = plt.axes()
# li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
# ax.set_yticklabels(li)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_path, 'x.pdf'))
# # plt.show()
# plt.close()
#
# plt.figure(figsize=(11.69, 8.27))
# sns.barplot(x="MAE_test", y="metrics", hue="reg", data=df)
# ax = plt.axes()
# li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
# ax.set_yticklabels(li)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_path, 'x.pdf'))
# # plt.show()
# plt.close()
