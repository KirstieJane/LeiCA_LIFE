import os, glob
import seaborn as sns
import pylab as plt
import numpy as np
import pandas as pd

ana_stream_root = '*_scaler_standard_rfe_False_strat_False_reg_*_results_error'
metrics_sel_list = ['lh_ct_fsav5_sm0__rh_ct_fsav5_sm0__lh_csa_fsav5_sm0__rh_csa_fsav5_sm0__aseg',
                    'GM_3mm_sm0',
                    'craddock_788_BP']
metrics_sel_list = ['*']
ds_root_path = '/Users/franzliem/PowerFolders/00_Print/tmp/learning_out_20160322'
fig_path = '/Users/franzliem/Dropbox/LeiCA/PPT/20160321_results/figs'
if not os.path.exists(fig_path):
    os.mkdir(fig_path)

ds_dir = os.path.join(ds_root_path, 'pdfs')

os.chdir(ds_dir)
df = None
dd=[]
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
            #m = metrics.split('_')[0]
            #res = metrics.split('_')[-2]
            #sm = metrics.split('_')[-1]
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
            #df_in['fsav'] = fsav
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

# FILTER
#df = df[df.m=='falff']

#df = df[df.r2_test > 0]
df.r2_test[df.r2_test < 0] = -.1
df['n_metrics'] = df.index.str.count('__') - 1
df['index'] = df.index
df.sort_values(['n_metrics', 'index'], inplace=True)



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
plt.savefig(os.path.join(fig_path, '04_selected_1_r2.pdf'))
#plt.show()
plt.close()

plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="MAE_test", y="metrics", hue="reg", data=df[~df.reg])
ax = plt.axes()
li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
ax.set_yticklabels(li)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '04_selected_2_mae.pdf'))
#plt.show()
plt.close()

df_plt=df[~df.reg]
plt.figure(figsize=(11.69, 8.27))
sns.regplot(x="MAE_test", y="r2_test", data=df_plt, ci=0)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '04_selected_3_r2_mae.pdf'))
#plt.show()
plt.close()

plt.figure(figsize=(11.69, 8.27))
sns.regplot(x="MAE_test", y="r2_test", data=df_plt, ci=0)
ax = plt.axes()
for i in range(len(df_plt)):
    ax.annotate(df_plt.iloc[i].metrics, (df_plt.iloc[i].MAE_test, df_plt.iloc[i].r2_test))
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '04_selected_4_r2_mae_annot.pdf'))
#plt.show()
plt.close()


plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="r2_test", y="metrics", hue="reg", data=df)
ax = plt.axes()
li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
ax.set_yticklabels(li)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '04_selected_5_reg_r2.pdf'))
#plt.show()
plt.close()

plt.figure(figsize=(11.69, 8.27))
sns.barplot(x="MAE_test", y="metrics", hue="reg", data=df)
ax = plt.axes()
li = [break_text(l.get_text(), 60) for l in ax.get_yticklabels()]
ax.set_yticklabels(li)
plt.tight_layout()
plt.savefig(os.path.join(fig_path, '04_selected_6_reg_mae.pdf'))
#plt.show()
plt.close()

#
#
# from bokeh.plotting import figure, output_file, show, ColumnDataSource
# from bokeh.models import HoverTool, BoxZoomTool
# from bokeh.models import BoxSelectTool, LassoSelectTool, Paragraph
# from bokeh.plotting import figure, hplot, vplot
# source = ColumnDataSource(df)
#
# scatter_df = {}
# for i in range(len(df)):
#     metric = df.iloc[i].name
#     ana_stream = df.iloc[i]['ana_stream']
#     scatter_path = '_'.join(ana_stream.split('_')[:-2] + ['predicted'])
#     scatter_filename = metric + '_df_predicted.pkl'
#     scatter_file = os.path.join(scatter_path, scatter_filename)
#
#     if ana_stream in scatter_df:
#         d = scatter_df[ana_stream]
#     else:
#         d = dict()
#     d.update({metric: pd.read_pickle(scatter_file)})
#     scatter_df.update({ana_stream: d})
#
# hover = HoverTool(tooltips=[("ana_stream", "@ana_stream"),
#                             ("index", "@index"),
#                             ("r2", "@r2_test"),
#                             ("mae", "@MAE_test"),
#                             ("n_metrics", "@n_metrics"),
#                             ])
# box_zoom = BoxZoomTool()
# output_file("scatter_test.html")
#
# if simple:
#     p = figure(tools=[hover, 'wheel_zoom,box_zoom,reset'], plot_width=1100, plot_height=600, title=None, min_border=10,
#                min_border_left=500, )
#     r = p.scatter('r2_test', 'MAE_test', source=source, alpha=0.6)
#     show(p)
#
# else:
#     p = figure(tools=[hover, 'wheel_zoom,box_zoom,reset'], plot_width=1100, plot_height=600, title=None, min_border=10,
#                min_border_left=500)
#     r = p.scatter('r2_test', 'MAE_test', source=source, alpha=0.6)
#     ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, title=None, min_border=10,
#                 min_border_left=50)
#     zoom_df = scatter_df['04_scaler_standard_rfe_True_strat_False_reg_False_results_error'][
#         'age__bothSexes_neuH__lh_ct_fsav4_sm0__rh_ct_fsav4_sm0']
#     zoom_source = ColumnDataSource(zoom_df)
#
#     ph.circle('age', 'pred_age_test', source=zoom_source)
#     layout = vplot(p, ph)
#
# show(p)
#


# all_ana = set(df.query('MAE_test < 10').index)
# df_all = df.loc[all_ana]
# df_all.ix[:, 'ana'] = df_all.index.values
# multi_index = pd.MultiIndex.from_tuples(zip(df_all['ana'], df_all['FD_res']), names=['ana_id', 'FD_red'])
# df_all.set_index(multi_index, inplace=True)
# df_all.sort_index(inplace=True)
# df_all.to_csv('00_df_all.csv')
#
# if len(all_ana) < 20:
#     plt.figure()
#     ax = sns.barplot(x="MAE_test", y="ana", data=df_all, hue='FD_res')
#     plt.tight_layout()
#     plt.savefig('00_all_bars_MAE.pdf')
#
#     plt.figure()
#     ax = sns.barplot(x="r2_test", y="ana", data=df_all, hue='FD_res')
#     plt.tight_layout()
#     plt.savefig('00_all_bars_r2.pdf')
#
# good_ana = set(df.query('r2_test > .7').index)
#
# df_good = df.loc[good_ana]
# df_good.ix[:, 'ana'] = df_good.index.values
# multi_index = pd.MultiIndex.from_tuples(zip(df_good['ana'], df_good['FD_res']), names=['ana_id', 'FD_red'])
# df_good.set_index(multi_index, inplace=True)
# df_good.sort_index(inplace=True)
#
# def plot_bars(**kwargs):
#
#     def format_ticks(ax):
#         def break_text(s, n):
#             o = []
#             while s:
#                 o.append(s[:n])
#                 s = s[n:]
#             o = '\n'.join(o)
#             return o
#
#         li = [break_text(l.get_text(), 50) for l in ax.get_yticklabels()]
#         ax.set_yticklabels(li)
#         return ax
#
#
#     plt.figure()
#     ax = sns.barplot(**kwargs)
#     ax = format_ticks(ax)
#     try:
#         plt.tight_layout()
#     except:
#         pass
#
#
# c = sns.color_palette()
# plot_bars(x="MAE_test", y="ana", data=df_good, hue='FD_res')
# plt.savefig('00_best_bars_MAE.pdf')
#
#
# plt.figure()
# plot_bars(x="r2_test", y="ana", data=df_good, hue='FD_res')
# plt.savefig('00_best_bars_r2.pdf')
#
# f, axarr = plt.subplots(len(good_ana), 2, sharex=True, sharey=True, figsize=(15, 35))
# for row, a in enumerate(good_ana):
#     if len(a) > 50:
#         a_str = a[:50] + '\n' + a[50:]
#     else:
#         a_str = a
#
#     ax = axarr[row, 0]
#     sns.regplot(x='age', y='pred_age_test', data=scatter_df[False][a], ax=ax, color=c[0])
#     ax.set_aspect('equal')
#     ax.plot([10, 80], [10, 80], 'k')
#     ax.set_title('%s\nr2 = %3.2f ' % (a_str, df[df.FD_res == False].loc[a]['r2_test']))
#
#     ax = axarr[row, 1]
#     sns.regplot(x='age', y='pred_age_test', data=scatter_df[True][a], ax=ax, color=c[1])
#     ax.set_aspect('equal')
#     ax.plot([10, 80], [10, 80], 'k')
#     ax.set_title('%s\nr2 = %3.2f ' % (a_str, df[df.FD_res == True].loc[a]['r2_test']))
#
# plt.tight_layout()
# plt.savefig('00_best_scatters.pdf')
