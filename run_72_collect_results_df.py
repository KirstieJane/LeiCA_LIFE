__author__ = 'franzliem'

import os, glob
import pandas as pd
#from variables import ds_root_path
import seaborn as sns
import pylab as plt
import numpy as np

ds_root_path = '/SCR2/Franz/LeiCA_LIFE'


ds_dir = os.path.join(ds_root_path, 'learning_out/pdfs')

os.chdir(ds_dir)
df = None
for reg in ['01_split_results_error', '03_split_regFD_results_error']:
    df_list = glob.glob(reg + '/*.txt')
    for df_file in df_list:
        df_in = pd.read_csv(df_file, index_col=0)
        if df is None:
            df = df_in
        else:
            df = df.append(df_in)

scatter_df = {}
ana_list = np.unique(df.index.values)
for reg_path, reg in {'01_split_predicted': False, '03_split_regFD_predicted': True}.items():
    scatter_df[reg] = {}
    for a in ana_list:
        scatter_df[reg][a] = pd.read_pickle(os.path.join(ds_dir, reg_path, a + '_df_predicted.pkl'))

all_ana = set(df.query('MAE_test < 10').index)
df_all = df.loc[all_ana]
df_all.ix[:, 'ana'] = df_all.index.values
multi_index = pd.MultiIndex.from_tuples(zip(df_all['ana'], df_all['FD_res']), names=['ana_id', 'FD_red'])
df_all.set_index(multi_index, inplace=True)
df_all.sort_index(inplace=True)
df_all.to_csv('00_df_all.csv')

if len(all_ana) < 20:
    plt.figure()
    ax = sns.barplot(x="MAE_test", y="ana", data=df_all, hue='FD_res')
    plt.tight_layout()
    plt.savefig('00_all_bars_MAE.pdf')

    plt.figure()
    ax = sns.barplot(x="r2_test", y="ana", data=df_all, hue='FD_res')
    plt.tight_layout()
    plt.savefig('00_all_bars_r2.pdf')

good_ana = set(df.query('r2_test > .7').index)

df_good = df.loc[good_ana]
df_good.ix[:, 'ana'] = df_good.index.values
multi_index = pd.MultiIndex.from_tuples(zip(df_good['ana'], df_good['FD_res']), names=['ana_id', 'FD_red'])
df_good.set_index(multi_index, inplace=True)
df_good.sort_index(inplace=True)

def plot_bars(**kwargs):

    def format_ticks(ax):
        def break_text(s, n):
            o = []
            while s:
                o.append(s[:n])
                s = s[n:]
            o = '\n'.join(o)
            return o

        li = [break_text(l.get_text(), 50) for l in ax.get_yticklabels()]
        ax.set_yticklabels(li)
        return ax


    plt.figure()
    ax = sns.barplot(**kwargs)
    ax = format_ticks(ax)
    try:
        plt.tight_layout()
    except:
        pass


c = sns.color_palette()
plot_bars(x="MAE_test", y="ana", data=df_good, hue='FD_res')
plt.savefig('00_best_bars_MAE.pdf')


plt.figure()
plot_bars(x="r2_test", y="ana", data=df_good, hue='FD_res')
plt.savefig('00_best_bars_r2.pdf')

f, axarr = plt.subplots(len(good_ana), 2, sharex=True, sharey=True, figsize=(15, 35))
for row, a in enumerate(good_ana):
    if len(a) > 50:
        a_str = a[:50] + '\n' + a[50:]
    else:
        a_str = a

    ax = axarr[row, 0]
    sns.regplot(x='age', y='pred_age_test', data=scatter_df[False][a], ax=ax, color=c[0])
    ax.set_aspect('equal')
    ax.plot([10, 80], [10, 80], 'k')
    ax.set_title('%s\nr2 = %3.2f ' % (a_str, df[df.FD_res == False].loc[a]['r2_test']))

    ax = axarr[row, 1]
    sns.regplot(x='age', y='pred_age_test', data=scatter_df[True][a], ax=ax, color=c[1])
    ax.set_aspect('equal')
    ax.plot([10, 80], [10, 80], 'k')
    ax.set_title('%s\nr2 = %3.2f ' % (a_str, df[df.FD_res == True].loc[a]['r2_test']))

plt.tight_layout()
plt.savefig('00_best_scatters.pdf')
