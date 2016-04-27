import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedKFold, cross_val_score, StratifiedShuffleSplit, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

root_path='/Users/franzliem/PowerFolders/00_Print/tmp/learning_out_20160419/pdfs/scaler_standard_rfe_False_strat_False_reg_False_predicted'
source_dict = {
    'aseg': os.path.join(root_path, 'age__bothSexes_neuH_FD06__aseg_df_predicted.pkl'),
    'ct':  os.path.join(root_path, 'age__bothSexes_neuH_FD06__lh_ct_fsav4_sm0__rh_ct_fsav4_sm0_df_predicted.pkl'),
    'csa':  os.path.join(root_path, 'age__bothSexes_neuH_FD06__lh_csa_fsav4_sm0__rh_csa_fsav4_sm0_df_predicted.pkl'),
    'basc197':  os.path.join(root_path, 'age__bothSexes_neuH_FD06__basc_197_df_predicted.pkl'),
    'basc444':  os.path.join(root_path, 'age__bothSexes_neuH_FD06__basc_444_df_predicted.pkl'),
}


# ren_list = ['pred_age_train', 'pred_age_test', 'no_motion_grp', 'random_motion_grp', 'pred_age_no_motion',
#             'pred_age_random_motion', 'y_predicted_cv']

df_in = {}
df_all = None
for s, f in source_dict.items():
    df_in[s] = pd.read_pickle(f)
    df_in[s]['source'] = s

    if df_all is None:
        df_all = df_in[s]
        df_single_source = df_in[s]
    else:
        df_all = pd.concat((df_all, df_in[s]))

df = df_all[['source', 'age', 'split_group', 'select', 'y_predicted_cv', 'pred_age_test']]
df_train = df.query("split_group=='train'").copy()
df_test = df.query("split_group=='test'").copy()

dd_train = df_train.pivot_table(values='y_predicted_cv', columns='source', index=df_train.index)
single_sources = dd_train.columns.values
dd_train['mean_pred'] = dd_train.mean(1)
dd_train = dd_train.join(df_single_source[['age']], how='left')

dd_test = df_test.pivot_table(values='pred_age_test', columns='source', index=df_test.index)
dd_test['mean_pred'] = dd_test.mean(1)
dd_test = dd_test.join(df_single_source[['age']], how='left')

print(dd_train.corr())
sns.pairplot(dd_train);#plt.show()


n_age_bins = 20
dd_train['age_bins_rf'] = pd.cut(dd_train['age'], n_age_bins, labels=range(n_age_bins))

strat_k_fold = StratifiedKFold(dd_train['age_bins_rf'].values, n_folds=20, shuffle=True, random_state=0)
sss = StratifiedShuffleSplit(dd_train['age_bins_rf'].values, n_iter=20, test_size=.2, random_state=0)

X_train, y_train = dd_train[single_sources], dd_train['age']
X_test, y_test = dd_test[single_sources], dd_test['age']

cv_r2 = []
cv_list = [None] + range(1,20)
for md in cv_list:
    rf = RandomForestRegressor(n_estimators=50, max_depth=md, oob_score=True, random_state=0, n_jobs=-1)
    #cv_pred = cross_val_predict(rf, X_train, y_train, cv=strat_k_fold)
    #dd_train['rf_pred'] = cv_pred
    rf.fit( X_train, y_train)
    cv_r2.append(rf.oob_score_)
    #cv_r2.append(cross_val_score(rf, X_train, y_train, cv=strat_k_fold))

best_max_depth = cv_list[np.argmax(cv_r2)]
print("best max_depth: ",best_max_depth )

# refit
rf = RandomForestRegressor(n_estimators=50, max_depth=best_max_depth, oob_score=True, random_state=0, n_jobs=-1)

rf.fit(X_train, y_train)


y_test_predicted = rf.predict(X_test)
dd_test['rf_pred'] = y_test_predicted

print("##################################################")
print("TRAIN\n")
print(dd_train.corr())
print("##################################################")
print("TEST\n")
corr_test = dd_test.corr()
print(corr_test)


r2_test = pd.DataFrame([])
r2_test['r2']=np.nan
r2_test['rpear']=np.nan
r2_test['rpear2']=np.nan
r2_test['mae']=np.nan
r2_test['medae']=np.nan
for m in source_dict.keys() + ['mean_pred','rf_pred']:
    r2_test.ix[m, 'r2'] = r2_score(dd_test['age'], dd_test[m])
    r2_test.ix[m, 'rpear'] = np.corrcoef(dd_test['age'], dd_test[m])[0,1]
    r2_test.ix[m, 'rpear2'] = np.corrcoef(dd_test['age'], dd_test[m])[0,1]**2
    r2_test.ix[m, 'mae'] = mean_absolute_error(dd_test['age'], dd_test[m])
    r2_test.ix[m, 'medae'] = median_absolute_error(dd_test['age'], dd_test[m])
r2_test.to_clipboard()

