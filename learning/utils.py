def vectorize_ss(in_data_file, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal, use_fishers_z, df_file,
                 df_col_names):
    import os, pickle
    import numpy as np
    from learning.vectorize_helpers import _vectorize_nii, _vectorize_matrix, _vectorize_fs, _vectorize_fs_tab, \
        _vectorize_behav_df

    save_template = in_data_file

    masker = None
    if in_data_file.endswith('.nii.gz'):  # 3d nii files
        vectorized_data, masker = _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm)
        data_type = '3dnii'

    elif in_data_file.endswith('.pkl') & (df_col_names is None):  # pickled matrix files
        vectorized_data = _vectorize_matrix(in_data_file, matrix_name, use_diagonal)
        data_type = 'matrix'

    elif in_data_file.endswith('.mgz'):  # freesurfer: already vetorized
        vectorized_data = _vectorize_fs(in_data_file)
        data_type = 'fs_cortical'

    elif os.path.basename(in_data_file).startswith('aseg') | os.path.basename(in_data_file).startswith(
            'aparc'):  # aseg: just export values from df
        vectorized_data = _vectorize_fs_tab(in_data_file)
        data_type = 'fs_tab'

    elif df_col_names:  # X from df behav
        # subject is inputted via in_data_file
        vectorized_data, save_template = _vectorize_behav_df(df_file=df_file, subject=in_data_file,
                                                             df_col_names=df_col_names)
        data_type = 'behav'

    else:
        raise Exception('Cannot guess type from filename: %s' % in_data_file)

    def r_to_z(r):
        r = np.atleast_1d(r)
        r[r == 1] = 1 - 1e-15
        r[r == -1] = -1 + 1e-15
        return np.arctanh(r)

    if use_fishers_z:
        vectorized_data = r_to_z(vectorized_data)

    vectorized_data = np.atleast_2d(vectorized_data)
    return vectorized_data, data_type, masker, save_template


def vectorize_and_aggregate(in_data_file_list, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal,
                            use_fishers_z, df_file, df_col_names):
    import os, pickle
    import numpy as np
    from learning.utils import vectorize_ss

    # get an example of the data:
    vectorized_data, data_type, masker, save_template = vectorize_ss(in_data_file_list[0], mask_file, matrix_name,
                                                                     parcellation_path, fwhm, use_diagonal,
                                                                     use_fishers_z, df_file,
                                                                     df_col_names)
    vectorized_data = np.zeros((len(in_data_file_list), vectorized_data.shape[1]))
    vectorized_data.fill(np.nan)

    for i, in_data_file_ss in enumerate(in_data_file_list):
        vectorized_data[i, :], _, _, _ = vectorize_ss(in_data_file_ss, mask_file, matrix_name, parcellation_path, fwhm,
                                                      use_diagonal, use_fishers_z, df_file, df_col_names)

    vectorized_aggregated_file = os.path.abspath('vectorized_aggregated_data.npy')
    np.save(vectorized_aggregated_file, vectorized_data)

    unimodal_backprojection_info = {'data_type': data_type,
                                   'masker': masker,
                                   'save_template': save_template}
    unimodal_backprojection_info_file = os.path.abspath('unimodal_backprojection_info.pkl')
    pickle.dump(unimodal_backprojection_info, open(unimodal_backprojection_info_file, 'w'))
    return vectorized_aggregated_file, unimodal_backprojection_info_file


def aggregate_data(file_list):
    '''
    tries to guess data type.
    loads data an concatenates it.
    returns merged file path

    special case behav files (detected by df_col_names not None):
    here the data already is aggregated; just pipe through (save)
    '''
    import os
    import nibabel as nb
    import numpy as np

    out_data = None
    for file_ss in file_list:
        in_data_ss = np.atleast_2d(np.load(file_ss))
        if out_data is None:
            out_data = in_data_ss
        else:
            out_data = np.concatenate((out_data, in_data_ss))

    merged_file = os.path.abspath('vectorized_aggregated_data.npy')
    np.save(merged_file, out_data)
    return merged_file


#
# def vectorize_data_old(in_data_file, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal,
#                    use_fishers_z, df_file, df_col_names):
#     import os, pickle
#     import numpy as np
#     from learning.vectorize_helpers import _vectorize_nii, _vectorize_matrix, _vectorize_fs, _vectorize_fs_tab, \
#         _vectorize_behav_df
#
#     save_template = in_data_file
#
#     masker = None
#     if in_data_file.endswith('.nii.gz'):  # 3d nii files
#         vectorized_data, masker = _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm)
#         data_type = '3dnii'
#
#     elif in_data_file.endswith('.pkl') & (df_col_names is None):  # pickled matrix files
#         vectorized_data = _vectorize_matrix(in_data_file, matrix_name, use_diagonal)
#         data_type = 'matrix'
#
#     elif in_data_file.endswith('.mgz'):  # freesurfer: already vetorized
#         vectorized_data = _vectorize_fs(in_data_file)
#         data_type = 'fs_cortical'
#
#     elif os.path.basename(in_data_file).startswith('aseg') | os.path.basename(in_data_file).startswith(
#             'aparc'):  # aseg: just export values from df
#         vectorized_data = _vectorize_fs_tab(in_data_file)
#         data_type = 'fs_tab'
#
#     elif df_col_names:  # X from df behav
#         # subject is inputted via in_data_file
#         vectorized_data, save_template = _vectorize_behav_df(df_file=df_file, subject=in_data_file, df_col_names=df_col_names)
#         data_type = 'behav'
#
#     else:
#         raise Exception('Cannot guess type from filename: %s' % in_data_file)
#
#     def r_to_z(r):
#         r = np.atleast_1d(r)
#         r[r == 1] = 1 - 1e-15
#         r[r == -1] = -1 + 1e-15
#         return np.arctanh(r)
#
#     if use_fishers_z:
#         vectorized_data = r_to_z(vectorized_data)
#
#     vectorized_data = np.atleast_2d(vectorized_data)
#
#     vectorized_data_file = os.path.join(os.getcwd(), 'vectorized_data.npy')
#     np.save(vectorized_data_file, vectorized_data)
#     return vectorized_data_file, data_type, masker, save_template


def pred_real_scatter(y_test, y_test_predicted, title_str, in_data_name):
    import os
    import pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.figure()
    plt.scatter(y_test, y_test_predicted)
    plt.plot([10, 80], [10, 80], 'k')
    plt.xlabel('real')
    plt.ylabel('predicted')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_' + in_data_name + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


def plot_brain_age(y_test, y_test_predicted, in_data_name):
    import os
    import pylab as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages

    brain_age = y_test_predicted - y_test
    title_str = 'mean brain age: %s\nr = %s' % (brain_age.mean(), np.corrcoef(brain_age, y_test)[0, 1])
    plt.figure()
    plt.scatter(y_test, brain_age)
    plt.xlabel('real')
    plt.ylabel('brain_age (pred-real)')
    plt.title(title_str)
    plt.tight_layout()
    scatter_file = os.path.join(os.getcwd(), 'scatter_brain_age_' + in_data_name + '.pdf')
    pp = PdfPages(scatter_file)
    pp.savefig()
    pp.close()
    return scatter_file


def residualize_group_data(signals, confounds):
    '''
    regresses out confounds from signals
    signals.shape: subjects x n_data_points
    confounds.shape: subjects x n_confounds
    returns residualized_signals.shape: subjects x n_data_points
    '''
    from nilearn.signal import clean
    residualized_signals = clean(signals, detrend=False, standardize=False, confounds=confounds, low_pass=None,
                                 high_pass=None, t_r=None)
    return residualized_signals


def backproject_weights_to_full_space(trained_model_file):
    '''
    Takes a vector of weights and fills NaNs for all the features that have been eliminated during preprocessing
    '''
    import pickle
    import numpy as np
    import sklearn

    ADD_CONSTANT = 1000
    trained_model = pickle.load(open(trained_model_file))

    if isinstance(trained_model, sklearn.grid_search.GridSearchCV):  # necessary because gs object is set up diffrently
        steps = trained_model.best_estimator_.named_steps
    else:
        steps = trained_model.named_steps

    imp = steps['fill_missing']
    var_thr = steps['var_thr']
    scaler = steps['normalize']

    if 'rfe' in steps.keys():
        rfe = steps['rfe']
        regression_model = rfe.estimator_
        # back trainsform weights from sparse (rfe) to full
        # insert nans in eliminated features
        weights = rfe.inverse_transform(regression_model.coef_ + ADD_CONSTANT)
        sparse_ind = (weights == 0)
        weights -= ADD_CONSTANT

    elif 'regression_model' in steps.keys():
        regression_model = steps['regression_model']
        weights = regression_model.coef_
        sparse_ind = np.zeros_like(weights).astype(np.bool)

    else:
        raise Exception('neither rfe nor regression_model found in steps.keys() %s' % steps.keys())

    # add constant to the weights in order to differentiate between 0 weights and zeros that have been iputed in the
    # case of 0-variance features
    # after replacing 0s with nans, substract constant
    mapped_back_var_thr = var_thr.inverse_transform(weights + ADD_CONSTANT)
    mapped_back_var_thr[mapped_back_var_thr == 0] = np.nan
    mapped_back_var_thr -= ADD_CONSTANT

    mapped_back = np.zeros((1, imp.statistics_.shape[0]))
    mapped_back.fill(np.nan)
    mapped_back[0, ~np.isnan(imp.statistics_)] = mapped_back_var_thr

    # set sparse weights to nan
    mapped_back[sparse_ind] = np.nan
    return mapped_back


def vector_to_matrix(v, use_diagonal=False):
    '''
    Takes vector of length vector_size and creates 2D matrix to fill off-diagonal cells
    vector_size = matrix_size*(matrix_size-1)*.5
    matrix diagonal is set to 0
    '''
    import numpy as np
    vector_size = v.shape[0]
    if use_diagonal:
        diag_add = -1
        k = 0
    else:
        diag_add = 1
        k = -1

    matrix_size = int(0.5 * (np.sqrt(8 * vector_size + 1) + diag_add))

    m = np.zeros((matrix_size, matrix_size))
    i = np.ones_like(m).astype(np.bool)

    tril_ind = np.tril(i, k)
    m[tril_ind] = v
    m_sym = m + m.T

    return m_sym


def matrix_to_vector(m):
    '''
    returns lower triangle of 2D matrix (without diagonale) as vector
    '''
    import numpy as np
    i = np.ones_like(m).astype(np.bool)
    tril_ind = np.tril(i, -1)
    v = m[tril_ind]
    return v


def test_vector_to_matrix():
    '''
    tests vector_to_matrix() and matrix_to_vector()
    '''
    import numpy as np
    # simulate data
    matrix_size = 200
    m_in = np.random.randn(matrix_size, matrix_size)
    m_in_sym = m_in + m_in.T
    np.fill_diagonal(m_in_sym, 0)
    v = matrix_to_vector(m_in_sym)
    m_out = vector_to_matrix(v)
    assert np.all(m_out == m_in_sym), "test of vector_to_matrix failed"
    assert m_out[2, 3] == m_out[3, 2], "out matix not symmetrical"


def save_weights(data, data_type, save_template, outfile_name, masker=None):
    '''
    saves weights as nii or other file
    '''
    import nibabel as nb
    import numpy as np
    import pandas as pd
    import os
    from nilearn import plotting
    import pylab as plt

    weights_file_str = '_weights'
    render_file_str = '_rendering'
    outfile_render = os.path.abspath(outfile_name + render_file_str + '.pdf')

    plt.figure()

    if data_type == '3dnii':
        nii = masker.inverse_transform(data)
        outfile = os.path.abspath(outfile_name + weights_file_str + '.nii.gz')
        nii.to_filename(outfile)

        plotting.plot_stat_map(outfile)


    elif data_type == 'matrix':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.npy')
        np.save(outfile, data)

        plt.imshow(data, interpolation='nearest')
        plt.colorbar()

    elif data_type == 'fs_cortical':
        fs_img = nb.load(save_template)
        # Sabina says that's how nb handels this case
        fs_data = fs_img.get_data()
        fs_data[:] = data.reshape(fs_data.shape[0], 1, 1)
        # check whether writing data into image worked
        np.testing.assert_array_almost_equal(fs_img.get_data().squeeze(), data.squeeze())

        outfile = os.path.abspath(outfile_name + weights_file_str + '.mgz')
        fs_img.to_filename(outfile)

        # # fixme DOES NOT WORK
        # from surfer import Brain
        # # replace nans by 0, otherwise pysurfer cannot display them
        # data_nonans = data.copy()
        # data_nonans[np.isnan(data_nonans)] = 0
        # hemi =  outfile_name[outfile_name.find('__')+2:][:2]
        # brain = Brain('fsaverage5', hemi, 'inflated', config_opts={"background": "white"}) #, background="white") ,
        # brain.add_data(data_nonans.squeeze())
        # brain.save_montage(outfile_render, order=['lat', 'med'], orientation='h', border_size=10)
        # brain.close()

    elif (data_type == 'fs_tab'):
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        df = pd.read_csv(save_template, index_col=0, delimiter='\t')
        df.index.name = 'subject_id'
        df.drop(df.index[0], axis=0, inplace=True)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    elif data_type == 'behav':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        df = pd.read_pickle(save_template)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    else:
        raise Exception('wrong data type %s' % data_type)

    plt.savefig(outfile_render)
    plt.close()

    return outfile, outfile_render
