__author__ = 'franzliem'

def r_to_z(r):
    import numpy as np
    m_r = r.copy()
    m_r[m_r == 1] = 1 - 1e-15
    m_r[m_r == -1] = -1 + 1e-15
    return np.arctanh(m_r)

def aggregate_data(file_list):
    '''
    tries to guess data type.
    loads data an concatenates it.
    returns merged file path
    '''

    import os
    import nibabel as nb
    import numpy as np

    def _merge_nii(file_list, out_filename):
        '''
        merges list of nii over subjects
        '''
        from nipype.pipeline.engine import Node, Workflow
        import nipype.interfaces.fsl as fsl

        # For long file lists, nipype throws an error if not split
        if len(file_list) < 1300:
            merge = Node(fsl.Merge(dimension='t'), name='merge')
            merge.base_dir = os.getcwd()
            merge.inputs.in_files = file_list
            merge.inputs.merged_file = out_filename
            result = merge.run()
        else:
            merge_1 = Node(fsl.Merge(dimension='t'), name='merge_1')
            merge_1.base_dir = os.getcwd()
            merge_1.inputs.in_files = file_list[:1300]
            merge_1.inputs.merged_file = 'merge_1.nii.gz'
            result_1 = merge_1.run()

            merge_2 = Node(fsl.Merge(dimension='t'), name='merge_2')
            merge_2.base_dir = os.getcwd()
            merge_2.inputs.in_files = file_list[1300:]
            merge_2.inputs.merged_file = 'merge_2.nii.gz'
            result_2 = merge_2.run()

            merge = Node(fsl.Merge(dimension='t'), name='merge')
            merge.base_dir = os.getcwd()
            merge.inputs.in_files = [result_1.outputs.merged_file, result_2.outputs.merged_file]
            merge.inputs.merged_file = out_filename
            result = merge.run()

        return result.outputs.merged_file

    def _merge_matrix(file_list, out_filename):
        '''
        merges list of pickled matrices so that -> matrix[subject,x,y]
        '''
        import pickle
        import numpy as np
        import os

        out_data = None

        for matrix_file in file_list:
            with open(matrix_file, 'r') as f:
                in_data = pickle.load(f)

            if out_data is None:
                out_data = {}
                for k in in_data.keys():
                    out_data[k] = in_data[k][np.newaxis, ...]
            else:
                for k in in_data.keys():
                    out_data[k] = np.concatenate((out_data[k], in_data[k][np.newaxis, ...]))

        full_out_filename = os.path.join(os.getcwd(), out_filename)
        with open(full_out_filename, 'w') as f:
            pickle.dump(out_data, f)

        return full_out_filename

    def _merge_fs(file_list, out_filename):
        '''
        merges list of fs surfs
        '''
        import numpy as np
        import os
        import nibabel as nb

        out_data = None

        for fs_file in file_list:
            img = nb.load(fs_file)
            in_data = img.get_data().squeeze()

            if out_data is None:
                out_data = in_data[np.newaxis, ...]
            else:
                out_data = np.concatenate((out_data, in_data[np.newaxis, ...]))

        full_out_filename = os.path.join(os.getcwd(), out_filename)
        np.save(full_out_filename, out_data)

        return full_out_filename

    def _merge_fs_tab(file_list, out_filename):
        '''
        merges list of fs_tab textfiles
        '''
        import numpy as np
        import os
        import pandas as pd

        out_data = None

        for i, tab_file in enumerate(file_list):
            df_single = pd.read_csv(tab_file, index_col=0, delimiter='\t')
            df_single.index.name = 'subject_id'

            if i == 0:
                df = df_single
            else:
                df = pd.concat([df, df_single])
        full_out_filename = os.path.join(os.getcwd(), out_filename)
        df.to_csv(full_out_filename)

        return full_out_filename

    out_filename = os.path.basename(file_list[0])
    if file_list[0].endswith('.nii.gz'):  # 3d nii files
        merged_file = _merge_nii(file_list, out_filename)

    elif file_list[0].endswith('.pkl'):  # pickled matrix files
        merged_file = _merge_matrix(file_list, out_filename)

    elif file_list[0].endswith('.mgz'):  # freesurfer surface files
        out_filename = os.path.splitext(os.path.basename(file_list[0]))[0] + '.npy'
        merged_file = _merge_fs(file_list, out_filename)

    elif out_filename.startswith('aseg') | out_filename.startswith('aparc'):
        merged_file = _merge_fs_tab(file_list, out_filename)

    else:
        raise Exception('Cannot guess type from filename: %s' % file_list[0])

    save_template = file_list[0]
    return merged_file, save_template


def vectorize_data(in_data_file, mask_file, matrix_name, parcellation_path, fwhm, use_diagonal=False, use_fishers_z=False):
    import os, pickle
    import numpy as np

    # LITTLE HELPERS
    def _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm):
        from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

        if parcellation_path is None:
            masker = NiftiMasker(mask_img=mask_file, smoothing_fwhm=fwhm)
        else:
            masker = NiftiLabelsMasker(labels_img=parcellation_path, smoothing_fwhm=fwhm)

        vectorized_data = masker.fit_transform(in_data_file)

        return vectorized_data, masker

    def _vectorize_matrix(in_data_file, matrix_name, use_diagonal=False, use_fishers_z=False):

        def _lower_tria_vector(m, use_diagonal=False):
            '''
            use_diagonal=False: returns lower triangle of matrix (without diagonale) as vector
            use_diagonal=True: returns lower triangle of matrix (with diagonale) as vector; e.g. for downsampled matrices
            matrix dims are x,y,subject
            '''
            i = np.ones_like(m).astype(np.bool)
            if use_diagonal:
                k = 0
            else:
                k = -1
            tril_ind = np.tril(i, k)
            vectorized_data = m[tril_ind].reshape(m.shape[0], -1)
            return vectorized_data

        # load pickled matrix
        with open(in_data_file, 'r') as f:
            matrix = pickle.load(f)

        # get lower triangle
        vectorized_data = _lower_tria_vector(matrix[matrix_name], use_diagonal=use_diagonal)

        if use_fishers_z:
            vectorized_data = r_to_z(vectorized_data)

        return vectorized_data

    def _vectorize_fs(in_data_file):
        # load fs matrix (is already vectorized)
        vectorized_data = np.load(in_data_file)
        return vectorized_data

    def _vectorize_fs_tab(in_data_file):
        import pandas as pd
        df = pd.read_csv(in_data_file, index_col=0)
        vectorized_data = df.values
        return vectorized_data

    # RUN
    masker = None

    if in_data_file.endswith('.nii.gz'):  # 3d nii files
        vectorized_data, masker = _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm)
        data_type = '3dnii'

    elif in_data_file.endswith('.pkl'):  # pickled matrix files
        vectorized_data = _vectorize_matrix(in_data_file, matrix_name)
        data_type = 'matrix'

    elif in_data_file.endswith('.npy'):  # freesurfer: already vetorized
        vectorized_data = _vectorize_fs(in_data_file)
        data_type = 'fs_cortical'

    elif os.path.basename(in_data_file).startswith('aseg') | os.path.basename(in_data_file).startswith('aparc'):  # aseg: just export values from df
        vectorized_data = _vectorize_fs_tab(in_data_file)
        data_type = 'fs_tab'

    else:
        raise Exception('Cannot guess type from filename: %s' % in_data_file)

    vectorized_data_file = os.path.join(os.getcwd(), 'vectorized_data.npy')

    np.save(vectorized_data_file, vectorized_data)
    return vectorized_data, vectorized_data_file, data_type, masker


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

    # step_ind = {}
    # if isinstance(trained_model, sklearn.grid_search.GridSearchCV): #necessary because gs object is set up diffrently
    #     steps = trained_model.best_estimator_.steps
    # else:
    #     steps = trained_model.steps
    #
    # for step_no, step in enumerate(steps):
    #     step_name, step_content = step
    #     step_ind[step_name] = step_no
    #
    # imp = steps[step_ind['fill_missing']][1]
    # var_thr = steps[step_ind['var_thr']][1]
    # scaler = steps[step_ind['normalize']][1]
    # regression_model = steps[step_ind['regression_model']][1]

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

    elif data_type == 'fs_tab':
        outfile = os.path.abspath(outfile_name + weights_file_str + '.csv')
        df = pd.read_csv(save_template, index_col=0, delimiter='\t')
        df.index.name = 'subject_id'
        df.drop(df.index[0], axis=0, inplace=True)
        df.loc[outfile_name, :] = data
        df.to_csv(outfile)
        df.T.plot(kind='barh')

    else:
        raise Exception('wrong data type %s' % data_type)

    plt.savefig(outfile_render)
    plt.close()

    return outfile, outfile_render
