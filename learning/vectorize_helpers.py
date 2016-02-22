# VECTORIZE DATA helper functions

def _vectorize_nii(in_data_file, mask_file, parcellation_path, fwhm):
    import numpy as np
    import pandas as pd
    import os
    from nilearn.input_data import NiftiMasker, NiftiLabelsMasker

    if parcellation_path is None:
        masker = NiftiMasker(mask_img=mask_file, smoothing_fwhm=fwhm)
    else:
        masker = NiftiLabelsMasker(labels_img=parcellation_path, smoothing_fwhm=fwhm)

    vectorized_data = masker.fit_transform(in_data_file)

    return vectorized_data, masker


def _vectorize_matrix(in_data_file, matrix_name, use_diagonal=False):
    import numpy as np
    import pickle
    import os
    def _lower_tria_vector(m, use_diagonal=False):
        '''
        use_diagonal=False: returns lower triangle of matrix (without diagonale) as vector
        use_diagonal=True: returns lower triangle of matrix (with diagonale) as vector; e.g. for downsampled matrices
        matrix dims are
            x,y,subject for aggregated data
            or  x,y for single subject data
        '''
        i = np.ones_like(m).astype(np.bool)
        if use_diagonal:
            k = 0
        else:
            k = -1
        tril_ind = np.tril(i, k)

        if m.ndim == 3:  # subjects alredy aggregated
            vectorized_data = m[tril_ind].reshape(m.shape[0], -1)
        else:  # single subject matrix
            vectorized_data = m[tril_ind]
        return vectorized_data

    # load pickled matrix
    with open(in_data_file, 'r') as f:
        matrix = pickle.load(f)

    # get lower triangle
    vectorized_data = _lower_tria_vector(matrix[matrix_name], use_diagonal=use_diagonal)

    return vectorized_data


def _vectorize_fs(in_data_file):
    import numpy as np
    import nibabel as nb

    img = nb.load(in_data_file)
    in_data = img.get_data().squeeze()
    vectorized_data = in_data[np.newaxis, ...]
    return vectorized_data


def _vectorize_fs_tab(in_data_file):
    import pandas as pd
    df = pd.read_csv(in_data_file, index_col=0, delimiter='\t')
    vectorized_data = df.values
    return vectorized_data


def _vectorize_behav_df(df_file, subject, df_col_names):
    import pandas as pd
    df = pd.read_pickle(df_file)
    vectorized_data = df.loc[subject][df_col_names].values
    return vectorized_data
