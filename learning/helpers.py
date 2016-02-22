

# MERGE DATA HELPER FUNCTIONS
def _merge_nii(file_list, out_filename):
    '''
    merges list of nii over subjects
    '''
    # For long file lists, nipype throws an error if not split
    from nipype.interfaces import fsl as fsl
    from nipype import Node
    import os
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
    import numpy as np
    import pandas as pd
    import pickle
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
    import nibabel as nb
    import os
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
    import pandas as pd
    import os
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

        if m.ndim == 3: # subjects alredy aggregated
            vectorized_data = m[tril_ind].reshape(m.shape[0], -1)
        else: # single subject matrix
            vectorized_data = m[tril_ind]
        return vectorized_data

    # load pickled matrix
    with open(in_data_file, 'r') as f:
        matrix = pickle.load(f)

    # get lower triangle
    vectorized_data = _lower_tria_vector(matrix[matrix_name], use_diagonal=use_diagonal)

    return vectorized_data


def _vectorize_fs(in_data_file):
    # load fs matrix (is already vectorized)
    import numpy as np
    vectorized_data = np.load(in_data_file)
    return vectorized_data


def _vectorize_fs_tab(in_data_file):
    import pandas as pd
    df = pd.read_csv(in_data_file, index_col=0)
    vectorized_data = df.values
    return vectorized_data


def _vectorize_behav_df(in_data_file, df_col_names):
    import pandas as pd
    df = pd.read_pickle(in_data_file)
    vectorized_data = df[df_col_names].values
    return vectorized_data
