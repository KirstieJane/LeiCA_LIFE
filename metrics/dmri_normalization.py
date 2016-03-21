def run_dmri_preprocessing_wf(subjects_list,
                              working_dir,
                              ds_dir,
                              use_n_procs,
                              plugin_name,
                              dMRI_templates,
                              in_path):
    import os
    from nipype import config
    from nipype.pipeline.engine import Node, Workflow
    import nipype.interfaces.utility as util
    import nipype.interfaces.io as nio
    from nipype.interfaces import fsl
    from moco_ecc import create_moco_ecc

    def eddy_rotate_bvecs(in_bvec, eddy_params):
        """
        Rotates the input bvec file accordingly with a list of parameters sourced
        from ``eddy``, as explained `here
        <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/EDDY/Faq#Will_eddy_rotate_my_bevcs_for_me.3F>`_.
        """
        import os
        import numpy as np
        from math import sin, cos

        name, fext = os.path.splitext(os.path.basename(in_bvec))
        if fext == '.gz':
            name, _ = os.path.splitext(name)
        out_file = os.path.abspath('%s_rotated.bvec' % name)
        bvecs = np.loadtxt(in_bvec).T
        new_bvecs = []

        params = np.loadtxt(eddy_params)

        if len(bvecs) != len(params):
            raise RuntimeError(('Number of b-vectors and rotation '
                                'matrices should match.'))

        for bvec, row in zip(bvecs, params):
            if np.all(bvec == 0.0):
                new_bvecs.append(bvec)
            else:
                ax = row[3]
                ay = row[4]
                az = row[5]

                Rx = np.array([[1.0, 0.0, 0.0],
                               [0.0, cos(ax), -sin(ax)],
                               [0.0, sin(ax), cos(ax)]])
                Ry = np.array([[cos(ay), 0.0, sin(ay)],
                               [0.0, 1.0, 0.0],
                               [-sin(ay), 0.0, cos(ay)]])
                Rz = np.array([[cos(az), -sin(az), 0.0],
                               [sin(az), cos(az), 0.0],
                               [0.0, 0.0, 1.0]])
                R = Rx.dot(Ry).dot(Rz)

                invrot = np.linalg.inv(R)
                newbvec = invrot.dot(bvec)
                new_bvecs.append(newbvec / np.linalg.norm(newbvec))

        np.savetxt(out_file, np.array(new_bvecs).T, fmt='%0.15f')
        return out_file

    #####################################
    # GENERAL SETTINGS
    #####################################
    wf = Workflow(name='dmri_prep_wf')
    wf.base_dir = os.path.join(working_dir)

    nipype_cfg = dict(logging=dict(workflow_level='DEBUG'), execution={'stop_on_first_crash': True,
                                                                       'remove_unnecessary_outputs': False,
                                                                       'job_finished_timeout': 15})
    config.update_config(nipype_cfg)
    wf.config['execution']['crashdump_dir'] = os.path.join(working_dir, 'crash')

    ds = Node(nio.DataSink(), name='ds')

    ds.inputs.regexp_substitutions = [
        ('_subject_id_[A0-9]*/', ''),
    ]

    subjects_infosource = Node(util.IdentityInterface(fields=['subject_id']),
                               name='subjects_infosource')
    subjects_infosource.iterables = ('subject_id', subjects_list)

    def add_subject_id_to_ds_dir_fct(subject_id, ds_dir):
        import os
        out_path = os.path.join(ds_dir, subject_id, 'dmri_preprocessing')
        return out_path

    wf.connect(subjects_infosource, ('subject_id', add_subject_id_to_ds_dir_fct, ds_dir), ds, 'base_directory')


    # GET SUBJECT SPECIFIC FUNCTIONAL DATA
    selectfiles = Node(nio.SelectFiles(dMRI_templates, base_directory=in_path), name="selectfiles")
    wf.connect(subjects_infosource, 'subject_id', selectfiles, 'subject_id')

    #####################################
    # WF
    #####################################


    moco_ecc = create_moco_ecc('moco_ecc')
    wf.connect(selectfiles, 'dMRI_data', moco_ecc, 'inputnode.in_file')
    wf.connect(selectfiles, 'bval_file', moco_ecc, 'inputnode.bval_file')
    moco_ecc.inputs.inputnode.nodiff_b = 5

    node_name = 'moco'
    wf.connect([
        (moco_ecc, ds, [#('outputnode.moco_out_file', node_name + '.@moco_out'),
                    ('outputnode.par_file', node_name + '.@par'),
                    ('outputnode.rms_files', node_name + '.@rms'),
                    ('outputnode.mat_files', node_name + '.mat.@mats'),
                    ('outputnode.ecc_out_file', node_name + '.@ecc'),
                    ('outputnode.mask_file', node_name + '.@mask')])
    ])


    # DTIFIT
    dtifit = Node(interface=fsl.DTIFit(), name='dtifit')
    wf.connect(moco_ecc, 'outputnode.ecc_out_file', dtifit, 'dwi')
    wf.connect(moco_ecc, 'outputnode.mask_file', dtifit, 'mask')
    wf.connect(selectfiles, 'bvec_file', dtifit, 'bvecs')
    wf.connect(selectfiles, 'bval_file', dtifit, 'bvals')

    wf.connect(dtifit, 'FA', ds, 'dtifit.@FA')
    wf.connect(dtifit, 'L1', ds, 'dtifit.@L1')
    wf.connect(dtifit, 'L2', ds, 'dtifit.@L2')
    wf.connect(dtifit, 'L3', ds, 'dtifit.@L3')
    wf.connect(dtifit, 'MD', ds, 'dtifit.@MD')
    wf.connect(dtifit, 'MO', ds, 'dtifit.@MO')
    wf.connect(dtifit, 'S0', ds, 'dtifit.@S0')
    wf.connect(dtifit, 'V1', ds, 'dtifit.@V1')
    wf.connect(dtifit, 'V2', ds, 'dtifit.@V2')
    wf.connect(dtifit, 'V3', ds, 'dtifit.@V3')

    # mni_fa = create_normalize_pipeline('mni_fa')
    # mni_fa.inputs.inputnode.standard = fsl.Info.standard_image('FMRIB58_FA_1mm.nii.gz')
    # wf.connect(dtifit, 'FA', mni_fa, 'inputnode.anat')
    #
    # wf.connect([(mni_fa, ds, [('outputnode.anat2std', 'mni_Aregistration.@anat2std'),
    #                           ('outputnode.anat2std_transforms', 'mni_Aregistration.transforms2mni.@anat2std_transforms'),
    #                           ('outputnode.std2anat_transforms', 'mni_Aregistration.transforms2mni.@std2anat_transforms')])
    #             ])

    def mni_sym_fct(f, m, o='mni_'):
        import os, glob
        cmd = 'antsRegistrationSyN.sh -d 3 -f {f} -m {m} -o {o}'.format(f=f, m=m, o=o)
        os.system(cmd)
        out_list = glob.glob(o + '*')
        out_files = [os.path.abspath(i) for i in out_list]
        return out_files

    mni_sym = Node(util.Function(input_names=['f', 'm', 'o'], output_names=['out_files'], function=mni_sym_fct),
                   name='mni_sym')
    mni_sym.inputs.f = fsl.Info.standard_image('FMRIB58_FA_1mm.nii.gz')
    mni_sym.inputs.o = 'mni_'
    wf.connect(dtifit, 'FA', mni_sym, 'm')

    wf.connect(mni_sym, 'out_files', ds, 'mni_Asyn.@files')

    # from dMRI.interfaces.syn_test import RegistrationSynQuick
    # ants_syn_quick = Node(RegistrationSynQuick, name='ants_syn_quick')
    # ants_syn_quick.inputs.fixed_image = fsl.Info.standard_image('FMRIB58_FA_1mm.nii.gz')
    # wf.connect(dtifit, 'FA', ants_syn_quick, 'moving_image')
    # ants_syn_quick.inputs.output_prefix = 'mni_syn_quick'
    #
    # wf.connect([(ants_syn_quick, ds, [('warped_image', 'mni_syn_quick.@warped_image'),  ])
    #             ])

    # rotate_bvecs = Node(util.Function(input_names=['in_bvec', 'eddy_params'], output_names=['rotated_bvecs'],
    #                                   function=eddy_rotate_bvecs), name='rotate_bvecs')
    # wf.connect(selectfiles, 'bvec_file', rotate_bvecs, 'in_bvec')
    # wf.connect(moco, 'outputnode.mat_files', rotate_bvecs, 'eddy_params')
    # wf.connect(rotate_bvecs, 'rotated_bvecs', ds, 'rotated_bvecs')


    #####################################
    # RUN WF
    #####################################
    # wf.write_graph(dotfilename=wf.name, graph2use='colored', format='pdf')  # 'hierarchical')
    # wf.write_graph(dotfilename=wf.name, graph2use='orig', format='pdf')
    # wf.write_graph(dotfilename=wf.name, graph2use='flat', format='pdf')

    if plugin_name == 'CondorDAGMan':
        wf.run(plugin=plugin_name, plugin_args={'initial_specs': 'request_memory = 1500'})
    if plugin_name == 'MultiProc':
        wf.run(plugin=plugin_name, plugin_args={'n_procs': use_n_procs})  #
