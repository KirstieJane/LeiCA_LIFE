#!/bin/sh


PYTHONPATH=${PYTHONPATH}:/home/raid2/liem/PowerFolders/Workspace:/home/raid2/liem/PowerFolders/Workspace/LeiCA_LIFE:/home/raid2/liem/PowerFolders/Workspace/LeiCA_LIFE/optunity/
#:/home/raid2/liem/PowerFolders/Workspace/Bayesian-Regression-Methods/Relevance\ Vector\ Machine\ \&\ ARD/
export PYTHONPATH


#make sure that DCMSTACK is called AFTER CPAC
FSL --version 5.0 FREESURFER --version 5.3.0 AFNI CPAC DCMSTACK PANDAS NILEARN --version 0.1.4 SKLEARN MATPLOTLIB NUMPY SEABORN
