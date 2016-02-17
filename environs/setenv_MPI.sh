#!/bin/sh


PYTHONPATH=${PYTHONPATH}:/home/raid2/liem/Dropbox/Workspace:/home/raid2/liem/Dropbox/Workspace/LeiCA_LIFE:/home/raid2/liem/Dropbox/Workspace/LeiCA_LIFE/optunity/:/home/raid2/liem/Dropbox/Workspace/Bayesian-Regression-Methods/Relevance\ Vector\ Machine\ \&\ ARD/
export PYTHONPATH


#make sure that DCMSTACK is called AFTER CPAC
FSL --version 5.0 FREESURFER --version 5.3.0 AFNI CPAC DCMSTACK PANDAS NILEARN SKLEARN MATPLOTLIB NUMPY SEABORN
