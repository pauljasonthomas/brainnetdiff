# BrainNetDiff
### Description
This repository contains code for multimodal brain network diffusion-based analysis as in Thomas et. al., 2021, along with a jupyter notebook tutorial (`brainnetdiff/example.ipynb`) demonstrating how to use the package and reproduce figures from the paper (in progress). 

### Data
Data used in Thomas et. al., 2021 is found in `brainnetdiff/data`. These data include preprocessed resting state fMRI and DTI adjacency matrix connectome arrays (FC.npy and SC.npy, respectively) and csv files containing demographics and baseline IDAS-II scales and percentage change in IDAS-II scales following treatment (demo_idas_baseline.csv and idas_treatment_response.csv, respectively) (in progress). Also included is a csv containing MMP1.0 atlas (mmp_atlas.csv) data and a .npy file containing xyz coordinates for ROIs from this parcellation (mmp_xyz.npy).

### Plotting with `nilearn.plotting.plot_connectome()`
In order to use the plotting methods found in `brainnetdiff/brainnetdiff/vis`, a custom version of nilearn, modified to allow for the passage of egde related colormap feature, is necessary. The modified nilean package python files that are needed for this are found in `brainnetdiff/nilearn_mod` (in progress). 

### Reference
```
@article{thomas2021network,
  title={Network Diffusion Embedding Reveals Transdiagnostic Subnetwork Disruption and Potential Treatment Targets in Internalizing Psychopathologies},
  author={Thomas, Paul J and Leow, Alex and Klumpp, Heide and Phan, K Luan and Ajilore, Olusola},
  journal={medrXiv preprint},
  year={2021}
}
```
