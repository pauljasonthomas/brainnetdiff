# BrainNetDiff
### Description
This repository contains code for multimodal brain network diffusion-based analysis as in Thomas et al, 2021, along with a jupyter notebook tutorial (`brainnetdiff/example.ipynb`) demonstrating how to use the package and reproduce figures from the paper. 

### Data
Data used in Thomas et al, 2021 is found in `brainnetdiff/data`. These data include csv files containing demographics and baseline IDAS-II scales and percentage change in IDAS-II scales following treatment (demo_idas_baseline.csv and idas_treatment_response.csv, respectively) (in progress). Also included is a csv containing MMP1.0 atlas (mmp_atlas.csv) data and a .npy file containing xyz coordinates for ROIs from this parcellation (mmp_xyz.npy). Preprocessed resting state fMRI and DTI adjacency matrix connectome arrays (FC.npy and SC.npy, respectively) are too large (~100MB each) to add directly to this repository. As of now, please contact me directly for access to these files (pthoma4 at uic dot edu). 

### Plotting with `nilearn.plotting.plot_connectome()`
In order to use the plotting methods found in `brainnetdiff/brainnetdiff/vis`, a custom version of nilearn, modified to allow for the passage of egde related colormap feature, is necessary. The modified nilean package python files that are needed for this are found in `brainnetdiff/nilearn_mod`. You must navigate to where these files are installed ('path/to/nilearn/plotting/) and replace the identically named files with those in `brainnetdiff/nilearn_mod`. 

### Reference
```
@article {thomas2020network,
	author = {Thomas, Paul J. and Leow, Alex and Klumpp, Heide and Phan, K. Luan and Ajilore, Olusola},
	title = {Network Diffusion Embedding Reveals Transdiagnostic Subnetwork Disruption and Potential Treatment Targets in Internalizing Psychopathologies},
	elocation-id = {2021.04.01.21254790},
	year = {2021},
	doi = {10.1101/2021.04.01.21254790},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Network diffusion models are a common and powerful way to study the propagation of information through a complex system, they and offer straightforward approaches for studying multimodal brain network data. We developed an analytic framework to identify brain subnetworks with impaired information diffusion capacity using the structural basis that best maps to resting state functional connectivity and applied it towards a heterogenous internalizing psychopathology (IP) cohort. This research provides preliminary evidence of a transdiagnostic deficit characterized by information diffusion impairment of the right area 8BM, a key brain region involved in organizing a broad spectrum of cognitive tasks, that may underlie previously reported dysfunction of multiple brain circuits in the IPs. We also demonstrate that models of neuromodulation involving targeting this brain region normalize IP diffusion dynamics towards those of healthy controls. These analyses provide a framework for multimodal methods that identity diffusion disrupted subnetworks and potential targets for neuromodulatory intervention based on previously well-characterized methodology.Competing Interest StatementThe authors have declared no competing interest.Clinical TrialNCT01903447Funding StatementThis study was supported by funding from the National Institute of Mental Health of the National Institutes of Health (NIMH-NIH) grants R01MH101497 (to KLP) and the NIMH grant 5T32MH067631-14 (support for PJT).Author DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:This study was approved by the University of Illinois at Chicago Institutional Review Board and written informed consent was obtained for each participant.All necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).YesI have followed all appropriate research reporting guidelines and uploaded the relevant EQUATOR Network research reporting checklist(s) and other pertinent material as supplementary files, if applicable.YesData and software specific to this manuscript are available at the lead author{\textquoteright}s github page. All other resources are detailed in the methods section of the manuscript. https://github.com/pauljasonthomas/brainnetdiff},
	URL = {https://www.medrxiv.org/content/early/2021/04/07/2021.04.01.21254790},
	eprint = {https://www.medrxiv.org/content/early/2021/04/07/2021.04.01.21254790.full.pdf},
	journal = {medRxiv}
}

```
