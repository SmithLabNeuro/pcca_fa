# pCCA-FA (probabilistic canonical correlation analysis - factor analysis)

pCCA-FA is a dimensionality reduction framework that combines probabilistic CCA and factor analysis to model across- and within- dataset interactions. This project contains a Python implementation of pCCA-FA. pCCA-FA is proposed for and validated on population activity from two brain areas, and is described in the following reference. 

- McDonnell, M.E.&ast;, Umakantha, A.&ast;, Williamson, R.C.&ast;, Smith, M.A.&dagger;, & Yu, B. M.&dagger; Interactions across hemispheres in prefrontal cortex reflect global cognitive processing. bioRxiv (2025). <https://www.biorxiv.org/content/10.1101/2025.06.12.659406v1> (&ast; and &dagger; indicate equal contribution)

pCCA-FA terminology, use cases, and fitting procedures are described in detail in the Methods section of this reference. Please read it carefully before using the model.

## Installation

### Quick install

Download and extract the file `pcca_fa.zip` to the desired working directory.

### GitHub install

Download the latest release of pCCA-FA to the desired working directory. This repository uses submodules, please use the following line to download.
```
git clone --recursive https://github.com/SmithLabNeuro/pcca_fa
```
If the repository was cloned without the `--recursive` flag, when in the root directory, run the following line to initialize submodules.
```
git submodule update --init --recursive
```
Alternatively, files for the dependencies can be manually downloaded from GitHub: [factor analysis](https://github.com/meganmcd13/fa) and [canonical correlation analysis](https://github.com/meganmcd13/cca). This step will also be needed if the package was downloaded via .zip file. Files should be placed inside the `fa` and `cca` folders, respectively.

## Setup

A file containing tested Python package versions is provided for convenience (`environment.yml`). pCCA-FA code should be run inside this environment, although the tutorial should work in general Python 3.8 environments. To create the environment using conda, run the following command:

```
conda env create -f environment.yml
conda activate pccafa
```

## Usage

To get started, the `tutorial.ipynb` file walks through basic steps on fitting a pCCA-FA model to two populations of simulated neural activity. It also includes uses for various model metrics.

## Contact

For questions, please contact Megan McDonnell at <mmcdonnell@cmu.edu>.
