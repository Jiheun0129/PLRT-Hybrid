# PLRT-Hybrid
Code and data for the JoH manuscript: "Synthesis of Reservoir Operation across the CONUS using PLRT-Hybrid Modeling".
=======

PLRT-Hybrid is an interpretable machine-learning framework for representing reservoir operations. The framework extends the Piecewise Linear Regression Tree (PLRT) model ([Ford and Sankarasubramanian, 2023](https://github.com/lcford2/py-plrt)) by combining interpretable operational modes with mode-specific multilayer perceptron models.

This repository provides the source code required to run the original PLRT and PLRT-Hybrid models, pretrained neural-network components, and an example reservoir dataset demonstrating the full prediction workflow.

## Repository Structure

```text
PLRT-Hybrid/
├── README.md
├── LICENSE
├── environment.yml
├── .gitignore
│
├── data/
│   └── example/
│       ├── 41_key.csv
│       ├── 41_meta.csv
│       ├── 41_raw.csv
│       └── 41_std.csv
│
├── models/
│   ├── M345.pkl
│   ├── M78.pkl
│   ├── M9.pkl
│   ├── M10.pkl
│   ├── M1112.pkl
│   ├── M13.pkl
│   └── M16M17.pkl
│
├── scripts/
│   └── run_example.py
│
├── src/
│   └── plrt_hybrid/
│       ├── __init__.py
│       ├── plrt.py
│       ├── hybrid.py
│       ├── preprocessing.py
│       ├── simulation.py
│       └── metrics.py
│
└── outputs/
    ├── predictions/
    └── summary/
```

## Model Overview

The original PLRT framework represents reservoir operating behavior using an interpretable regression-tree structure. Each terminal node corresponds to an operational mode and contains a linear release equation.

PLRT-Hybrid retains the operational-mode structure of PLRT while replacing selected linear equations with pretrained multilayer perceptron models. The hybrid structure is intended to improve prediction for operational modes exhibiting complex or nonlinear behavior while preserving the interpretability of the original PLRT framework.

The model uses reservoir state, inflow, release history, rolling hydrologic variables, reservoir attributes, and calendar variables to estimate daily reservoir release.

## Installation

The recommended environment can be created using Conda.

```bash
conda env create -f environment.yml
conda activate plrt-hybrid
```

The main Python dependencies include:

* Python
* NumPy
* pandas
* scikit-learn
* joblib

The exact package versions used for the released pretrained models are specified in `environment.yml`.

## Example Dataset

The example dataset is located in:

```text
data/example/
```

The example uses reservoir ID `41` and includes four files.

### `41_raw.csv`

Contains reservoir variables in their original units, including observed release, inflow, storage, and reservoir attributes.

### `41_std.csv`

Contains standardized input variables used by PLRT and PLRT-Hybrid.

### `41_key.csv`

Contains monthly means and standard deviations used to transform variables between raw and standardized units.

### `41_meta.csv`

Contains metadata for the example reservoir.

The example dataset is provided to demonstrate the model execution workflow. It is not intended to reproduce every aggregate result reported in the manuscript, which was based on a substantially larger reservoir dataset.

## Running the Example

Run the example from the repository root directory.

```bash
python scripts/run_example.py
```

By default, the script runs both PLRT and PLRT-Hybrid using initialization frequencies of:

```text
1, 2, 7, 30, 90, and 180 days
```

The script:

1. Loads the example raw, standardized, scaling, and metadata files.
2. Validates the required variables and dates.
3. Loads the pretrained PLRT-Hybrid models.
4. Runs PLRT and PLRT-Hybrid.
5. Calculates the coefficient of determination, R-squared.
6. Saves daily predictions and a performance summary.

## Optional Commands

Run only PLRT-Hybrid:

```bash
python scripts/run_example.py --methods plrt_hybrid
```

Run only the original PLRT:

```bash
python scripts/run_example.py --methods plrt
```

Run selected initialization frequencies:

```bash
python scripts/run_example.py --frequencies 1 7 30
```

Run PLRT-Hybrid at selected frequencies:

```bash
python scripts/run_example.py \
    --methods plrt_hybrid \
    --frequencies 1 2 7 30
```

Specify a reservoir ID:

```bash
python scripts/run_example.py --res-id 41
```

Do not save the full daily prediction files:

```bash
python scripts/run_example.py --skip-save-predictions
```

## Outputs

Prediction files are written to:

```text
outputs/predictions/
```

Example output filenames include:

```text
reservoir_41_plrt_freq_7.csv
reservoir_41_plrt_hybrid_freq_7.csv
```

Each prediction file includes variables such as:

* date
* predicted release in standardized units
* predicted release in original units
* predicted storage in standardized units
* predicted storage in original units
* identified PLRT operational mode
* observed release
* model name
* initialization frequency

The performance summary is written to:

```text
outputs/summary/reservoir_41_performance_summary.csv
```

The summary file includes:

* reservoir ID
* model
* initialization frequency
* R-squared
* number of observations
* operational modes used
* execution status
* output file location

## Initialization Frequency

Reservoir release and storage predictions are generated recursively between initialization dates.

At each initialization date, observed previous storage and release values are used to initialize the simulation. Between initialization dates, simulated storage and release values are used as model inputs.

For example:

* `frequency = 1`: observed state variables are used every day.
* `frequency = 7`: the simulation is reinitialized weekly.
* `frequency = 30`: the simulation is reinitialized approximately monthly.
* `frequency = 180`: the simulation is reinitialized approximately semiannually.

Longer initialization intervals allow prediction errors to propagate through simulated storage and release states.

## Pretrained Models

The pretrained MLP models are stored in:

```text
models/
```

The released model files correspond to selected PLRT operational modes or mode groups:

```text
M345.pkl
M78.pkl
M9.pkl
M10.pkl
M1112.pkl
M13.pkl
M16M17.pkl
```

The model files are used only for inference in the current repository.

Because serialized Python models can depend on package versions, users should create the environment using the provided `environment.yml` before loading the model files.

Users should only load model files obtained from this official repository. Python pickle and joblib files should not be loaded from untrusted sources.

## Applying the Model to Another Reservoir

To apply PLRT or PLRT-Hybrid to another reservoir, users should prepare files following the structure and variable names of the example dataset.

At minimum, the workflow requires:

* daily raw reservoir data,
* standardized model inputs,
* monthly standardization parameters,
* reservoir metadata,
* a unique reservoir ID.

The raw and standardized datasets must:

* contain matching dates,
* contain the same number of observations,
* be sorted chronologically,
* include the reservoir ID,
* contain all required PLRT and PLRT-Hybrid variables.

The monthly scaling file must include the means and standard deviations required to convert model inputs and predictions between standardized and original units.

## Data Availability

The complete reservoir dataset used to develop and evaluate PLRT-Hybrid was assembled from the ResOpsUS database (Steyaert et al., 2022) and GDROM dataset (Li et al., 2025).


## Authors

* Jiheun Kim
* Shiqi Fang
* A. Sankarasubramanian

Department of Civil, Construction, and Environmental Engineering
North Carolina State University

## License

The source code in this repository is distributed under the license provided in the `LICENSE` file.

The license applies to the source code unless otherwise stated. Data files and pretrained model files may be subject to separate terms described in their respective directories.

## Reference

Ford, L., & Sankarasubramanian, A. (2023). Generalizing reservoir operations using a piecewise classification and regression approach. Water Resources Research, 59(9), e2023WR034890.

Li, D., Chen, Y., Cai, X., Zhao, Q. (2025). Data-driven Reservoir Operation Rules for 450+ Reservoirs in Contiguous United States, HydroShare, https://doi.org/10.4211/hs.63add4d5826a4b21a6546c571bdece10

Steyaert, J. C., Condon, L. E., WD Turner, S., & Voisin, N. (2022). ResOpsUS, a dataset of historical reservoir operations in the contiguous United States. Scientific Data, 9(1), 34.

