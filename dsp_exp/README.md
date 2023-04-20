# Traffic Analysis through DSP

This repository is mainly for traffic analysis through digital signals processing methods.

## Requirements

1. On your PC or server these modules should be installed:

```console
scipy
tqdm
sklearn
```

If you are one of the users of our lab server, you can just run:

```shell
conda activate zysrt39
```

All the required modules have been installed in this conda environment.

2. Before you start any experiments, make sure that your experiment script is under the corresponding experiment folder if you try different datasets, and the code should be inserted to find the modules:

```python
import sys
sys.path.append("../") 
```

## File structure

The files are listed below:

```console
.
├── DF
├── Front
├── README.md
├── WTF_PAD
├── result
├── run.sh
└── tools
```

- All the scripts for experiments are saved in `/DF`, `/Front`, `WTF_PAD`, the folder names correspond to the name of the datasets.
- The results will be saved in `/result`. If you need to save the result, you can just run:

```shell
nohup python -u PyFileName.py > ResultName.log 2>&1 &
```

You must replace `PyFileName` and `ResultName` as the real file name.

- If you want to complete the experiments in parallel, you can just run `run.sh`:

```
cd ${REPO_PATH}/dsp_exp
./run.sh
```

All the results will be written into `/result`. The results are named as the name of corresponding experimental script.

- If you want to do further experiments with different datasets and DSP tools, you can go to `/tools` whose APIs will be explained in detail in `Tools` part.

## Tools

`Tools` is the core module of the experiment. It contains all the utilities necessary, including `data_loading` module, `dsp` module, `classifiers` module and `plotting` module. The modules and their major jobs are listed below:

| Module       | Job                                                          |
| ------------ | ------------------------------------------------------------ |
| data_loading | Loading and processing datasets.<br>You can tune the parameters to get the type of data you want. |
| dsp          | Processing the signals (traffic).<br>You can tune the parameters to process the signals in different ways. |
| classifiers  | Completing classification tasks.<br>The classfiers include k-NN, random forest, multilayer perception and linear regression. |
| plotting     | Visualizing the data.<br>You can choose scatter diagram and rgb diagram. (spectrum diagram part is incomplete) |

the APIs of these modules are explained in detail in the following. Before using them please make sure 

### Data Loading Module

To load the dataset, add the code 

```python
from tools.data_loading import dataset
```

The original data-loading API is shown as below:

```python
def dataset(prop_test=0.1, 
            prop_valid=0.2, 
            db_name="DF", 
            type="td", 
            spec="ps-corr", 
            filter="none") -> NDArray
```

The parameters are listed in the table below:

| Parameter  | Type                                                         | Usage                                                        |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| prop_test  | float, (0,1)                                                 | The proportion of testing data in the whole dataset. <br>Set as 0.1 by default. |
| prop_valid | float, (0,1)                                                 | The proportion of validating data in the whole dataset. <br>Set as 0.2 by default. |
| db_name    | str, <br>"DF"<br>"Front"<br>"WTF_PAD"                        | The name of the dataset, the corresponding .npz file will be downloaded.<br>Set as "DF" by default. |
| type       | str,<br>"td"<br>"d"                                          | The type of the dataset.<br> "td" is for data taking both timestamps and directions into account, <br> "d" is for data only considering directions.<br>Set as "td" by default. |
| spec       | str,<br>"ps-corr"<br>"freq"<br>"none"                        | The spectrum type of the output dataset. <br>"ps-corr" is for power spectrum with correlation,<br> "freq" is for frequency spectrum, <br> "none" is for not using any spectrums.<br>Set as "ps-corr" by default. |
| filter     | str,<br>"none"<br>butter-low"<br>"butter-high"<br>"gaussian"<br>"direct"<br>"window"<br>"winb-low"<br>"kalman" | The wave filter when processing the dataset. All the filters are saved in `dsp` module<br>"butter-low" is for Butterworth lowpass filter,<br>"butter-high" is for Butterworth highpass filter,<br>"gaussian" is for Gaussian lowpass filter,<br>"direct" is for cutting the spectrum according to the cutting off frequency directly and generating the remaining part of the spectrum,<br>"window" is for applying window function to the signal,<br>"winb-low" is for combining window function and Butterworth lowpass filter together,<br>"kalman" is for Kalman filter,<br>"none" is for not using any filter.<br>Set as "none" as default. |

When you use the API, the dataset will be split automatically into training set, validating set and testing set. For example, to load the frequency spectrum data from WTF_PAD dataset for training which timestamps should be taken into account and the original sequences are filtered by gaussian lowpass filter, and the dataset is splited in to (0.1, 0.2, 0.7) for testing, validation and training, you should run:

```python
X_train, y_train, X_test, y_test, X_valid, y_valid \
= dataset(prop_test=0.1, prop_valid=0.2, 
          db_name="WTF_PAD", type="td", 
          spec="freq", filter="gaussian")
```

