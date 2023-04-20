# Traffic Analysis through DSP

This repository is mainly for traffic analysis through digital signals processing methods.

## Requirements



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

- If you want to do further experiments with different datasets and DSP tools, you can go to `/tools` whose APIs will be explained in detail in the following part.

## Tools

`Tools` is the core module of the experiment. It contains all the utilities necessary, including `data_loading` module, `dsp` module, `classifiers` module and `plotting` module. The modules and their major jobs are listed below:

| Module       | Job                                                          |
| ------------ | ------------------------------------------------------------ |
| data_loading | Loading and processing datasets.<br>You can tune the parameters to get the type of data you want. |
| dsp          | Processing the signals (traffic).<br>You can tune the parameters to process the signals in different ways. |
| classifiers  | Completing classification tasks.<br>The classfiers include k-NN, random forest, multilayer perception and linear regression. |
| plotting     | Visualizing the data.<br>You can choose scatter diagram and rgb diagram. (spectrum diagram part is incomplete) |
