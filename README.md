# Traffic Analysis through DSP

This repo is mainly for traffic analysis thorugh digital signals processing methods.

## File Structures

The files are shown as below:

```console
.
├── Multitab-Tor-WFP
├── README.md
├── datasets
├── defense_datasets
└── dsp_exp
```

- The datasets should be downloaded into `/defense_datasets`.
- The experiments should be finished in `/dsp_exp`, detailed descriptions are explained in `README.md` under `/dsp_exp`.

## Datasets

Our datasets include `DF`, `WTF_PAD` and `Front`. 

When you are running experiments in `/dsp_exp` as the README under the folder tells you, you don't need to worry about the path if you download them into `/defense_datasets`, our data-loading API will find the path automatically.