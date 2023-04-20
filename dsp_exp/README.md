# Traffic Analysis through DSP

## File Structure

The directories for experiments include `/nodef`, `/wtfpad`, `/wt`, which correspond to experiments upon no-defense traffic, WTF-PAD-defensed traffic, and Walkie-Talkie-defensed traffic.

For each experiment directory, we have 5 python files having different jobs to do:

```console
xxx_exp.py				Main experiment, it will output the result
xxx_raw_exp.py		Main experiment upon non-processed traffic
xxx_psd.py				Drawing PSD figure
xxx_rgb.py				Drawing RGB figure for all websites
xxx_rgb_single.py	Drawing RGB figure for single website
```

|Files            |Job                                        |
|-----------------|-------------------------------------------|
|xxx_exp.py       |Main experiment, it will output the result |
|xxx_raw_exp.py   |Main experiment upon non-processed traffic |
|xxx_psd.py       |Drawing PSD figure                         |
|xxx_rgb.py       |Drawing RGB figure for all websites        |
|xxx_rgb_single.py|Drawing RGB figure for single website      |
