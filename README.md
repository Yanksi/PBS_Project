# Physically-Based Simulation Project: Position Based Fluid
Group 18 Xudong Jiang , Shuhao Li, Boyan Duan
---

## Environment Setup

GPU is required.
```bash
conda env create -f environment.yml
conda activate pbs
```

## Run


```bash
python main.py -cfg CONFIG_FILE
```

Example configurations:
| File     | Content |
| ----------- | ----------- |
| pbd_config_sample | Documentation of config files |
| pbd_config_fluid.yml | Position Based fluid |
| pbd_config_2s | Solid-Solid interaction |
| pbd_config_2p | Liquid-Liquid interaction |
| pbd_config_fs | Solid-Liquid interaction |

## Control

| Key      | Action |
| ----------- | ----------- |
| Space | Start/Pause Simulation|
| Q/W/E/A/S/D      | Navigation       |
| &uarr;/&darr;/&larr;/&rarr;   | Gravity direction        |
