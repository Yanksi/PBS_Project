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
For example:
```bash
python main.py -cfg pbd_config_2f2s.yml
```

Example configurations:
| File     | Content |
| ----------- | ----------- |
| pbd_config_sample.yml | Documentation of config files |
| pbd_config_fluid.yml | Position Based fluid |
| pbd_config_2s.yml | Solid-Solid interaction |
| pbd_config_2p.yml | Liquid-Liquid interaction |
| pbd_config_fs.yml | Solid-Liquid interaction |
| pbd_config_2f2s.yml | Two Solids in Two Liquids|

## Control

| Key      | Action |
| ----------- | ----------- |
| Space | Start/Pause Simulation|
| Q/W/E/A/S/D      | Navigation       |
| &uarr;/&darr;/&larr;/&rarr;   | Gravity direction        |

## [Video Demo](https://youtu.be/Z5_aBBxgeF4)
