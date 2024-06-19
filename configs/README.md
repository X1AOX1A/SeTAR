# Experiment Configs

## Introduction

> Please change the `data_root` in the config file to your datasets path.

All configs are arranged with the following structure:

```
configs/experiments/model/ID_dataset/run_name
```

Here are the details of the experiments:

- Table1&Tabel8: Training-free results
    - [configs/training-free](./training_free/)
- Table2: Fine-tuned results
    - [configs/finetune](./finetune/)
- Table3: Ablation on modality
    - [configs/abs_modality](./abs_modality/)
- Figure3 & Figure4: Ablation of weight types
    - [configs/abs_weight_type](./abs_weight_type/)
- Table4: Ablation on w/w.o. projection
    - [configs/abs_proj](./abs_proj/)
- Table5: Ablation on search algorithm
    - [configs/abs_search_algorithm](./abs_search_algorithm/)
- Table6: Ablation on pruning strategies
    - [configs/abs_prune_type](./abs_prune_type/)

## Usages

For fine-tune experiments ([configs/finetune](./finetune/)), use `finetune.py` script to run the experiments.

```bash
python finetune.py --config_file path/to/finetune/config.json
```

For other experiments, use `search.py` script to run the experiments.

```bash
python search.py --config_file path/to/config.json
```