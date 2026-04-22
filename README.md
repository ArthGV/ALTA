# Adaptive Latent Trajectory Anchoring (ALTA) for Action Segmentation Dataset Condensation

> Official implementation of the paper: [*Paper title to be provided*](#) <!-- link to be provided -->

---

## Overview

ALTA is a framework for action segmentation dataset condensation using adaptive latent trajectory anchoring. This repository contains the full implementation as described in the associated publication.

---

## Authors

**Main Contributors:** 
- Arthème Gauthier-Villars — ETH Zürich / National Univerity of Singapore
- Guodong Ding — National Univerity of Singapore

**Project Supervisors:** 
- Dr. Angela Yao — National Univerity of Singapore
- Prof. Dr. Siyu Tang — ETH Zürich

---

## Getting Started

Once the datasets have been uploaded, follow the steps below to run the full pipeline.

### 1. Prepare the Data

Convert your dataset into the required format for the ALTA framework:

```bash
python data_utils/create_datasets.py --dataset <dataset_name>
```

```bash
python data_utils/create_sequences.py --dataset <dataset_name>
```

### 2. Train the Diffusion Model

```bash
python src/<data_format>_compression/train_diffusion_model.py --dataset <dataset_name> --config_path <config_path>

```

### 3. Compress the Dataset

`original_dataset` → `anchors` + `anchors_indices` + `reconstructed_dataset`

```bash
python src/<data_format>_compression/anchor_compressing.py --dataset <dataset_name> --config_path <config_path> --model_path <model_path>

```

### 4. Reconstruct dataset

`anchors` + `anchors_indices` → `reconstructed_dataset`

```bash
python src/<data_format>_compression/dataset_reconstruction.py --dataset <dataset_name> --config_path <config_path> --model_path <model_path> --compressed_folder <folder_path> 

```
WITH:
```bash
<data_format> : 'feat' if precomputed 1D features, 'frame' if raw frames
```
---

## Citation

If you use this code in your research, please cite our paper: 

[*citation to be provided*]

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.