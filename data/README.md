# Data Preparation

This directory contains the scripts and instructions to prepare the datasets used in the experiments.

Please set `data_root` to the directory where the datasets will be stored.

# Overview

```
data_root
|-- ID_ImageNet1K
|   |-- test/ e.g. n01440764/           # 50,000 JPEG images
|   `-- val/ e.g. n01440764/            #  1,000 JPEG images
|-- ID_VOC
|   |-- test/ e.g. aeroplane/           #    906 jpg images
|   `-- val/ e.g. aeroplane/            #     94 jpg images
|-- OOD_COCO
|   `-- test/images                     #  1,000 jpeg images
|-- OOD_ImageNet22K
|   `-- test/ e.g. n01937909/           # 18,335 JPEG images
|-- OOD_Places
|   `-- test/images                     #  10,000 jpg images
|-- OOD_Sun
|   `-- test/images                     # 10,000 jpg images
|-- OOD_Texture
|   `-- test/images/ e.g. banded/       #  5,640 jpg images
`-- OOD_iNaturalist
    `-- test/images                     #  10,000 jpg images
```

# In-Domain (ID) Datasets Preparation

## 1. ID_ImageNet1K

> We use the ImageNet-1000 (ILSVRC2012) dataset for ID validation and testing. The original dataset contains 1.2 million training images and 50,000 validation images from 1000 classes, and is widely used for image classification. We follow [MCM](https://github.com/deeplearning-wisc/MCM#in-distribution-datasets) to construct the ImageNet1K ID test set from the validation set. Additionally, we curated an ImageNet1K ID validation set from the training set by randomly selecting one image for each label.

### ID_ImageNet1K_val

We provide the curated ImageNet1K ID validation set [here](https://drive.google.com/drive/folders/1_qAOaYNCMfR2yY7pLeCyn1u8JLKawPhM?usp=share_link), please download and extract it to `data_root/ID_ImageNet1K/val/`.

We also provide the code to reproduce the ImageNet1K ID validation set if needed.

<details>
<summary>Construct ImageNet1K ID validation set</summary>

    export $data_root=/path/to/data_root
    cd $data_root/downloads

    # download train(task1&2) images, 138GB
    wget https://download_link_to_ILSVRC_2012/ILSVRC2012_img_train.tar
    tar -xvf ILSVRC2012_img_train.tar -P ImageNet_train

    # restore the ImageNet1K ID validation set from the training set
    cd $SVD_OOD
    python data/utils/restore_files.py \
    --json_file data/ID_ImageNet1K/imagenet1k_val_data.json \
    --source_folder $data_root/downloads/ImageNet_train \
    --target_folder $data_root/ID_ImageNet1K/val
</details>

### ID_ImageNet1K_test

1. Download the ImageNet-1K dataset ([source](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)).

    ```shell
    export data_root=/path/to/data_root
    export SVD_OOD=/path/to/SVD_OOD
    cd $data_root
    # download valid(all tasks) images, 6.3GB
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P downloads

    mkdir ./ID_ImageNet1K/test
    tar -xvf downloads/ILSVRC2012_img_val.tar -C $data_root/ID_ImageNet1K/test
    ```

2. Excute the following script ([source](https://github.com/soumith/imagenetloader.torch/blob/master/valprep.sh)) to restrucure the ImageNet1K ID test set from original val set.

    ```shell
    cd $SVD_OOD
    bash ./data/utils/restore_imagenet1k_test.sh $data_root/ID_ImageNet1K/test
    ```


## 2. ID_VOC

> The Pascal VOC (Visual Object Classes) dataset is a benchmark dataset widely used in computer vision, featuring annotated images across multiple object categories. We use the Pascal-VOC subset collected by [GL-MCM](https://github.com/AtsuMiyai/GL-MCM/tree/master#in-distribution-datasets) as ID validation and test set, each image has single-class ID objects and one or more OOD objects. The ID validation and test set are split by 1:9 for each class, resulting in 94 and 906 images, respectively.

1. Download the `datasets.tar.gz` to `data_root/downloads` from [Google Drive](https://drive.google.com/file/d/1he4jKi2BfyGT6rkcbFYlez7PbLMXTBMR/view?usp=sharing).  This file will be reused in [OOD datasets preparation](#4-coco_test-voc_test).

2. Unzip and extract `ID_VOC_val` and `ID_VOC_test` to `data_root/ID_VOC`.

    ```shell
    cd $data_root/downloads
    # download `datasets.tar.gz` from Google Drive
    # unzip file
    tar -xzvf datasets.tar.gz

    # clean hidden files, e.g. ._2008_003846.jpg
    find datasets -type f -name ".*" -delete

    cd $SVD_OOD

    # extract ID_VOC_val and ID_VOC_test
    mkdir -p $data_root/ID_VOC
    python data/utils/restore_files.py \
    --json_file data/ID_VOC/voc_val_data.json \
    --source_folder $data_root/downloads/datasets/ID_VOC_single \
    --target_folder $data_root/ID_VOC/val

    python data/utils/restore_files.py \
    --json_file data/ID_VOC/voc_test_data.json \
    --source_folder $data_root/downloads/datasets/ID_VOC_single \
    --target_folder $data_root/ID_VOC/test
    ```

# Out-of-Domain (OOD) Datasets Preparation

## 1. iNaturalist_test, Places_test, Sun_test, Texture_test

- Excute the following script to download and extract the Sun and Texture OOD datasets.

    ```shell
    cd $data_root/downloads

    # download and unzip iNaturalist
    wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
    tar -xvf iNaturalist.tar.gz
    mkdir -p $data_root/OOD_iNaturalist/test
    mv iNaturalist/images $data_root/OOD_iNaturalist/test

    # download and unzip Places
    wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
    tar -xvf Places.tar.gz
    mkdir -p $data_root/OOD_Places/test
    mv Places/images $data_root/OOD_Places/test


    # download and unzip Sun
    wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
    tar -xvf SUN.tar.gz
    mkdir -p $data_root/OOD_Sun/test
    mv SUN/images $data_root/OOD_Sun/test

    # download and unzip Texture
    wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
    tar -xvf dtd-r1.0.1.tar.gz
    mkdir -p $data_root/OOD_Texture/test
    mv dtd/images $data_root/OOD_Texture/test
    rm $data_root/OOD_Texture/test/images/waffled/.directory
    ```

## 2. COCO_test

> [MCM](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) curated a Pascal-VOC OOD test set (VOC for short) with 4,000 images that are not overlapped with the MS-COCO ID classes, which we use as OOD testing data for MS-COCO ID test set.

- Reuse the `datasets.tar.gz` downloaded [before](#2-id_coco_val-id_coco_test-id_voc_val-id_voc_test), and extract `COCO_test` and `VOC_test` to `data_root/OOD_COCO_VOC`.

    ```shell
    cd $data_root

    # extract COCO_test
    mkdir -p OOD_COCO/test
    mv $data_root/downloads/datasets/OOD_COCO/images OOD_COCO/test
    ```

## 3. ImageNet22K_test

> The ImageNet-22K dataset, formerly known as ImageNet-21K, addresses the underestimation of its additional value compared to the standard ImageNet-1K pretraining, aiming to provide high-quality pretraining for a broader range of models. We use the filtered subset collected by [multi-label-ood](https://github.com/deeplearning-wisc/multi-label-ood#out-of-distribution-dataset) as the OOD test set for MC-COCO and Pascal-VOC ID test sets.

1. Download the `ImagenetOOD_for_COCO_VOC.tar` to `data_root/downloads` from [Google Drive](https://drive.google.com/drive/folders/1BGMRQz3eB_npaGD46HC6K_uzt105HPRy).

2. Extract `ImageNet22K_test`:

    ```shell
    cd $data_root/downloads
    # download `ImagenetOOD_for_COCO_VOC.tar` from Google Drive
    tar -xvf ImagenetOOD_for_COCO_VOC.tar

    cd ../
    # extract ImageNet22K_test
    mkdir -p $data_root/OOD_ImageNet22K
    mv $data_root/downloads/ImageNet-22K OOD_ImageNet22K
    mv OOD_ImageNet22K/ImageNet-22K OOD_ImageNet22K/test
    ```

# Check the data structure

- Excute the following script to check the data structure.

    ```shell
    cd $SVD_OOD
    python data/utils/check_data_structure.py --data_root $data_root
    ```

    You should see the following output:

    ```shell
    Comparing folder structure...

    Comparing ID_ImageNet1K...
    Split test: matched! Number of images:  50000
    Split val: matched! Number of images:  1000

    Comparing ID_VOC...
    Split test: matched! Number of images:  906
    Split val: matched! Number of images:  94

    Comparing OOD_iNaturalist...
    Split test: matched! Number of images:  10000

    Comparing OOD_Sun...
    Split test: matched! Number of images:  10000

    Comparing OOD_Places...
    Split test: matched! Number of images:  10000

    Comparing OOD_Texture...
    Split test: matched! Number of images:  5640

    Comparing OOD_ImageNet22K...
    Split test: matched! Number of images:  18335

    Comparing OOD_COCO...
    Split test: matched! Number of images:  1000
    ```