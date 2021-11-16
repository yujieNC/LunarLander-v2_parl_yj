# Efficient SuperPoint

## Installation
### Requirements
- python == 3.6
- pytorch >= 1.1 (tested in 1.3.1)
- torchvision >= 0.3.0 (tested in 0.4.2)
- cuda (tested in cuda10)

```
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
```

### Path setting
- paths for COCO datasets ($DATA_DIR), logs are set in `setting.py`

### Dataset
Datasets should be downloaded into $DATA_DIR. The Synthetic Shapes dataset will also be generated there. The folder structure should look like:

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created

- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
```


## run the code

### 1) Training MagicPoint on Synthetic Shapes
```
python train4.py train_base configs/magicpoint_shapes_pair.yaml magicpoint_synth --eval
```
You will generate it when first running it.
Synthetic data is exported in `./datasets`. You can change the setting in `settings.py`.

### 2) Exporting detections on MS-COCO
This is the step of homography adaptation(HA) to export pseudo ground truth for joint training.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_DIR' (defined in setting.py)
<!-- - you can export hpatches or coco dataset by editing the 'task' in config file -->
- config file:
```
export_folder: <'train' | 'val'>  # set export for training or validation
```
#### export coco - do on training set 
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```
#### export coco - do on validation set 
- Edit 'export_folder' to 'val' in 'magicpoint_coco_export.yaml'
```
python export.py export_detector_homoAdapt configs/magicpoint_coco_export.yaml magicpoint_synth_homoAdapt_coco
```

### 3) Training Superpoint on MS-COCO
You need pseudo ground truth labels to traing detectors. Labels are exported from step 2). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### COCO
```
python train4.py train_joint configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```


- set your batch size (originally 1)
- refer to: 'train_tutorial.md'

### 4) Export/ Evaluate the metrics on HPatches
- Use pretrained model or specify your model in config file
- ```./run_export.sh``` will run export then evaluation.

#### Export
- download HPatches dataset (link above). Put in the $DATA_DIR.
```python export.py <export task> <config file> <export folder>```
- Export keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### evaluate
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```
