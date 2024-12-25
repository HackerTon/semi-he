# Semantatic segmentation trainer
![screenshot](src/notebook/test.png)

Heart and lung segmentation with bounding box visualisation

## How to prepare dataset
1. Download images from [NIH Chest X-rays Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data).
    - Unzip everything in `archive.zip` to `chestxray` directory.
2. Download labels from [CheXmask Dataset](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/).
    - Download `ChestX-Ray8.csv` only.
3. Create `data/cardiac` directory.
4. Move `ChestX-Ray8.csv` file to `data/cardiac/`.
5. Move `chestxray/` directory to `data/cardiac/`.
6. Process images and label by running `python -m src.cardiac_process_data -p data/cardiac -o data/cardiac`.


## How to train
Run `python main.py -p {dataset} -b {batch size} -x {experiment number} -m {compute mode} -l {learning rate}`

- {dataset}: path to your dataset 
- {batch size}: batch size 
- {experiment number}: experiment number (Refer to [trainer.py](src/service/trainer.py))  

Example command 
`python main.py -p data/textocr -b 32 -x 0 -m mps -l 0.001 -e 50 -n experimentname`

## How to monitor training progress
Run `tensorboard --logdir data/log`

## How to run benchmark
Run `python -m src.benchmark`


## semi supervised
* Generate pseudo-label for baseline
```
python generate_pseudolabel.py --model data/model/20241211_013306_tiger_baseline_fix_lambda_0.02max_500_bs16/499_tiger_baseline_fix_lambda_0.02max_500_bs16_model.pt \
--dataset_dir /mnt/storage/Dataset130_ukmtils \
--pseudo_dir data/ukmtils_pseudo_b \
--mode baseline
```
* Generate pseudo-label for proposed model
```
python generate_pseudolabel.py --model data/model/20241211_004039_tiger_proposed_fix_lambda_0.02max_500_bs16/361_tiger_proposed_fix_lambda_0.02max_500_bs16_model.pt \
--dataset_dir /mnt/storage/Dataset130_ukmtils \
--pseudo_dir data/ukmtils_pseudo \
--mode dirichlet
```

* Train baseline on pseudo-label
```
python main.py -p data/ukmtils_pseudo_b  -b 16 -m cuda --learning_rate 0.0001 --name 'tiger baseline 500 bs16 pseudo' -e 500
```