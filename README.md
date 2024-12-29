# Semi-supervised Cross Domain H&E Stain Images Segmentation

## How to train
Run `python main.py -p {dataset} -b {batch size} -x {experiment number} -m {compute mode} -l {learning rate}`

- {dataset}: path to your dataset 
- {batch size}: batch size 
- {experiment number}: experiment number (Refer to [trainer.py](src/service/trainer.py))  

Example command 
`python main.py -p data/textocr -b 32 -x 0 -m mps -l 0.001 -e 50 -n experimentname`

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