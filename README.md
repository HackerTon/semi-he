# Semi-supervised Cross Domain H&E Stain Images Segmentation

## How to train
Run `python main.py -p {dataset} -b {batch size} -x {experiment number} -m {compute mode} -l {learning rate}`

- {dataset}: path to your dataset 
- {batch size}: batch size 
- {experiment number}: experiment number (Refer to [trainer.py](src/service/trainer.py))  

Example command 
`python main.py -p data/textocr -b 32 -x 0 -m mps -l 0.001 -e 50 -n experimentname`

# semi supervised
## UKMTIGER
* Train baseline on tiger
* Train proposed on tiger


## UKMTILS
1. Generate pseudo-label for baseline
```
python generate_pseudolabel.py \
--model data/model/20241211_013306_tiger_baseline_fix_lambda_0.02max_500_bs16/499_tiger_baseline_fix_lambda_0.02max_500_bs16_model.pt \
--dataset_dir /mnt/storage/Dataset130_ukmtils \
--pseudo_dir data/ukmtils_pseudo_b \
--mode baseline
```

2. Generate pseudo-label for proposed model
```
python generate_pseudolabel.py \
--model data/model/20241221_163831_tiger_proposed_lambda_0.02max_500_bs16_pseudo/499_tiger_proposed_lambda_0.02max_500_bs16_pseudo_model.pt \
--dataset_dir /mnt/storage/Dataset130_ukmtils \
--pseudo_dir data/ukmtils_pseudo \
--mode dirichlet
```

3. Train baseline on pseudo-label
```
python main.py --path data/ukmtils_pseudo_baseline \
--batchsize 16 \
--mode cuda \
--learning_rate 0.001 \
--name 'ukmtils baseline epoch500 bs16 pseudo' \
--epoch 500 \ 
--pretrain_path data/model/20241211_013306_tiger_baseline_fix_lambda_0.02max_500_bs16/499_tiger_baseline_fix_lambda_0.02max_500_bs16_model.pt
```

4. Train proposed on pseudo-label
```
python main.py --path data/ukmtils_pseudo \
--batchsize 16 \
--mode cuda \
--learning_rate 0.001 \
--name 'ukmtils proposed epoch500 bs16 pseudo' \
--epoch 500 \ 
--pretrain_path data/model/20241221_163831_tiger_proposed_lambda_0.02max_500_bs16_pseudo/499_tiger_proposed_lambda_0.02max_500_bs16_pseudo_model.pt
```

5. Evaluate on dirichlet UKMTILS
```
python evaluation.py --model data/model/20241230_151702_proposed/499_proposed_model.pt \
--dataset_dir data/ukmtils_pseudo \         
--mode dirichlet
```

6. Evaluate on baseline UKMTILS
```
python evaluation.py --model data/model/20241221_185129_tiger_baseline_500_bs16_pseudo/499_tiger_baseline_500_bs16_pseudo_model.pt \
--dataset_dir data/ukmtils_pseudo_baseline \
--mode baseline 
```

## Ocelot
1. Generate Ocelot pseudo dataset for baseline
```
python generate_pseudolabel.py \
--model data/model/20241211_013306_tiger_baseline_fix_lambda_0.02max_500_bs16/499_tiger_baseline_fix_lambda_0.02max_500_bs16_model.pt \
--dataset_dir /mnt/storage/ocelot2023_v1.0.1 \
--pseudo_dir data/ocelot_pseudo_baseline \
--mode baseline
```

2. Generate Ocelot pseudo dataset for proposed
```
python generate_pseudolabel.py \
--model data/model/20241221_163831_tiger_proposed_lambda_0.02max_500_bs16_pseudo/499_tiger_proposed_lambda_0.02max_500_bs16_pseudo_model.pt \
--dataset_dir /mnt/storage/ocelot2023_v1.0.1 n\
--pseudo_dir data/ocelot_pseudo_dirichlet \
--mode dirichlet
```

3. Train Ocelot pseudo dataset on baseline
```
python main.py --path data/ocelot_pseudo_baseline \
--batchsize 16 \
--mode cuda \
--learning_rate 0.0001 \
--name 'ocelot baseline epoch500 bs16 pseudo' \
--epoch 500 \
--pretrain_path data/model/20241221_185129_tiger_baseline_500_bs16_pseudo/499_tiger_baseline_500_bs16_pseudo_model.pt
```

4. Train Ocelot pseudo dataset on proposed
```
python main.py --path data/ocelot_pseudo_dirichlet \
--batchsize 16 \
--mode cuda \
--learning_rate 0.0001 \
--name 'ocelot proposed epoch500 bs16 pseudo' \
--epoch 500 \
--pretrain_path data/model/20241221_163831_tiger_proposed_lambda_0.02max_500_bs16_pseudo/499_tiger_proposed_lambda_0.02max_500_bs16_pseudo_model.pt
```

5. Evaluate proposed on Ocelot dataset
```
python evaluation.py \
--model 'data/model/20250110_040911_ocelot_proposed_epoch500_bs16_pseudo/499_ocelot_proposed_epoch500_bs16_pseudo_model.pt' \
--dataset_dir /mnt/storage/ocelot2023_v1.0.1 \         
--mode dirichlet
```

6. Evaluate baseline on Ocelot dataset
```
python evaluation.py \
--model data/model/20250110_012619_ocelot_baseline_epoch500_bs16_pseudo/499_ocelot_baseline_epoch500_bs16_pseudo_model.pt \
--dataset_dir /mnt/storage/ocelot2023_v1.0.1 \         
--mode baseline
```