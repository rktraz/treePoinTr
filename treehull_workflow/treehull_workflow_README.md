## Data preparation and model training workflow

**Data Preparation**

File with all the reusable functions:
functions.py 

Making pairs (full+partial) of shape-pointclouds from tree point clouds
	
	From full tree point clouds (for training):
	make_completeandpartial_topcompletion.py

	Separate sensor fusion data (for testing):
	predefhull_from_fusion_data_v2.py 
	
	From full tree point clouds, with predefined cutoff (for testing):
	make_completeandpartial_topcompletion_testsets.py


Define a train/test split:
make_train_test.py

**train model**
First, create a .yaml file that defines the input data and parameters for model training.
Then train a new model, either from scratch or finetune existing weights, for example:

```
bash ./scripts/train.sh 1  --config ./cfgs/predefhull_models/AdaPoinTr12.yaml   --exp_name finetune_predefhull_AdaPointr12 --start_ckpts ./ckpts/AdaPoinTr_s55.pth
```

Look at training log:
read_training_log.py


## Inference workflow

Make shape-pointclouds from just incomplete tree point clouds
shape-pointclouds_for_inference.py

Do inference on new data. 
e.g.:

```
python tools/inference.py cfgs/predefhull_models/AdaPoinTr12.yaml trained_models/AdaPoinTr12/predefhull_models/finetune_predefhull_AdaPointr12_CDL2_main12/ckpt-best.pth --pc_root data/swbm/completeandpartial/  --save_xyz --out_pc_root swbm_inference_AdaPoinTr12/
```

