1. Loco dataset

Training:

python main_loco.py --category breakfast_box

Validation:

1) Generate anomaly maps:

python get_anomaly_maps.py --category breakfast_box

2) Run the validation code:

cd mvtec_loco_ad_evaluation

evaluate_experiment.py

--object_name breakfast_box

--dataset_base_dir '../../autodl-tmp/mvtec-loco/'

--anomaly_maps_dir '../../autodl-tmp/mvtec-loco_anomaly_maps/'

--output_dir 'metrics/'

Note: Replace dataset_base_dir, anomaly_maps_dir, and output_dir with your own paths.

2. AD dataset

python main_ad.py --category bottle

Verification results can be viewed directly in the results/ directory.
