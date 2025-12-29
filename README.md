# lstnet
LSTNet model for domain adaptation in computer vision.


Run training
?| 
Run translation
`python run.py translate mnist --load_model --output_folder output_base`
`python run.py translate usps --load_model --output_folder output_base`


Run evaluation
`python run.py eval mnist eval_models/USPS/USPS_model.pth --output_folder output_base --output_results_file mnist2usps_res --dataset_path output_base/MNIST_translated_data.pt`
`python run.py eval usps eval_models/MNIST/MNIST_model.pth --output_folder output_base --output_results_file usps2mnist_res --dataset_path output_base/USPS_translated_data.pt`

`python run.py eval mnist $svhn_clf --output_folder $output_folder --output_results_file mnist2svhn_res --dataset_path $mnist_trans`


`pyhon run.py all mnis usps mnis_usps_params.json eval_models/USPS/USPS_model.pth eval_models/MNIST/MNIST_model.pth --ouput_folder output_base`


Run Optuna
`python run.py train mnist usps params/mnist_usps_params.json --output_folder output_optuna_tuning --num_workers 48 --optuna --optuna_study_name tuning --optuna_trials 200 --optuna_sampler_start_trials 30 --optuna_pruner_sample_trials 20 --optuna_pruner_warmup_steps 30 --optuna_pruner_interval_steps 10 --percentile 90 --hyperparam_mode augm_ops train_params architecture --compile`

## Training Classifiers

### Evaluation
`python run.py eval mnist mnist_clf_base/MNIST_clf_model.pth --output_folder mnist_clf_base --output_results_file  mnist_clf_eval`