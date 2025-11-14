# lstnet
LSTNet model for domain adaptation in computer vision.


Run training
`python run.py train mnist usps params/mnist_usps_params.json --output_folder output_base_unsup --num_workers 12 --full_training`

Run translation
`python run.py translate mnist --load_model --output_folder output_base`
`python run.py translate usps --load_model --output_folder output_base`


Run evaluation
`python run.py eval mnist eval_models/USPS/USPS_model.pth --output_folder output_base --output_results_file mnist2usps_res --dataset_path output_base/MNIST_translated_data.pt`
`python run.py eval usps eval_models/MNIST/MNIST_model.pth --output_folder output_base --output_results_file usps2mnist_res --dataset_path output_base/USPS_translated_data.pt`


`pyhon run.py all mnis usps mnis_usps_params.json eval_models/USPS/USPS_model.pth eval_models/MNIST/MNIST_model.pth --ouput_folder output_base`


Run Optuna
`python run.py train mnist usps params/mnist_usps_params.json --output_folder output_optuna_uda --num_workers 12 --optuna --optuna_study_name name --optuna_min_resource 5 --optuna_max_resource 50 --optuna_reduction_factor 3 --optuna_sampler_start_trials 10 --optuna_trials 50`

## Training Classifiers

### Evaluation
`python run.py eval mnist mnist_clf_base/MNIST_clf_model.pth --output_folder mnist_clf_base --output_results_file  mnist_clf_eval`