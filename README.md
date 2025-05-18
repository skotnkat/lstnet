# lstnet
LSTNet model for domain adaptation in computer vision.


Run training
`python run.py train mnist usps mnist_usps_params.json --output_folder output_base --num_workers 12 --full_training`

Run translation
`python run.py translate mnist --load_model --output_folder output_base`
`python run.py translate usps --load_model --output_folder output_base`


Run evaluation
`python run.py eval mnist eval_models/USPS/USPS_model.pth --output_folder output_base --output_results_file mnist2usps_res --dataset_path MNIST_translated_data.pt`
`python run.py eval usps eval_models/MNIST/MNIST_model.pth --output_folder output_base --output_results_file usps2mnist_res --dataset_path USPS_translated_data.pt`


ttttfftttgbg
`pyhon run.py all mnis usps mnis_usps_params.json eval_models/USPS/USPS_model.pth eval_models/MNIST/MNIST_model.pth --ouput_folder output_`
// set max epoch to something like 100 to stop after that?