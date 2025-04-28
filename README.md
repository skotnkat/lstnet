# lstnet
LSTNet model for domain adaptation in computer vision.


Run training
`python run.py train mnist usps mnist_usps_params.json --output_folder output_p1_e4 --patience 1 --num_workers 10`

Run translation
`python run.py translate mnist --load_model --model_name lstnet_model.pth --output_folder output_combi --output_data_file mnist_translated`
`python run.py translate usps --load_model --model_name lstnet_model.pth --output_folder output_combi --output_data_file usps_translated`


Run evaluation
`python run.py eval mnist eval_models/USPS/USPS_model.pth --output_folder output_hf_ll_label_smooth --output_results_file mnist2usps_res --dataset_path MNIST_translated_data.pt`
`python run.py eval usps eval_models/MNIST/MNIST_model.pth --output_folder output --output_results_file usps2mnist_res --dataset_path USPS_translated_data.pt`


ttttfftttgbg
`pyhon run.py all mnis usps mnis_usps_params.json eval_models/USPS/USPS_model.pth eval_models/MNIST/MNIST_model.pth --ouput_folder output_`
// set max epoch to something like 100 to stop after that?