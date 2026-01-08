# lstnet
LSTNet is deep learning model performing image-to-image translation to tackle domain adaptation.

## Requirements
Python version `>=3.11` is required. 
Based on the OS version and if GPU is available, install correct torch and torchvision packages.
https://pytorch.org/

The rest of the requierements are specified in `requirements.txt`. 


## Training, Image Translation and Evaluation
### Run Training

`python run.py train mnist usps params/mnsit_usps_params.json --output_folder output_mnist_usps`

### Run translation
`python run.py translate mnist --load_model --output_folder output_base`
Loads model from the output folder and returns the translated images there.


### Run evaluation
`python run.py eval mnist eval_models/USPS/USPS_model.pth --output_folder output_base --dataset_path output_base/MNIST_translated_data.pt`

`python run.py eval usps eval_models/MNIST/MNIST_model.pth --output_folder output_base --dataset_path output_base/USPS_translated_data.pt`


`pyhon run.py all mnis usps mnis_usps_params.json eval_models/USPS/USPS_model.pth eval_models/MNIST/MNIST_model.pth --ouput_folder output_base`


Run Optuna
`python run.py train mnist usps params/mnist_usps_params.json --output_folder output_optuna_tuning --num_workers 48 --optuna --optuna_study_name tuning --optuna_trials 200 --optuna_sampler_start_trials 30 --optuna_pruner_sample_trials 20 --optuna_pruner_warmup_steps 30 --optuna_pruner_interval_steps 10 --percentile 90 --hyperparam_mode augm_ops train_params architecture`

## Training Classifiers

### Evaluation
`python run.py eval mnist mnist_clf_base/MNIST_clf_model.pth --output_folder mnist_clf_base --output_results_file  mnist_clf_eval`




# CORRECT CLF Evaluation
`python run.py eval office_31_webcam $webcam_clf --resize_target_size 224`


## Command Line Arguments Reference

### Common Arguments (Available for all operations: train, translate, eval, all)

| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--output_folder` | No | `output/` | Path to the output folder |
| `--batch_size` | No | `64` | Size of batches used in data loaders |
| `--num_workers` | No | `4` | Number of worker threads for data loading |
| `--manual_seed` | No | `42` | Manual seed for random number generators used in data splitting |
| `--resize_target_size` | No | `None` | If set, images are resized to the provided size, preserving aspect ratio |

### Train Operation Arguments

#### Positional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `first_domain` | Yes | - | Name of the first dataset |
| `second_domain` | Yes | - | Name of the second dataset |
| `params_file` | Yes | - | Path to the file with stored parameters of model |

#### Optional Arguments

**Training Configuration**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--supervised` | No | `False` | Run supervised domain adaptation. If not set, unsupervised domain adaptation is run |
| `--disc_update_freq` | No | `2` | Number of discriminator updates per encoder-generator update |
| `--optim_name` | No | `Adam` | Name of the optimizer to use |
| `--learning_rate` | No | `1e-4` | Learning rate used in an optimizer |
| `--betas` | No | `(0.8, 0.999)` | Momentum parameters for optimizer (2 values) |
| `--weight_decay` | No | `1e-2` | Weight decay for optimizer |
| `--full_training_only` | No | `False` | If set, the full training set will be used. No validation phase after training |
| `--epoch_num` | No | `50` | Number of training epochs |
| `--val_size` | No | `0.25` | Proportion of data used for validation |
| `--early_stopping` | No | `False` | Enable early stopping during training |
| `--patience` | No | `10` | Number of epochs with no improvement after which training will be stopped |

**Model Saving**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--model_file_name` | No | `lstnet.pth` | Name of the model under which the trained LSTNET model will be saved in the `output_folder` |
| `--logs_file_name` | No | `loss_logs.json` | Name of the file where training loss logs will be saved |

**Data Augmentation**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--rotation` | No | `10` | Degree of rotation for data augmentation |
| `--zoom` | No | `0.1` | Zoom factor for data augmentation |
| `--shift` | No | `2` | Pixel shift for data augmentation |
| `--strong_augment` | No | `False` | Whether to run strong or weak augmentation. Weak augmentation uses only zoom, shift and rotation. Strong augmentation includes horizontal flip and color jitter |
| `--horizontal_flip_prob` | No | `0.3` | Probability of horizontal flip for strong augmentation |
| `--color_jitter_brightness` | No | `0.3` | Brightness factor for color jitter in strong augmentation |
| `--color_jitter_contrast` | No | `0.3` | Contrast factor for color jitter in strong augmentation |
| `--color_jitter_saturation` | No | `0.3` | Saturation factor for color jitter in strong augmentation |
| `--color_jitter_hue` | No | `0.1` | Hue factor for color jitter in strong augmentation |
| `--inplace_augmentation` | No | `False` | If set, data augmentation will be performed inplace to reduce memory usage |

**Resize Operations**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--pad_mode` | No | `edge` | Padding mode for resize operations. Options are: constant, edge, reflect, symmetric |
| `--random_crop_resize` | No | `False` | If set, random resized crop will be applied during resizing |
| `--resize_init_size` | No | `256` | Initial size to which images are resized before random crop resize |
| `--resized_crop_scale` | No | `0.8 1.0` | Scale range for random resized crop (min max, 2 values) |
| `--resized_crop_ratio` | No | `0.9 1.1` | Aspect ratio range for random resized crop (min max, 2 values) |

**Loss and Model Configuration**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--weights` | No | `[20, 20, 30, 100, 100, 100, 100]` | List of 7 float weights for the loss components |
| `--use_checkpoint` | No | `False` | If set, gradient checkpointing will be enabled to reduce GPU memory usage |
| `--wasserstein` | No | `False` | If set, use Wasserstein critique approach instead of adversarial discriminator |

**Dataset Options**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--use_svhn_extra` | No | `False` | If set, the extra SVHN training data will be used when SVHN is one of the domains |

**Optuna Hyperparameter Optimization**
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--optuna` | No | `False` | If set, Optuna hyperparameter optimization will be performed |
| `--optuna_study_name` | No | `lstnet_study` | Name for the Optuna study |
| `--optuna_trials` | No | `50` | Number of Optuna trials to perform if --optuna is set |
| `--optuna_sampler_start_trials` | No | `20` | Number of initial trials used as exploratory for TPE sampler |
| `--optuna_pruner_sample_trials` | No | `50` | Number of trials to consider for median pruner |
| `--optuna_pruner_warmup_steps` | No | `15` | Number of warmup steps before starting to prune unpromising trials |
| `--optuna_pruner_interval_steps` | No | `5` | Interval steps for pruning checks |
| `--percentile` | No | `10` | Percentile for pruning threshold |
| `--hyperparam_mode` | No | `[]` | List of hyperparameter modes to run. Options: `weights`, `weights_reduced`, `augm_ops`, `train_params`, `architecture` |

### Translate Operation Arguments

#### Positional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `domain` | Yes | - | Name of the domain to be translated to the other domain |

#### Optional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--load_model` | No | `False` | If a model with name 'model_name' should be loaded for data translation |
| `--model_name` | No | `lstnet.pth` | Name of the model to be loaded for translation |

### Eval Operation Arguments

#### Positional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `domain` | Yes | - | Name of the domain to be evaluated |
| `clf_model` | Yes | - | Name of the model to classify the data |

#### Optional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--dataset_path` | No | `""` | Name of file to load the dataset from |
| `--log_name` | No | `test_acc` | Name for the evaluation metric in the results file |

### All Operation Arguments (End-to-End)

This operation combines training, translation, and evaluation. It uses all train operation arguments plus:

#### Additional Positional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `clf_first_domain` | Yes | - | Path to the trained classifier of the first domain |
| `clf_second_domain` | Yes | - | Path to the trained classifier of the second domain |

#### Additional Optional Arguments
| Argument | Mandatory | Default | Description |
|----------|-----------|---------|-------------|
| `--save_trans_data` | No | `False` | If set, the translated data should be saved |
