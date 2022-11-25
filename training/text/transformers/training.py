from utils.train_evaluate import TrainingClass
from utils.helper import class_names

import json
import torch
import optuna
import pandas as pd
import numpy as np
from utils.helper import get_default_device
device = get_default_device()

MAX_LEN = 512
EPOCHS = 1
MODEL_NAME = 'bert'  # 'bert', 'roberta', 'xlnet', 'gpt2'
NUMBER_TRAIN_DATA = 1


def objective(trial):

    # set categories of hyperparameters / get random parameters
    parameters['batch_size'] = trial.suggest_categorical("batch_size", [4 * 4, 4 * 8])
    parameters['lr'] = trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    parameters['warmup_ratio'] = trial.suggest_categorical("warmup_ratio", [0.06, 0.1])
    parameters['weight_decay'] = trial.suggest_categorical("weight_decay", [0.01, 0.1])
    parameters['with_weight'] = trial.suggest_categorical("with_weight", [True, False])

    # train classifier
    history = train_class.train(parameters, trial=trial)

    # add results to history
    finetuning_history.append(history)

    # return max F1
    return np.max(history['val_f1'])


if __name__ == '__main__':

    # prints
    print(f'Start training model: {MODEL_NAME} with dataset {str(NUMBER_TRAIN_DATA)} and {str(EPOCHS)} epochs')
    print('-' * 20)

    # load train and val data
    df_train = pd.read_csv(f'data/text_train_{NUMBER_TRAIN_DATA}.csv')
    df_val = pd.read_csv('data/text_val.csv')

    # shuffle data
    df_train = df_train.sample(frac=1).reset_index(drop=True)[:10]
    df_val = df_val.sample(frac=1).reset_index(drop=True)[:5]

    # create X and y for train
    X_train = df_train['text']
    y_train = df_train[class_names].to_numpy()
    X_train.reset_index(inplace=True, drop=True)

    # create X and y for val
    X_val = df_val['text']
    y_val = df_val[class_names].to_numpy()
    X_val.reset_index(inplace=True, drop=True)

    # print the shapes for validation
    print('X_train shape: ', X_train.shape)
    print('y_train shape: ', y_train.shape)
    print('-' * 20)
    print('X_val shape: ', X_val.shape)
    print('y_val shape: ', y_val.shape)
    print('-' * 20)

    # initialize history to be filled
    main_history = {}

# Data-specific modifications

    # set parameters for data-specific modeifications
    parameters = {
        'epochs': 10,  # for fine_tuning: 10, for final: 15
        'batch_size': 4 * 8,  # will be fine_tuned
        'lr': 3e-05,  # will be fine_tuned
        'with_weight': False,  # will be validated in Modification
        'warmup_ratio': 0,  # will be fine_tuned
        'weight_decay': 0,  # will be fine_tuned
        'balancing': None,  # will be validated in Modification
        'further_pretraining': False,  # will be validated in Modification
        'trial_pruned': False
    }


# 1. Balancing

    # set options for balancing
    balance_options = ['original', 'ratio_0.25', 'ratio_0.50', 'ratio_0.75', 'ratio_0.90']
    balance_results = {}
    balance_history = {}

    # for each option train and evaluate a classifier
    for opt in balance_options:

        # change balancing in parameters for documentation
        parameters['balancing'] = opt

        # apply balance ratio by filtering in respectif column for 1
        X_train_bal = X_train[df_train[opt] == 1].reset_index(drop=True)
        y_train_bal = y_train[df_train[opt] == 1]

        # initialize Training Class
        train_class = TrainingClass(
            device=device,
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_val=X_val,
            y_val=y_val,
            MAX_LEN=MAX_LEN,
            MODEL_NAME=MODEL_NAME,
            NUMBER_TRAIN_DATA=NUMBER_TRAIN_DATA,
            further_pretrained=False,
            label_weights=None
        )

        # train and save max f1
        print('Balance Option:', opt)
        history = train_class.train(parameters)
        balance_history[opt] = history
        balance_results[opt] = np.max(history['val_f1'])

    # find best option
    best_opt = max(balance_results, key=balance_results.get)
    best_f1 = balance_results[best_opt]
    print('Best Balancing: ', best_opt, ' Best F1: ', best_f1)

    # Adapt best option
    parameters['balancing'] = best_opt
    X_train = X_train[df_train[best_opt] == 1].reset_index(drop=True)
    y_train = y_train[df_train[best_opt] == 1]

    # calculate label_weights according to best balancing option for Fine-Tuning
    amount_samples_sdg = y_train.sum(axis=0)
    label_weights = torch.tensor(((y_train.shape[0] - amount_samples_sdg) / amount_samples_sdg).tolist())

    # add results to history
    main_history['modification'] = {'balancing': balance_history}

# 1. Pre-training

    # change further_pretraining in parameters for documentation
    parameters['further_pretraining'] = True

    # load further pretrained model
    train_class = TrainingClass(
        device=device,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        MAX_LEN=MAX_LEN,
        MODEL_NAME=MODEL_NAME,
        NUMBER_TRAIN_DATA=NUMBER_TRAIN_DATA,
        further_pretrained=True,  # here change to True
        label_weights=None
    )

    # train and save max F1
    print('Further Pretrained')
    history = train_class.train(parameters)
    pretrain_history = history
    fp_f1 = np.max(history['val_f1'])

    # Adapt best option
    if fp_f1 < best_f1:
        further_pretrained = False
        parameters['further_pretraining'] = False
        print('-> Stay with Pretrained Model')
    else:
        parameters['further_pretraining'] = True
        further_pretrained = True
        print('-> Chose Further Pretrained Model')

    # add results to history
    main_history['modification']['further_pretraining'] = pretrain_history


# 3. Fine-Tuning

    # initialize list for histories
    finetuning_history = []

    # initialize class with data according to best balancing ratio and best further_pretrain option
    train_class = TrainingClass(
        device=device,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        MAX_LEN=MAX_LEN,
        MODEL_NAME=MODEL_NAME,
        NUMBER_TRAIN_DATA=NUMBER_TRAIN_DATA,
        further_pretrained=further_pretrained,
        label_weights=label_weights
    )

    # run fine-tuning with 20 trials
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    trial = study.best_trial

    # print results
    print('Loss: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # add results to history
    main_history['fine_tuning'] = finetuning_history

# 4. Final Run

    # set parameters for final run using the best trail from fine-tuning
    parameters['epochs'] = 15  # run with more epochs to maybe even retrieve better F1-scores
    parameters['batch_size'] = trial.params['batch_size']
    parameters['lr'] = trial.params['lr']
    parameters['warmup_ratio'] = trial.params['warmup_ratio']
    parameters['weight_decay'] = trial.params['weight_decay']
    parameters['with_weight'] = trial.params['with_weight']

    # train and add results to history
    main_history['final'] = train_class.train(parameters, save_model=True)

    # save history in /data/histories folder
    with open(f'data/histories/{NUMBER_TRAIN_DATA}_{MODEL_NAME}.json', 'w') as f:
        json_data = {}
        json_data[f'{NUMBER_TRAIN_DATA}_{MODEL_NAME}'] = main_history
        jason_obj = json.dumps(json_data, indent=4)
        f.write(jason_obj)
