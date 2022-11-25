"""Train SVM and predict test data."""
# from local
from utils.text_processing import text_preprocessing, text_transfrom
from utils.utils import class_names
from utils.svm_training import TrainingClass

# basics
import pandas as pd
import numpy as np
import json

# ml
import gensim.downloader
import optuna


# Define main settings
EMBEDDING_MODELs = ['glove', 'word2vec']
NUMBER_TRAIN_DATAs = [1, 2]
TRANSFORM = True  # choose False if wordvectors already created and saved


def objective(trial):
    """Run fine_tuning trial."""
    # define parameters
    c = trial.suggest_categorical('c', [0.001, 0.01, 0.1, 1, 10, 100, 1000])
    loss = trial.suggest_categorical('loss', ['hinge', 'squared_hinge'])
    with_weight = trial.suggest_categorical('with_weight', [True, False])

    # train
    f1, _ = train_instance.train(
        balance_opt=best_balance_opt,
        pca_opt=best_pca_opt,
        scale_opt=best_scale_opt,
        comb_opt=best_comb_opt,
        c=c,
        loss=loss,
        with_weight=with_weight
    )

    # add parameters and result to history for documentation
    fine_tuning_history.append({'c': c, 'loss': loss, 'with_weight': with_weight, 'f1': f1})

    return f1


if __name__ == '__main__':

    for EMBEDDING_MODEL in EMBEDDING_MODELs:

        # reset result DataFrame for predictions
        df_preds = pd.DataFrame()
        df_preds.to_csv(f'data/text_test_results/{EMBEDDING_MODEL}_svm.csv')

        for NUMBER_TRAIN_DATA in NUMBER_TRAIN_DATAs:

            # set pretraind model to be downloaded
            if EMBEDDING_MODEL == 'glove':
                pretraind_model_name = 'glove-twitter-25'
                # vector_size = 25
            elif EMBEDDING_MODEL == 'word2vec':
                pretraind_model_name = 'word2vec-google-news-300'
                # vector_size = 300

            # Load data
            train = pd.read_csv(f'data/text_train_{NUMBER_TRAIN_DATA}.csv')
            val = pd.read_csv('data/text_val.csv')
            test = pd.read_csv('data/text_test.csv')

            # preprocess data
            train = text_preprocessing(train, lemmantization=False, stop_words=False, lower_case=True)
            val = text_preprocessing(val, lemmantization=False, stop_words=False, lower_case=True)
            test = text_preprocessing(test, lemmantization=False, stop_words=False, lower_case=True)

            # Glove set II: lemmantization=False, stop_words=False, lower_case=False -> ratio: 0.14
            # Glove set II: lemmantization=False, stop_words=False, lower_case=True -> ratio: 0.024
            # Glove set II: lemmantization=False, stop_words=True, lower_case=True -> ratio: 0.041
            # Glove set II: lemmantization=True, stop_words=True, lower_case=True ->  ratio: 0.133

            # creat paths for save/load word vectors
            train_path = f'training/text/base_model/cache/text_train_{NUMBER_TRAIN_DATA}_{EMBEDDING_MODEL}.npy'
            val_path = f'training/text/base_model/cache/text_val_{EMBEDDING_MODEL}.npy'
            test_path = f'training/text/base_model/cache/text_test_{EMBEDDING_MODEL}.npy'

            # If data not already embedded, transform it and save
            if TRANSFORM is True:

                # load model
                embedding_model = gensim.downloader.load(pretraind_model_name)
                print('model loaded')

                # transform and save train
                print(f'start transforming train {NUMBER_TRAIN_DATA}')
                X_train, words_not_in_vocab = text_transfrom(train['text'], embedding_model)
                np.save(train_path, X_train)
                print('vectors created and saved')

                # Save words that could not be tranformed
                with open(f'training/text/base_model/words_not_transformed/{NUMBER_TRAIN_DATA}_{EMBEDDING_MODEL}.txt', 'w') as f:
                    f.write(str(words_not_in_vocab))

                # transform and save val
                print(f'start transforming val {NUMBER_TRAIN_DATA}')
                X_val, _ = text_transfrom(val['text'], embedding_model)
                np.save(val_path, X_val)
                print('vectors created and saved')

                # transform and save test
                print(f'start transforming test {NUMBER_TRAIN_DATA}')
                X_test, _ = text_transfrom(test['text'], embedding_model)
                np.save(test_path, X_test)
                print('vectors created and saved')

            # if already transformed load from path
            else:
                # load
                X_train = np.load(train_path)
                X_val = np.load(val_path)
                X_test = np.load(test_path)

            # Build y
            y_train = train[class_names].values
            y_val = val[class_names].values
            y_test = test[class_names].values

            # initialize Training class
            train_instance = TrainingClass(
                train=train,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
            )

        # 1. Scaling

            # define scaling options
            scaling_options = ['no_scaling', 'StandardScaler', 'RobustScaler']
            scaling_results = {}

            # for each option scale X, train SVM and save result in scaling_results
            for scale_opt in scaling_options:

                # train with respective scaling option
                f1, _ = train_instance.train(
                    balance_opt='original',
                    pca_opt='no_pca',
                    scale_opt=scale_opt,
                    comb_opt='one_vs_rest',
                    c=1,
                    loss='hinge',
                    with_weight=False
                )

                # save result in dict
                scaling_results[scale_opt] = f1
                print('Scaling opt: ', scale_opt, ' done')

            # get option with highest F1
            best_scale_opt = max(scaling_results, key=scaling_results.get)
            best_f1 = scaling_results[best_scale_opt]

            # print best Scaling
            print(scaling_results)
            print('Best Scaling Methode: ', best_scale_opt, ' Best F1 Score: ', best_f1)
            print('_' * 10)

        # 2. Balancing

            # set balancing options already prepared
            # add with_weight to options
            balance_options = ['ratio_0.25', 'ratio_0.50', 'ratio_0.75', 'ratio_0.90']

            # initialize result dict already including the f1 from previously results
            balance_results = {'original': best_f1}

            # for each option filter data from option and train SVM
            for balance_opt in balance_options:

                # train with respective balance option
                f1, _ = train_instance.train(
                    balance_opt=balance_opt,
                    pca_opt='no_pca',
                    scale_opt=best_scale_opt,
                    comb_opt='one_vs_rest',
                    c=1,
                    loss='hinge',
                    with_weight=False
                )

                # add f1 to result dict
                balance_results[balance_opt] = f1
                print('Balance opt: ', balance_opt, ' done')

            # find best balancing ratio and f1
            best_balance_opt = max(balance_results, key=balance_results.get)
            best_f1 = balance_results[best_balance_opt]

            # print
            print(balance_results)
            print('Best Balancing: ', best_balance_opt, ' Best F1 Score: ', best_f1)
            print('_' * 10)

        # 3. PCA

            # set ratios for dimensions to be selected
            pca_options = [1, 0.75, 0.5]

            # initialize result dict already including the best f1 from previously results
            pca_results = {'no_pca': best_f1}

            # for each ratio reduce the dimensionality and train SVM
            for pca_opt in pca_options:

                f1, _ = train_instance.train(
                    balance_opt=best_balance_opt,
                    pca_opt=pca_opt,
                    scale_opt=best_scale_opt,
                    comb_opt='one_vs_rest',
                    c=1,
                    loss='hinge',
                    with_weight=False
                )

                # add f1 to result dict
                pca_results[pca_opt] = f1
                print('PCA opt: ', pca_opt, ' done')

            # find best option
            best_pca_opt = max(pca_results, key=pca_results.get)
            best_f1 = pca_results[best_pca_opt]

            # prints
            print(pca_results)
            print('Best PCA rate: ', best_pca_opt, ' Best F1 Score: ', best_f1)
            print('_' * 10)

        # 4. Classifier Chain

            # initialize result dict with previously best f1 using OneVsRestClassifier
            comb_results = {'one_vs_rest': best_f1}

            # Train with Classifier Chain
            f1, _ = train_instance.train(
                balance_opt=best_balance_opt,
                pca_opt=pca_opt,
                scale_opt=best_scale_opt,
                comb_opt='classifier_chain',
                c=1,
                loss='hinge',
                with_weight=False
            )

            # add f1 to result dict
            comb_results['classifier_chain'] = f1
            print('Classifier Chain done')

            # find best option
            best_comb_opt = max(comb_results, key=comb_results.get)
            best_f1 = comb_results[best_comb_opt]

            # print
            print(comb_results)
            print('Best Combi: ', best_comb_opt, ' Best F1 Score: ', best_f1)

        # 5. Fine-Tuning

            # initialize history with the previously best f1 score with default parameters
            fine_tuning_history = [{'c': 1.0, 'loss': 'squared_hinge', 'with_weight': False, 'f1': best_f1}]

            # run fine_tuning with optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=7)

            # get best trial
            best_opt_number = np.argmax([trial['f1'] for trial in fine_tuning_history])
            best_opt = fine_tuning_history[best_opt_number]

            # print best_opt
            print('Best Fine_Tuning Opt: ', best_opt)

        # Final Run and Prediction

            # train and predict
            _, y_pred = train_instance.train(
                balance_opt=best_balance_opt,
                pca_opt=best_pca_opt,
                scale_opt=best_scale_opt,
                comb_opt=best_comb_opt,
                c=best_opt['c'],
                loss=best_opt['loss'],
                with_weight=best_opt['with_weight'],
                return_preds=True
            )

            # Save predictions in dataframe
            df_preds = pd.read_csv(f'data/text_test_results/{EMBEDDING_MODEL}_svm.csv')
            df_preds['target'] = y_test.tolist()
            df_preds[f'pred_{NUMBER_TRAIN_DATA}'] = y_pred.tolist()
            df_preds.to_csv(f'data/text_test_results/{EMBEDDING_MODEL}_svm.csv')

        # Save history

            # create dict with all results
            history = {
                'scaling': scaling_results,
                'balancing': balance_results,
                'feature_selection': pca_results,
                'combination': comb_results,
                'fine_tuning': fine_tuning_history,
                'final': best_opt,
            }

            # read svm histories
            with open('data/histories/base_model_histories.json', 'r') as f:
                data = json.load(f)

            # add history to histories
            data[f'{NUMBER_TRAIN_DATA}_{EMBEDDING_MODEL}_svm'] = history

            # save histories as json
            with open('data/histories/base_model_histories.json', 'w') as f:
                json_data = json.dumps(data, indent=4)
                f.write(json_data)
