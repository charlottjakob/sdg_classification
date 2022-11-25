"""Train SVM according to parameters."""
# ml
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.decomposition import KernelPCA


class TrainingClass():

    def __init__(self, train, X_train, y_train, X_val, y_val, X_test, y_test):

        # save data to train models using same data but different parameters
        self.balance_cols = train
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test

    def train(self, balance_opt, pca_opt, scale_opt, comb_opt, c, loss, with_weight, return_preds=False):

        # get data of balancing option by filtering respective column
        X_train = self.X_train[self.balance_cols[balance_opt] == 1]
        y_train = self.y_train[self.balance_cols[balance_opt] == 1]

        # get rest of the data from class
        X_val = self.X_val[:]
        y_val = self.y_val[:]
        X_test = self.X_test[:]

        # Feature Selection
        if pca_opt != 'no_pca':

            # calculate number of dimensions to be selected as new dimensionality = old dimensionality * ratio
            dim_new = int(X_train.shape[1] * pca_opt)

            # train PCA with train data and transfrom validation
            pca = KernelPCA(dim_new)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)

        # Scaling
        if scale_opt != 'no_scaling':
            if scale_opt == 'StandardScaler':
                scaler = StandardScaler()
            elif scale_opt == 'RobustScaler':
                scaler = RobustScaler()

            # transform
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Multi-label problem transformation
        if comb_opt == 'one_vs_rest':
            comb = OneVsRestClassifier  # Binary Relevance
        elif comb_opt == 'classifier_chain':
            comb = ClassifierChain

        # Cost Sensitive Learning, set class_weight
        class_weight = 'balanced' if with_weight is True else None

        # train SVM
        svm = comb(LinearSVC(C=c, loss=loss, max_iter=10000, class_weight=class_weight))
        svm.fit(X_train, y_train)

        # predict and score
        y_pred = svm.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='micro')

        # for final run transfrom test set and predict
        if return_preds is True:

            # Feature Selection
            if pca_opt != 'no_pca':
                X_test = pca.transform(X_test)

            # Scaling
            if scale_opt != 'no_scaling':
                X_test = scaler.transform(X_test)

            # Predict
            y_test_pred = svm.predict(X_test)

        else:
            y_test_pred = None

        # return F1 score of validation set and predictions of textset if not None
        return f1, y_test_pred
