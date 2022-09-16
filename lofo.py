import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

class LOFO(object):
    def __init__(self, X, Y, model, cv, metric, direction, fit_params=None, 
                 predict_type='predict', return_bad_feats=False, groups=None,
                 is_keras_model=False):
        '''
        Inputs: 
            X: (Pandas DataFrame) input features to the model (predictors)
            Y: (array-like) target/label feature
            model: (object) the model class (e.g. sklearn.linear_model.LinearRegression())
            cv: (object) sklearn cross validatoin object (e.g. sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0))
            metric: (object) metric to use during search (e.g. sklearn.metrics.roc_auc_score)
            direction: (str) direction of optimization ('max' or 'min')
            fit_params: (str) parameters to use for fitting (e.g. "{'X': x_train, 'y': y_train}") . Default: fit_params = "{'X': x_train, 'y': y_train}"
            predict_type: (str) 'predict' or 'predict_proba'
            return_bad_feats: (boolean) whether to return a list with bad features
            groups: (array-like) used with StratifiedGroupKFold
            is_keras_model: (boolean) whether the model passed is Keras model
            
        Returns:
            A Pandas DataFrame with bad features removed
            if return_bad_feats is set to True, the class returns the bad features list beside the DataFrame
        '''
        self.X = X
        self.Y = Y
        self.model = model
        self.cv = cv
        self.metric = metric
        self.direction = direction
        self.fit_params = fit_params
        self.predict_type = predict_type
        self.return_bad_feats = return_bad_feats
        self.is_keras_model = is_keras_model
        
    def cross_validation(self, X):
        
        if 'Group' in type(self.cv).__name__:
            splits = self.cv.split(X, self.Y, self.groups)
        elif 'Stratified' in type(self.cv).__name__:
            splits = self.cv.split(X, self.Y)
        else:
            splits = self.cv.split(X)
        
        for train_idx, valid_idx in splits:
            x_train, x_valid = X.iloc[train_idx, :].values, X.iloc[valid_idx, :].values
            y_train, y_valid = self.Y.iloc[train_idx].values, self.Y.iloc[valid_idx].values
            
            if self.is_keras_model:
                model = self.model
                model.load_weights('original_weights.h5')
            else:
                model = copy.deepcopy(self.model)
            
            if self.fit_params is None:
                model.fit(x_train, y_train)
            else:
                model.fit(**eval(self.fit_params))
            
            if self.predict_type == 'predict':
                valid_preds = model.predict(x_valid)
            elif self.predict_type == 'predict_proba':
                valid_preds = model.predict_proba(x_valid)[:, -1]
                
            valid_score = self.metric(y_valid, valid_preds)
            
            return valid_score
        
    def __call__(self):
        
        if self.is_keras_model:
            self.model.save_weights('original_weights.h5')
        
        feats = self.X.copy()
        all_feats = feats.columns.tolist()
        bad_feats = []
        
        print(f'Number of columns before LOFO: {len(all_feats)}')
        print(f'Score before LOFO: {self.cross_validation(feats)}')
        
        if self.direction == 'min':
            best_score = np.inf
        elif self.direction == 'max':
            best_score = -np.inf
        
        print('\nLOFO in progress...')
        for col in tqdm(all_feats, total=len(all_feats)):
            if self.is_keras_model:
                current_feat = feats[col].values
                # set feature values to 0 let the network ignore it
                feats[col] = 0
                valid_score = self.cross_validation(feats)
                # return column values
                feats[col] = current_feat
            else:
                valid_score = self.cross_validation(feats.drop(col, axis=1))
                
            if self.direction == 'min':
                if valid_score <= best_score:
                    best_score = valid_score
                    bad_feats.append(col)
                    if self.is_keras_model:
                        feats[col] = 0
                    else:
                        feats = feats.drop(col, axis=1)
            
            elif self.direction == 'max':
                if valid_score >= best_score:
                    best_score = valid_score
                    bad_feats.append(col)
                    if self.is_keras_model:
                        feats[col] = 0
                    else:
                        feats = feats.drop(col, axis=1)
                    
        good_feats = [feat for feat in all_feats if feat not in bad_feats]
        
        print(f'Number of columns after LOFO: {len(good_feats)}')
        print(f'Score after LOFO: {best_score}')
        
        if self.return_bad_feats:
            return self.X[good_feats], bad_feats
        
        return self.X[good_feats]
