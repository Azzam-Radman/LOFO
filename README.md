# LOFO
Leave One Feature Out (**LOFO**) is on of the most powerful techniques for feature selection. 

This repository contains the implementation of **LOFO** in Python and can be used with any model of the followings:
1. Any Scikit-Learn model.
2. Any TensorFlow/Keras model.
3. LightGBM.
4. CatBoost.
5. XGBoost.

# Usage
- Clone the repo:
```
git clone git@github.com:Azzam-Radman/LOFO.git
```

- Import the needed libraries for your model, cross-validation, etc
```
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgbm
```

- Define the paramters
```
# shutdown warning messages
warnings.filterwarnings('ignore')

X = train_df.iloc[:, :-1]
Y = train_df.iloc[:, -1]
model= lgbm.LGBMClassifier(
                       objective='binary',
                       metric='auc',
                       subsample=0.7,
                       learning_rate=0.03,
                       n_estimators=100,
                       n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
metric = roc_auc_score
direction = 'max'
fit_params = "{'X': x_train, 'y': y_train, 'eval_set': [(x_valid,y_valid)], 'verbose': 0}"
predict_type = 'predict_proba'
return_bad_feats = True
groups = None
is_keras_model = False
```

- Define the LOFO object and call it
```
lofo_object = LOFO(X, Y, model, cv, metric, direction, fit_params, 
                   predict_type, return_bad_feats, groups, is_keras_model)
clean_X, bad_feats = lofo_object()
```

clean_X: is the dataset containing the useful features only.
bad_feats: are the harmful or useless features.
