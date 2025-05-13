## 🧠 LOFO (Leave One Feature Out)

> One of the most **powerful and intuitive techniques** for feature selection — now in Python, compatible with Scikit-Learn, LightGBM, XGBoost, CatBoost, and TensorFlow/Keras models.

![LOFO Banner](https://github.com/Azzam-Radman/LOFO/blob/main/assets/LOFO.png)
*Visualize the power of removing irrelevant features.*

---

### 📌 Features

* 🔍 Model-agnostic feature selection.
* 💡 Simple API compatible with your favorite ML frameworks.
* 🚀 Optimized for performance using CV-based feature elimination.

---

### 📦 Supported Models

| Framework        | Supported |
| ---------------- | --------- |
| Scikit-Learn     | ✅         |
| TensorFlow/Keras | ✅         |
| LightGBM         | ✅         |
| CatBoost         | ✅         |
| XGBoost          | ✅         |

---

### 🚀 Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/Azzam-Radman/LOFO.git
```

<sub>If you're using a notebook, prepend with `!`</sub>

#### 2. Import LOFO module

```python
import os, sys
sys.path.append(os.path.join(os.getcwd(), 'LOFO'))
import lofo
```

---

### 📖 Usage Examples

#### ✅ Scikit-Learn

<details>
<summary>Click to expand</summary>

```python
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
X = train_df.iloc[:, :-1]
Y = train_df.iloc[:, -1]

model = LogisticRegression()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
metric = roc_auc_score
fit_params = "{'X': x_train, 'y': y_train}"

lofo_object = lofo.LOFO(X, Y, model, cv, metric, 'max', fit_params, 'predict_proba', True, None, False)
clean_X, bad_feats = lofo_object()
```

</details>

---

#### 🌿 LightGBM

<details>
<summary>Click to expand</summary>

```python
import lightgbm as lgbm

model = lgbm.LGBMClassifier(...)
fit_params = "{'X': x_train, 'y': y_train, 'eval_set': [(x_valid, y_valid)], 'verbose': 0}"
# Call LOFO same as before
```

</details>

---

#### 🤖 TensorFlow/Keras

<details>
<summary>Click to expand</summary>

```python
def nn_model():
    inputs = tf.keras.Input(shape=(X.shape[-1],))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, output)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
```

</details>

---

### 📂 Files

* `lofo.py`: Main implementation (see [lofo.py](https://github.com/Azzam-Radman/LOFO/blob/main/lofo.py))

---

### 💬 Output

```python
clean_X  # DataFrame with only useful features
bad_feats  # List of harmful or useless features
```

---

### 📸 Suggested Additions

* Add a **logo** in the header (you can design one using [Canva](https://www.canva.com/) or [Hatchful](https://hatchful.shopify.com/)).
* Add **badges** like:

```md
![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![License](https://img.shields.io/github/license/Azzam-Radman/LOFO)
![Stars](https://img.shields.io/github/stars/Azzam-Radman/LOFO?style=social)
```
