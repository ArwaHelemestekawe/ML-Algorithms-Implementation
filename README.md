# Machine Learning Algorithms Implementation

A comprehensive collection of machine learning algorithms implemented from scratch in Python. This repository serves as both a learning resource and a reference for understanding the fundamentals of machine learning algorithms.

## 🚀 Project Overview

This repository contains pure Python implementations of various machine learning algorithms. Each implementation includes detailed documentation and mathematical explanations to help understand the underlying concepts.

## 📂 Repository Structure
```
ML-Algorithms-Implementation/
├── supervised_learning/
│   ├── linear_regression.py
│   ├── decision_tree_class.py
│   ├── decision_tree_reg_class.py
│   ├── gradient_boosting.py
│   ├── knn.py
│   ├── poly_class.py
│   └── poly_class_min_max_e1.py
├── unsupervised/
│   ├── LDA/
│   ├── PCA/
│   └── clustering.py
└── utils/
    └── [utility functions]
```

## 🎯 Implemented Algorithms

### Supervised Learning
- Linear Regression: Implementation of simple and multiple linear regression
- Decision Trees: Both classification and regression implementations
- K-Nearest Neighbors (KNN): A non-parametric classification algorithm
- Gradient Boosting: An ensemble learning method
- Polynomial Regression: Implementation with regularization options

### Unsupervised Learning
- Linear Discriminant Analysis (LDA): Dimensionality reduction and classification
- Principal Component Analysis (PCA): Dimensionality reduction technique
- Clustering: Implementation of clustering algorithms

## 🛠️ Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (for dataset generation and comparison)

## 💻 Usage Examples

### Linear Regression
```python
from supervised_learning.linear_regression import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### PCA
```python
from unsupervised.PCA.pca import PCA

# Initialize PCA with number of components
pca = PCA(n_components=2)

# Fit and transform the data
transformed_data = pca.fit_transform(X)

# Get explained variance ratio
variance_ratio = pca.explained_variance_ratio_
```

### LDA
```python
from unsupervised.LDA.lda import LDA

# Initialize LDA
lda = LDA(n_components=2)

# Fit and transform the data
transformed_data = lda.fit_transform(X, y)
```

### Clustering
```python
from unsupervised.clustering import KMeans

# Initialize KMeans
kmeans = KMeans(n_clusters=3)

# Fit the model and get cluster assignments
clusters = kmeans.fit_predict(X)
```

## 🧪 Testing
Each algorithm includes verification against standard datasets and comparison with scikit-learn implementations where applicable.

## 📘 Documentation
Each algorithm implementation includes:
- Theoretical background
- Mathematical formulation
- Performance characteristics

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
