# Document Classification

This notebook performs both **unsupervised** and **supervised** classification on a spam email dataset, including hyperparameter optimization. Below is an outline of each step, with explanations of what was done and why.

---

## 1. Setup and Dependencies

1. **Install required libraries**
   Added a cell to install `gensim` if it isn’t already present:

   ```python
   !pip install gensim
   ```
2. **Imports**
   Loaded essential Python packages for data handling, machine learning, clustering, visualization, and word embeddings.

---

## 2. Data Loading and Inspection

* **Read CSV**
  Loaded `spam.csv` (with Latin-1 encoding) into a pandas DataFrame, selecting only the `text` and `target` columns.
* **Label mapping**
  Converted the `target` column from string labels (`ham`/`spam`) to numeric (`0`/`1`).
* **Display checks**

  * Printed the first five rows to inspect structure and content.
  * Printed the distribution of ham vs. spam messages to understand class balance.

---

## 3. Preprocessing: TF-IDF and PCA

* **TF-IDF vectorization**
  Transformed each email’s text into a 5,000-dimensional TF-IDF feature vector.

  * **Why TF-IDF?** It weights terms by how unique they are across the corpus, reducing the impact of very common words.
* **Inspection**
  Printed the TF-IDF matrix shape to confirm the number of documents and features.
* **Optional PCA**
  Reduced the TF-IDF features to 2 principal components for ease of visualization and clustering.

  * Printed the explained variance ratios of the two PCs to see how much information they captured.

---

## 4. Unsupervised Learning: Clustering

1. **K-means clustering**

   * Ran K-means with `k=2` (spam vs. ham) on the 2D PCA‐transformed data.
   * Stored cluster labels in the DataFrame.
2. **Hierarchical clustering**

   * Computed a linkage matrix using Ward’s method.
   * Cut the dendrogram into two clusters and stored labels.
3. **Cluster inspection**
   Printed the number of points assigned to each cluster for both methods to gauge balance.

---

## 5. Visualization

* **Prepare PC DataFrame**
  Created a small DataFrame of the first two principal components.
* **Scatter plots**

  1. **Clusters**: Colored by K-means cluster assignment.
  2. **True labels**: Colored by the actual `spam`/`ham` labels.
* **Purpose**
  To visually assess how well the clusters align with the true categories in the 2D PCA space.

---

## 6. Word Embeddings & Data Preparation

1. **Tokenization**
   Split each email into a list of words.
2. **Word2Vec training**
   Trained a Word2Vec model on the entire corpus (vector size 100, window 5).

   * *Alternative:* GloVe embeddings could be loaded instead.
3. **Embedding emails**
   For each email, averaged its word vectors to produce a single 100-dimensional feature vector.
4. **Inspection**
   Printed one sample embedding vector to verify shape and content.
5. **Train/test split**
   Split the embeddings and labels into training (80%) and testing (20%) sets, printing their sizes.

---

## 7. Supervised Learning

Tested four classifiers on the Word2Vec embeddings:

* **Naive Bayes** (`GaussianNB`)
* **Random Forest** (`RandomForestClassifier`)
* **Gradient Boosting** (`GradientBoostingClassifier`)
* **XGBoost** (`XGBClassifier`)

For each model:

1. Fitted on the training set.
2. Predicted labels on the test set.
3. Printed the full classification report—including precision, recall, F1-score—for immediate feedback.

---

## 8. Hyperparameter Optimization

Applied `GridSearchCV` (3-fold CV, optimizing F1 score) to tune:

* **Random Forest**:
  – `n_estimators`: \[50, 100, 200]
  – `max_depth`: \[None, 10, 20]
  – `min_samples_split`: \[2, 5]
* **Gradient Boosting**:
  – `n_estimators`: \[50, 100, 200]
  – `learning_rate`: \[0.01, 0.1, 0.2]
  – `max_depth`: \[3, 5]
* **XGBoost**:
  – `n_estimators`: \[50, 100, 200]
  – `learning_rate`: \[0.01, 0.1, 0.2]
  – `max_depth`: \[3, 5]

For each, we:

1. Ran the grid search.
2. Printed the best parameter combination.
3. Stored the results for comparison.

---

**Result:**

