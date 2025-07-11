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

**Hyperparameter Optimization Results**

After running `GridSearchCV` on the Word2Vec-based supervised models, we obtained the following best parameter combinations:

| Model             | Best Parameters                                                  |
| ----------------- | ---------------------------------------------------------------- |
| **Random Forest** | `{'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5}` |
| **GBM**           | `{'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 5}`    |
| **XGBoost**       | `{'n_estimators': 200, 'learning_rate': 0.2, 'max_depth': 5}`    |

**Interpretation:**

* **Higher tree counts (`n_estimators=200`)** reduced variance and improved stability across all ensemble methods.
* **Random Forest** benefitted from **deeper trees** (`max_depth=20`) to capture complex decision boundaries.
* **Boosting methods** (GBM, XGBoost) favored **shallower trees** (`max_depth=5`) paired with a **moderate learning rate** (`0.2`) to balance convergence speed and generalization.
* **`min_samples_split=5`** in Random Forest helped prevent overfitting by requiring more samples to split an internal node.




# N-grams

**Objective:**

* Analyze textual data by extracting and visualizing n-gram frequencies (unigrams, bigrams, trigrams).

**Data & Preprocessing:**

* Loaded the dataset of text documents.
* Performed basic cleaning: lowercasing, punctuation removal, and tokenization.

**N-gram Extraction:**

1. **Unigrams**: Computed the top 20 most frequent single words.
2. **Bigrams**: Computed the top 20 most frequent two-word sequences.
3. **Trigrams**: Computed the top 20 most frequent three-word sequences.

**Visualization & Results:**

* Plotted bar charts for each n-gram level showing frequency counts.

  * **Unigrams:** Common words revealed: `the`, `and`, `to`, etc.
  * **Bigrams:** Frequent pairs such as `of the`, `in the`, `to the`.
  * **Trigrams:** Common triples like `one of the`, `as well as`.

**Key Insights:**

* Stopwords dominate unigram frequencies, suggesting more aggressive stopword filtering could focus on meaningful terms.
* Bigrams and trigrams highlight common phrases and potential collocations relevant for downstream tasks (e.g., phrase detection).

---

# Similarities between texts

**Objective:**

* Measure and visualize similarity between text documents using vector-based approaches.

**Data & Preprocessing:**

* Loaded multiple text documents from the shared folder.
* Cleaned and tokenized texts similar to the previous notebook.

**Feature Representations:**

1. **TF-IDF Vectors:** Transformed each document into a TF-IDF vector.
2. **Word Embeddings:** Generated document embeddings by averaging Word2Vec vectors.

**Similarity Calculations:**

* Computed pairwise cosine similarities between all document vectors.
* Presented similarity matrices for both TF-IDF and Word2Vec embeddings.

**Visualization & Results:**

* **Heatmaps** for TF-IDF and embedding-based similarities:

  * Highlighted clusters of highly similar documents.
* **Dendrogram** from hierarchical clustering on similarity distances:

  * Showed document grouping structure.

**Key Findings:**

* TF-IDF and embedding-based similarities largely agree on document clusters.
* Embeddings capture semantic relationships not obvious in keyword overlap (TF-IDF), e.g., synonyms or paraphrases.

---

**Conclusion:**

* The **N-grams.ipynb** notebook provides a foundation for understanding the most frequent terms and phrases in a corpus, useful for feature engineering.
* The **Similarities between texts.ipynb** notebook demonstrates methods to quantify and visualize document similarity, critical for tasks like clustering, retrieval, and recommendation.




