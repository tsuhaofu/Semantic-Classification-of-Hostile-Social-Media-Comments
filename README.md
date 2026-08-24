# Semantic Classification of Hostile Social Media Comments

This project aims to detect cyberbullying in social media comments using machine learning techniques. The model classifies comments based on their content into potentially hostile or non-hostile categories.

## Introduction
Cyberbullying is a significant issue in digital communication, involving the dissemination of harmful or mean content about others via various digital means. This project focuses on identifying such hostile comments on social media platforms to limit the harm caused by cyberbullies. A dataset of 20,001 Twitter messages, each labeled for online hostility, was used to train and evaluate the classification model.

## Running it
```bash
pip install pandas numpy matplotlib seaborn pillow requests wordcloud nltk gensim scikit-learn pyLDAvis
cd Semantic-Classification-of-Hostile-Social-Media-Comments
python Semantic-Classification.py
```
The `cd` matters. The script reads `Dataset for Detection of Cyber-Trolls.json` by
bare filename, so it resolves against the **working directory**, not against the
script's own location — run it from anywhere else and it will not find its data.
The file is in this repository, so nothing needs downloading. `MODEL_DIR` (default
`.`, where `lda.model` is written and re-read) follows the same rule.

Two things reach the network at run time: the four `nltk.download(...)` calls in the
import block, and the word-cloud mask image. Only the mask is optional —
`WORDCLOUD_MASK_URL` overrides it and the plot renders unmasked if the fetch fails.

Written against gensim 3.x. The only hard incompatibility with gensim 4+ — an unused
`gensim.models.wrappers` import, a module removed in 4.0 — has been dropped. Two other
removed APIs survive in `plot_10_most_common_words`, which is never called:
`CountVectorizer.get_feature_names()` (removed in scikit-learn 1.2) and positional
`sns.barplot(x, y)` (removed in seaborn 0.12). Treat other version sensitivities as
untested.

Expect it to take hours, not minutes: the topic-count search at line 177 fits 54 LDA
models with `passes=10` over ~19,800 documents, each followed by a `c_v` coherence
computation.

Where the numbers below come from: `Semantic-Classification.py` is a flattened script
rather than a notebook, so it stores no outputs. Every figure quoted here is the one
recorded in `Semantic-Classification.pdf`, the report written from the original run.
The script reproduces the same experiments, but SVM fits and the LDA sweep are not
pinned tightly enough for the decimals to match — read them as the result to expect,
not as a checksum.

## Credits
The data loading, cleaning and lemmatisation pipeline is my own, written against this
dataset's schema. The modelling sections follow well-known public walkthroughs closely:

- LDA topic modelling — [Topic Modeling with Gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/), Machine Learning Plus
- SVM classification — [Support Vector Machines with Scikit-learn](https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python), DataCamp
- Word cloud generation — the standard `wordcloud` masked-image walkthrough

## Data
The [Tweets dataset](https://www.kaggle.com/datasets/dataturks/dataset-for-detection-of-cybertrolls) includes the following variables:
- **Content**: The text content of the message.
- **Label**: Indicates whether the message is online hostile (1) or not (0).

## Data Cleaning
Due to the chaotic nature of online writing habits, several data cleaning steps were necessary:
1. Removal of punctuation.
2. Removal of numbers.
3. Conversion to lowercase.
4. Removal of extra spaces.
5. Removal of stop words.
6. Lemmatization of words.

## Exploratory Data Analysis
### Word Cloud and Frequency Analysis
High-frequency words like "hate" and "damn" appeared across all messages, indicating their usage might be contextual rather than inherently negative.
### Label Proportions
The dataset showed some imbalance with a 3:2 ratio between non-hostile (label 0) and hostile (label 1) messages.

### Label-Specific Observations
- **Label 0 (Non-hostile)**: High-frequency words were similar to the overall corpus, but lower-frequency words tended to be positive or neutral.
- **Label 1 (Hostile)**: High-frequency words did not differ much from the entire corpus, but lower-frequency words were more negative.

## Methods
### Latent Dirichlet Allocation (LDA)
Unsupervised learning was applied to divide the corpus into topics. Each message was represented by a feature vector of its topic composition. The coherence score suggested 31 topics as optimal.

### Support Vector Machine (SVM)
SVM was used for classification. The 31-dimensional LDA topic vectors were split into an
80% training set and a 20% testing set (`random_state=424`), and four kernels compared —
the loop over `['linear', 'poly', 'rbf', 'sigmoid']` in `Semantic-Classification.py`
prints this table:

1. **Linear Kernel**: 
   - Training: Accuracy = 60.576%, Precision = 73.846%, Recall = 7.631%
   - Testing: Accuracy = 61.112%, Precision = 81.818%, Recall = 0.388%

2. **Polynomial Kernel**: 
   - Training: Accuracy = 71.107%, Precision = 68.443%, Recall = 50.238%
   - Testing: Accuracy = 69.395%, Precision = 63.693%, Recall = 47.279%

3. **Gaussian Kernel**: 
   - Training: Accuracy = 72.726%, Precision = 71.653%, Recall = 51.558%
   - Testing: Accuracy = 69.572%, Precision = 64.114%, Recall = 47.213%

4. **Sigmoid Kernel**: 
   - Training: Accuracy = 49.981%, Precision = 32.948%, Recall = 25.358%
   - Testing: Accuracy = 48.892%, Precision = 29.479%, Recall = 23.738%

The Gaussian kernel showed the best overall performance.

### Principle Component Analysis (PCA)
PCA was used for dimensionality reduction with the target of explaining over 80% of the variance with the principal components. Nineteen components were selected.

- **Post-PCA Gaussian Kernel**: 
  - Training: Accuracy = 73.003%, Precision = 71.015%, Recall = 53.831%
  - Testing: Accuracy = 69.370%, Precision = 63.239%, Recall = 48.393%

## Results
The best results were obtained using the Gaussian kernel SVM model after PCA, though the performance was below expectations. Future work may involve adjusting the number of topics in LDA or finding a more suitable corpus for analysis.

## Conclusion
This study aimed to classify social media comments into hostile and non-hostile categories using machine learning. The accuracy reached about 70%, indicating room for improvement. Potential reasons for the suboptimal performance include not finding the ideal number of topics for LDA and the chaotic nature of the corpus, which may not be suitable for LDA analysis.
