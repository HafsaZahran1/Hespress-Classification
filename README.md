# Hespress-Classification
This repository contains the implementation of a machine learning-based text classifier for a multi-class text classification task. The goal is to classify news stories into different topics, such as art and culture, economy, faits-divers, marocains-du-monde, medias, politique, regions, societe, sport, and tamazight.

## Dataset 
This repository contains a Python script for analyzing Moroccan news articles from Hespress using various data analysis techniques. The script provides insights into the class distribution, top frequent n-grams, and lengths of examples in words and letters.
Dataset

The dataset consists of Moroccan news articles from the Kaggle, collected and saved in CSV format. The 'data' folder within the repository holds the CSV files.

The Kaggle dataset link: https://www.kaggle.com/datasets/tariqmassaoudi/hespress

## A brief description of the whole training process for the text classification:

1. **Data Loading and Preprocessing:**
   - Load the news stories dataset from CSV files.
   - Remove unwanted columns.

2. **Data Splitting:**
   - Split the dataset into text input (X) and target variable (y).
   - Use the last 20% of each file as the test set to ensure a diverse distribution of topics in the test data.

3. **Feature Extraction (TF-IDF):**
   - Convert the text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
   - Use the TfidfVectorizer from scikit-learn to transform the text data into TF-IDF features.
   - Limit the number of features to 5000 (can be adjusted based on available resources).

4. **Model Selection and Training:**
   - Choose a machine learning model suitable for multi-class text classification, such as Multinomial Naive Bayes.
   - Initialize the selected model.
   - Train the model using the TF-IDF features (X_train_tfidf) and the corresponding target labels (y_train).

5. **Model Evaluation:**
   - Make predictions on the test set (X_test_tfidf) using the trained classifier.
   - Calculate classification metrics, including precision, recall, F1-score, and accuracy, for each class and the overall test data.
   - Print the classification report, which summarizes the performance metrics for each class.

6. **Performance Analysis and Enhancements:**
   - Analyze the classification results to identify classes with high and low performance.
   - Suggest potential enhancements to improve overall performance, such as using data augmentation, experimenting with different text representations, and trying more advanced machine learning models.


## The performance of the Naive Bayes classifier on the test set. 
![class](https://github.com/HafsaZahran1/Hespress-Classification/assets/73903183/a425a53b-d796-467d-a1c2-bac9e3b20cca)

Let's understand the meaning of each metric:

1. **Precision:**
   - Precision measures the percentage of true positive predictions among the total positive predictions for a specific class. It indicates how many of the predicted instances of a class are actually correct. A high precision value means that the classifier has a low false positive rate for that class.

2. **Recall:**
   - Recall, also known as sensitivity or true positive rate, measures the percentage of true positive predictions among the total actual positive instances of a specific class. It indicates how well the classifier identifies all the positive instances of a class. A high recall value means that the classifier has a low false negative rate for that class.

3. **F1-score:**
   - The F1-score is the harmonic mean of precision and recall. It is used to balance the trade-off between precision and recall. The F1-score is a suitable metric when there is an uneven class distribution. Higher F1-score indicates a good balance between precision and recall.

4. **Accuracy:**
    - Accuracy is the overall correctness of the classifier, and it measures the percentage of correct predictions among all predictions. It is a general metric that can be useful when the classes are balanced. However, it might not be informative in cases of imbalanced classes.

## The results for each class and the overall test data:

1. The classifier performs well for "faits-divers," "tamazight," and "sport" classes, with high F1-scores (0.61, 0.78, and 0.59, respectively). This means the classifier has relatively balanced precision and recall for these classes, indicating good performance.

2. Classes like "economie," "marocains-du-monde," "politique," and "regions" have moderate F1-scores (ranging from 0.45 to 0.53). These classes could be improved with further tuning or using more advanced classifiers.

3. The "medias" class has the lowest F1-score (0.37), indicating that the classifier struggles to balance precision and recall for this class. This class may require more data or feature engineering to improve its performance.

4. The "societe" class has a low F1-score (0.20) and relatively low precision and recall, suggesting that the classifier performs poorly for this class.

5. The overall accuracy of the classifier on the test data is 0.494, which indicates that the classifier correctly predicts the class label for about 49.4% of the instances in the test set. However, considering the imbalanced class distribution, accuracy alone might not be the most informative metric in this case.

## Enhancements to achieve better results:

1. Data Augmentation: Since the dataset may be limited, one approach to improve performance is to use data augmentation techniques to create additional training data. This can involve techniques like synonym replacement, back-translation, or other text-based augmentations.

2. Advanced Models: Try using more advanced machine learning models such as support vector machines, random forests, or deep learning models like recurrent neural networks (RNNs) or transformers. Different models may capture different patterns in the data and lead to better results.

3. Hyperparameter Tuning: Perform hyperparameter tuning for the chosen classifier to find the best combination of parameters for the given task.

4. Incorporate Contextual Information: If possible, consider using metadata or contextual information related to the stories to improve classification performance.

The success of these enhancements depends on the data available, and experimentation is key to finding the best approach for the specific task at hand.
