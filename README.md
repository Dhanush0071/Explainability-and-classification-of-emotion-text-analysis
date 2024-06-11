# Explainability-and-classification-of-emotion-text-analysis
leveraging the power of explainability , NLP techniques and machine learning techniques for a 12 class classification of emotion text

# Introduction :
Emotion detection using text involves the application of natural language processing (NLP) techniques to analyse and classify the emotional tone expressed in textual data. Focuses on building an explainable emotion detection system, which not only identifies emotions such as joy, sadness, anger, and fear from text but also provides transparent and interpretable explanations for its predictions. By leveraging advanced machine learning models and explainability tools like LIME (Local Interpretable Model-agnostic Explanations), this system aims to offer insights into the decision-making process of the model, enhancing trust and usability in various applications such as customer feedback analysis, social media monitoring, and mental health assessment. The balanced dataset used in this project, consisting of diverse emotional classes, ensures robust and fair evaluation of the model's performance across different emotional categories.

# Dataset used :
We have chosen 2 data sets to work on this problem namely 
1.	ISEAR
2.	Emotion sentiment dataset 

The ISEAR dataset, contains a total of 9621 values categorized into four emotion classes. The classes are distributed as follows: Anger with 2252 instances, Fear with 1701 instances, Joy with 1616 instances, and Sadness with 1533 instances. 
The Emotion sentiment dataset, contains a total of 840,000 values categorized into several emotion classes. The classes are distributed as follows: Fun with 2655 instances, Enthusiasm with 2481 instances, Surprise with 1912 instances, Empty with 1454 instances, Worry with 1242 instances, Boredom with 33 instances, Neutral with 180,137 instances, Love with 10,343 instances, Happiness with 7289 instances, Sadness with 4596 instances, Relief with 4416 instances, Hate with 4127 instances, and Anger with 3352 instances.
Both dataset is structured into a single DataFrame where each class has been uniformly sampled to contain 2000 instances.

# Preprocessing :
•	Lowercase Conversion - Creates text standardization ensuring no case sensitivity.  
•	Removal of:
URLs – Removes irrelevant noise from the text.
HTML tags –  Strips out non-informative formatting elements.
Email address - Eliminates irrelevant and potentially sensitive information.
Punctuation and special characters -  Simplifies text by removing non-essential characters.
Numbers -  Focuses on textual content by removing numerical noise.
Extra white spaces - Ensures consistency and readability.
Stop words - Reduces noise by removing common but uninformative words.

# Features generation:
TF-IDF with n-gram
•	Assigns numerical values to words based on their importance in individual documents and rarity across all documents.
•	Using n-grams from one to three words, it captures both single words and word sequences, enriching the data representation.
Bag of words with n-gram
•	Numerical representations of text by counting the occurrences of individual words and word sequences.
•	With n-grams ranging from one to three words, it captures both single words and sequences, offering a detailed view.

# Evaluations:
Here's a brief explanation of the evaluation models used:
1.	XGBoost Classifier:
•	XGBoost (Extreme Gradient Boosting) is an optimized gradient-boosting algorithm.
•	It builds an ensemble of decision trees in a sequential manner, where each tree corrects the errors of the previous ones. It uses gradient descent to minimize the loss function.
2.	SVM with Linear Kernel:
•	SVM with a linear kernel is used for linear classification tasks.
•	It finds the best linear hyperplane that separates the classes by maximizing the margin between the closest points of the classes.
3.	SVM with RBF Kernel:
•	SVM with a Radial Basis Function (RBF) kernel is used for non-linear classification tasks.
•	It transforms the input space into a higher-dimensional space using the RBF kernel, allowing the algorithm to find a linear separation in this transformed space.
4.	Random Forest Classifier:
•	Random Forest is an ensemble learning method used for classification.
•	It builds multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees.
5.	Decision Tree:
•	A Decision Tree is a simple, interpretable model used for classification.
•	It splits the data into branches based on feature values, making decisions at each node to classify the data.
6.	Logistic Regression:
•	Logistic Regression is used for binary and multi-class classification.
•	It models the probability of the default class using a logistic function. It estimates the relationship between the dependent variable and one or more independent variables.

# Explainability:
Local Interpretable Model-agnostic Explanations (LIME):
It's a technique used in machine learning for explaining the predictions of complex models.Instead of providing a global understanding of the model on the entire dataset, LIME focuses on explaining the model’s prediction for individual instances.
LIME helps us understand why a machine learning model, makes certain predictions. It does this by showing us which parts of the input data were most important in making that prediction, helping us trust the program's decisions and potentially improve its accuracy.
The result contains three main pieces of information from left to right: 
•	the model’s predictions,
•	features contributions, 
•	the actual value for each feature.

# Neural tangent kernel:
•	The neural tangent kernel (NTK) is a kernel that describes the evolution of deep artificial neural networks during their training by gradient descent.
•	NTK helps us understand how neural networks learn and generalize, allowing us to use their knowledge in simpler and more reliable machine learning methods.
•	The average accuracy for 2000 random texts for 100 trails is 92.1.

