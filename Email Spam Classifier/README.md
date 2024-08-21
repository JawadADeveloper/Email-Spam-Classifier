# Email Spam Classifier
An Email Spam Classifier built using machine learning techniques to identify and filter out spam emails. This project leverages text processing and classification algorithms to determine whether an email is spam or not based on its content.

# Features
Text Preprocessing: Convert text to lowercase, remove punctuation and stopwords, and apply stemming.
Feature Extraction: Utilize TF-IDF (Term Frequency-Inverse Document Frequency) to transform text into numerical features.
Model Training: Multiple models like Naive Bayes, Logistic Regression, and Support Vector Machines are trained for spam detection.
Real-time Prediction: A web-based interface built with Streamlit for users to input email text and get instant spam classification.
# Installation
## 1. Clone the Repository
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
## 2.Install the Required Packages
pip install -r requirements.txt
## 3.Download NLTK Data
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# Usage
## 1.Train the Model (Optional)
If you want to retrain the model:
python train_model.py
## 2.Run the Streamlit App  
streamlit run app.py
## 3.Input Email Text

Open the web app in your browser, enter the text of the email, and the classifier will predict if it's spam or not.
# File Structure
1.app.py: Streamlit application for real-time spam prediction.
2.train_model.py: Script to train the model from scratch.
3.vectorizer.pkl: Pre-trained TF-IDF vectorizer.
4.model.pkl: Pre-trained classification model.
5.requirements.txt: List of required Python packages.
6.README.md: Project documentation.
# Model Overview
The classifier uses the following techniques:

Text Preprocessing: Tokenization, stemming, stopword removal.
Feature Extraction: TF-IDF to capture the importance of words.
Classification: Naive Bayes, Logistic Regression, SVM, and other models for classifying emails.
Example

 
input_sms = "Congratulations! You've won a free ticket. Click here to claim."
Prediction: Spam
# Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

# License
This project is licensed under the MIT License.

# Contact
If you have any questions or suggestions, feel free to reach out!
