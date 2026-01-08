# AutoJudge: Predicting Programming Problem Difficulty

AutoJudge is a machine learning–based system that predicts the difficulty of programming problems using only textual information. The system performs two tasks:
1. Classification of problems into Easy, Medium, or Hard.
2. Regression to predict a numerical difficulty score.

The predictions are based on the problem description, input description, and output description provided by the user. The project includes a trained machine learning pipeline and a simple web interface for interactive predictions.

## Project Structure

```
AutoJudge/
├── WebUI_code.py # Streamlit web application for difficulty prediction
├── rf_classifier.pkl # Trained Random Forest model for difficulty classification (Easy/Medium/Hard)
├── catboost_regressor.cbm # Trained CatBoost model for numerical difficulty score prediction
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer used for text feature extraction
├── requirements.txt # Python dependencies required to run the project locally
├── report.pdf # Detailed project report (4–8 pages) with analysis and results
├── Auto_judge.ipynb # Complete source code notebook
├── auto_judge_evaluation.py # Complete source code for evaluation as .py file
└── README.md # Project documentation and setup instructions

```

## Dataset Used

The project uses a dataset of programming problems collected from online coding platforms. Each problem is already labeled with both a difficulty class and a numerical difficulty score, making it suitable for supervised learning.

Each dataset entry contains the following fields:
- Title of the problem  
- Problem description  
- Input description  
- Output description  
- Sample input/output examples  
- Difficulty class (Easy, Medium, Hard)  
- Numerical difficulty score  

The dataset contains a total of 4,112 programming problems and was provided in JSONL format. The dataset link mentioned in the project description was used as a reference, and an equivalent labeled dataset was used for training and evaluation.

## Approach and Models Used

The project follows a structured machine learning pipeline consisting of exploratory data analysis, data preprocessing, feature extraction, model training, and evaluation.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis was performed to understand the dataset before model development. The distribution of difficulty classes (Easy, Medium, Hard) was analyzed to identify class imbalance, and it was observed that Hard problems were more frequent than Easy and Medium problems. 

For the numerical difficulty score, summary statistics were examined to understand the range and spread of difficulty values. These observations helped guide model evaluation and interpretation of results.

### Data Preprocessing
Textual fields were cleaned and prepared by handling missing values, removing duplicates, and ensuring consistent formatting. Relevant textual information was combined to form a single text representation for each programming problem.

### Feature Extraction
The combined text was converted into numerical features using the Term Frequency–Inverse Document Frequency (TF-IDF) technique. TF-IDF was chosen because it effectively represents the importance of words while reducing the influence of commonly occurring terms.

### Classification Model
Multiple classification models were experimented with, including Logistic Regression, Support Vector Machine (SVM), and Random Forest. Based on empirical evaluation, the **Random Forest classifier** achieved the highest accuracy and better class-wise performance and was selected as the final classification model to predict Easy, Medium, or Hard difficulty levels.

### Regression Model
For predicting the numerical difficulty score, several regression models were evaluated, including Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, and CatBoost Regressor. The **CatBoost Regressor** achieved the lowest error values and was selected as the final regression model.

## Evaluation Metrics and Results

The performance of the models was evaluated separately for classification and regression tasks using standard evaluation metrics.

### Classification Evaluation
For the difficulty classification task, model performance was evaluated using **classification accuracy** and **confusion matrices**. These metrics helped assess overall prediction correctness as well as class-wise performance for Easy, Medium, and Hard categories.

Among the evaluated classifiers, the Random Forest model achieved the highest accuracy of approximately **53%** and demonstrated improved identification of Hard problems compared to other models. 
Compared to other evaluated classifiers, Random Forest demonstrated stronger diagonal dominance for the Hard class which is the most important class and must not be misclassified and fewer extreme misclassifications. Most errors occurred between adjacent classes (Easy–Medium and Medium–Hard), which is expected due to the subjective nature of difficulty labeling. These observations supported the selection of Random Forest as the final classification model.

### Regression Evaluation
For the numerical difficulty score prediction, regression models were evaluated using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**. These metrics quantify the average prediction error and penalize larger deviations more heavily.

The CatBoost Regressor achieved the lowest MAE of approximately **1.68** and RMSE of approximately **2.03** among the tested models and was selected as the final regression model.

## Web Interface

A simple web interface was developed using Streamlit to allow users to interact with the trained models.

The web interface provides three input text fields:
- Problem Description  
- Input Description  
- Output Description  

Users can enter the textual details of a programming problem and submit the input for prediction. Upon submission, the system displays:
- The predicted difficulty class (Easy, Medium, or Hard)
- The predicted numerical difficulty score

The web interface uses the saved Random Forest classification model and CatBoost regression model to generate predictions in real time. No external hosting is required, and the application runs locally.

## Steps to Run the Project Locally

### Note:
This project has been tested with Python versions 3.9 to 3.11.  
Using very new Python versions (e.g., 3.13 or above) on Windows may cause dependency build issues for some libraries such as CatBoost.


Follow the steps below to run the project on a local machine.

1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/gopal1937/AutoJudge
   cd AutoJudge

2. Install the required dependencies:
   pip install -r requirements.txt

3. Ensure the following files are present in the project directory:

   tfidf_vectorizer.pkl

   rf_classifier.pkl

   catboost_regressor.cbm

   WebUI_code.py

4. Run the web application:
   streamlit run WebUI_code.py

5. Open the displayed local URL in a web browser to access the web interface.

The application runs entirely on the local machine and does not require any external hosting.

## Demo Video

A short demo video (2–3 minutes) demonstrating the project is provided at the link below.

The demo video covers:
- A brief explanation of the project objective
- Overview of the machine learning approach used
- Live demonstration of the web interface with sample predictions

Demo Video Link:  
https://drive.google.com/file/d/11xKBnJ7AzJxI7XzCJj20FzyS3BygPGdE/view?usp=sharing

## Author

**Name:** Gopal Datt Sharma  

**Enrollment No:** 23115047

**Branch & Year:** Electrical IIIrd

**Project Title:** AutoJudge – Predicting Programming Problem Difficulty  

This project was developed as part of an academic assignment and demonstrates the use of machine learning techniques for text-based difficulty prediction of programming problems.














