Project Overview
The goal of this project is to build a predictive model that can assess the creditworthiness of loan applicants based on various features such as income, loan amount, and credit history. This can help financial institutions make informed lending decisions.

Installation
To set up the project, ensure you have Python installed on your machine. You can then install the required libraries using pip:
bash:- pip install numpy pandas scikit-learn

Usage
Clone the repository:
bash:- git clone <repository-url> 
       cd CreditScorePrediction

Prepare your dataset:
Place the loan_train.csv file in the project directory.

Run the Jupyter Notebook:
Launch Jupyter Notebook and open CreditScore.ipynb to start analyzing the data and training the model.

Modeling
The model is built using 
model_1 = RandomForestClassifier()
model_2 = GradientBoostingClassifier()
model_3 = SVC()
model_4 = LogisticRegression()
model_5 = KNeighborsClassifier()
model_6 = GaussianNB()
model_7 = DecisionTreeClassifier().
The following steps are performed:
    Data loading and preprocessing.
    Exploratory data analysis to understand feature distributions.
    Model training .


Results
The results will be displayed within the Jupyter Notebook after running the model training cell. The performance metrics will help evaluate how well the model predicts loan approval status based on applicant features
    