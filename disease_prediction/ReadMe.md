This application is designed to predict diseases based on user-input symptoms using machine learning algorithms. The GUI allows users to select symptoms from a predefined list and receive a disease prediction based on those inputs.

Features
User-friendly graphical interface built with Tkinter.
Supports multiple machine learning models: Decision Tree, Random Forest, and Naive Bayes.
Disease predictions based on a dataset of symptoms and their corresponding diseases.

Requirements
To run this application, you need the following Python packages:
Tkinter (for GUI)
NumPy (for numerical operations)
Pandas (for data manipulation)
scikit-learn (for machine learning algorithms)

File Structure
The main code is contained within a single Python file. The application reads training and testing data from CSV files named Training.csv and Testing.csv. Ensure these files are present in the same directory as the script.

Data Preparation
The symptoms and diseases are defined in lists. The application reads the training data from Training.csv, replaces disease names with numerical labels, and prepares the feature set x and target variable y.

Machine Learning Models
Three machine learning models are implemented:
Decision Tree,Random Forest,Naive Bayes
Each model is trained on the training dataset, and predictions are made based on user-selected symptoms.

GUI Implementation
The GUI is built using Tkinter. It includes:
Labels for user input.
Dropdown menus for symptom selection.
Buttons to trigger predictions using different models.
Text fields to display results.

Running the Application
To run the application, execute the following command in your terminal:
bash: python app.py

Usage Instructions
Enter the patient's name in the provided entry field.
Select up to five symptoms from the dropdown menus.
Click on one of the prediction buttons (Decision Tree, Random Forest, or Naive Bayes).
The predicted disease will be displayed in the corresponding text field.


Conclusion
This application serves as a basic tool for disease prediction based on symptoms using machine learning techniques. It can be expanded further by integrating more complex models or additional features such as user authentication or data visualization.