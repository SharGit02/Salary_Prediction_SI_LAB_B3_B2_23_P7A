# Lab 7: Salary Prediction Using Simple Linear Regression

## Aim
To develop and analyze a Simple Linear Regression model using statistical programming tools to predict a dependent variable based on a single independent variable and evaluate the model's performance.

## Project Overview
This project predicts annual salary based on years of professional experience using Simple Linear Regression. The model is implemented in Python using Google Colab, evaluated using regression metrics, and integrated into a Streamlit web application for interactive salary prediction.

The project demonstrates data exploration, visualization, model training, prediction, and performance evaluation.

## Dataset Information

The dataset contains job and salary-related information for different roles.

| Property | Details |
|---|---|
| Total records | 1,500 |
| Total columns | 25 |
| Independent variable | `years_of_experience` |
| Dependent variable | `annual_salary_usd` |
| Missing values | 0 |
| Duplicate rows | Check dataset analysis output |
| Train-test split | 80:20 |
| Training samples | 1,200 |
| Testing samples | 300 |

### Selected Features

- **Independent variable (X):** Years of Experience
- **Dependent variable (Y):** Annual Salary in USD

Only one independent variable is used to satisfy the Simple Linear Regression requirement.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Methodology

1. **Dataset Loading:** Load the salary dataset into a Pandas DataFrame.
2. **Data Exploration:** Examine the dataset shape, column names, data types, statistical summary, missing values, and duplicate records.
3. **Data Preprocessing:** Check missing values in the selected variables. No missing values were found in the provided dataset.
4. **Feature Selection:** Select `years_of_experience` as the independent variable and `annual_salary_usd` as the dependent variable.
5. **Exploratory Data Analysis:** Plot a scatter plot, examine correlation, and visualize the salary trend.
6. **Data Splitting:** Divide the data into 80% training data and 20% testing data using `random_state=42`.
7. **Model Training:** Train a Simple Linear Regression model using Scikit-learn.
8. **Prediction:** Predict salaries for the testing dataset.
9. **Evaluation:** Calculate MAE, MSE, RMSE, and R² score.
10. **Visualization:** Compare actual and predicted salaries and plot the regression line.
11. **Deployment:** Use Streamlit to provide an interface for entering years of experience and obtaining a predicted salary.

## Simple Linear Regression Equation

The general equation is:

**Y = b₀ + b₁X**

Where:
- **Y:** Predicted annual salary
- **X:** Years of experience
- **b₀:** Intercept
- **b₁:** Regression coefficient

### Trained Model Equation

Based on the Colab training output:

**Predicted Salary = 178,233.49 + (2,779.98 × Years of Experience)**

The coefficient indicates that the fitted model estimates an increase of approximately **$2,779.98** in annual salary for each additional year of experience, on average, within the fitted linear relationship. This does not establish a causal relationship.

## Model Evaluation Results

The following results were obtained on the 300-sample test set.

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | $54,663.13 |
| Mean Squared Error (MSE) | 4,212,230,768.12 |
| Root Mean Squared Error (RMSE) | $64,901.70 |
| R² Score | -0.0276 |
| R² Percentage | -2.76% |

### Interpretation

- **MAE:** Predictions differ from actual salaries by approximately $54,663 on average in absolute terms.
- **MSE:** Measures the average squared prediction error, penalizing larger errors more heavily.
- **RMSE:** Indicates a typical error scale of approximately $64,902, expressed in the same units as salary.
- **R² Score:** The negative test R² indicates that the model performs worse than a baseline that predicts the training-set mean salary for every test sample.

The results suggest that years of experience alone does not explain enough of the salary variation in this dataset. Salary may also depend on job category, seniority, industry, location, company size, and other factors.

## Visualizations

The Colab notebook includes the following visualizations:

1. **Experience vs Annual Salary:** A scatter plot showing the distribution of salary values across experience levels.
2. **Linear Correlation:** A chart showing the correlation coefficient between years of experience and annual salary.
3. **Salary Growth Trend:** A regression plot showing the fitted linear trend.
4. **Actual vs Predicted Salary:** A scatter plot comparing actual and predicted values against the perfect-prediction reference line.
5. **Feature Correlation Analysis:** A ranking of numeric features by their correlation with annual salary.
6. **Average Salary by Experience:** A table showing mean annual salary for each experience level.

## Streamlit Application

The Streamlit application provides a user interface for salary prediction based on years of experience.

### Application Features
- Input field for years of experience.
- Predicted annual salary displayed in USD.
- Simple and interactive interface.
- Integration with the trained regression model.

### Live Demo

[Open the Salary Prediction Streamlit App](https://salary-prediction-academic-project.streamlit.app/)

## Project Structure

```text
Salary_Prediction_SI_LAB_B3_B2_23_P7A/
├── app.py
├── salary_data.csv
├── salary_model.pkl
├── requirements.txt
├── README.md
├── .streamlit/
├── Lab7_Colab.pdf
└── Lab7_Streamlit_App.pdf
```

The PDF filenames above are suggested names. Replace them with the actual names of the uploaded files.

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SharGit02/Salary_Prediction_SI_LAB_B3_B2_23_P7A.git
```

### 2. Enter the project directory

```bash
cd Salary_Prediction_SI_LAB_B3_B2_23_P7A
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

## Conclusion

A Simple Linear Regression model was developed to predict annual salary using years of experience as the only independent variable. The model was trained on 1,200 records and evaluated on 300 test records.

The test R² score of -0.0276 shows that the model has limited predictive performance for this dataset when using experience alone. The practical demonstrates the complete regression workflow, including data exploration, visualization, training, prediction, evaluation, and deployment through Streamlit.

## Lab Deliverables

- **Colab PDF:** Contains the dataset analysis, visualizations, model implementation, predictions, and evaluation.
- **Streamlit App PDF:** Contains the application implementation and relevant screenshots.
- **GitHub Repository:** Contains the source code, model, dataset, dependencies, and documentation.

## Author

**Sharvari Hiwarkhedkar** 

Roll No: 23: Class and Batch: B3_B2

CSE (Artificial Intelligence and Data Science)  
Ramdeobaba University, Nagpur

## Repository

[Salary Prediction – SI Lab](https://github.com/SharGit02/Salary_Prediction_SI_LAB_B3_B2_23_P7A)
[GoogleColab Link](https://colab.research.google.com/drive/1oqDx8fHnaArh_gcHx7McfW1mApp0kaMd?usp=sharing)
