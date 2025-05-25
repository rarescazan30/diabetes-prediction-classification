
**Diabetes Classification Model**

**Approach**

This project is based on a publicly available dataset from Kaggle, titled *diabetes-prediction-dataset*, which includes information on over 100,000 patients. The dataset contains features such as age, gender, body mass index (BMI), hypertension status, cardiovascular conditions, smoking history, HbA1c level, and blood glucose level.

Additional features were engineered to enhance model performance, including age bins (<30, 30–50, 50–70, and >70 years), a physical activity level (calculated based on age, with a 10% noise factor), and a binary obesity indicator (set to 1 if the BMI exceeds 30). These features were designed to improve model accuracy by incorporating domain-informed attributes.

The smoking history data, originally represented as strings, was converted using one-hot encoding (via the `get_dummies` function), categorizing patients into: never smoked, formerly smoked, does not currently smoke, and currently smokes.


**Exploratory Data Analysis**

**a) Missing Values**

Approximately 35.82% of the entries for smoking history were labeled as “No Info” (35,816 individuals). These missing values were replaced with the mode, i.e., the most frequent entry. All other fields in the dataset were complete, as confirmed through descriptive statistics.

**b) Descriptive Statistics**

Descriptive statistics were computed for both the training and test sets. Features such as age, blood glucose level, and physical activity level exhibit greater variability compared to others.

The similarity between the distributions of the training and test datasets reflects the expected outcome of a randomly generated 80-20 split. The training set was balanced using SMOTE (Synthetic Minority Over-sampling Technique) to address the initially low prevalence of diabetes cases, which could negatively impact the performance of the K-Nearest Neighbors algorithm.

**c) Variable Distribution Analysis**

Patients aged between 50 and 70 are overrepresented in the dataset, which is beneficial for training since diabetes diagnoses are more prevalent in this age group.

The distribution of categorical and numerical features also reflects the effect of synthetic data balancing through SMOTE in the training set.

The imputation of missing smoking history values using the mode disproportionately increased the "never smoked" category. While this could skew results if smoking history were a primary predictor, experiments showed that it did not significantly affect outcomes. This imputation method was chosen for consistency with laboratory practices.

**d) Outlier Detection**

No significant outliers were detected for age, indicating well-regulated data entry procedures during dataset creation.

Outliers were observed in the blood glucose level feature, particularly in the test set. Since elevated glucose levels strongly correlate with diabetes, these data points likely represent valid high-risk cases.

Extreme BMI values above 60, likely indicative of input errors, were removed from the dataset to improve model reliability.

Outliers in HbA1c were within plausible ranges for severe diabetes and thus retained. The correlation matrix later confirms this feature’s relevance to diabetes prediction.

Physical activity values showed no extreme outliers, though a broader distribution was observed in the training set. This variation is likely due to the age-based feature generation.

**e) Correlation Matrices**

The correlation analysis highlights several key relationships:

* A strong correlation exists between physical activity, diabetes, and age—expected due to the engineered nature of the physical activity feature.
* The is\_obese feature, derived from BMI, correlates accordingly with BMI.
* HbA1c level and blood glucose level both show strong positive correlations with diabetes, confirming their predictive power.
* Minor correlations between features such as HbA1c and physical activity, or age and diabetes, are attributed to feature engineering and do not significantly impair model performance.

**f) Target Variable Relationships**

The relationship between age and diabetes risk is clearly observed: older individuals are more likely to be diagnosed.

Blood glucose level is strongly associated with the diabetes label, reinforcing its importance as a predictive factor.

BMI shows a modest association with diabetes. The is\_obese feature is expected to enhance model performance by emphasizing this correlation.

HbA1c levels also display a strong positive correlation with diabetes, as observed previously in the correlation matrix.

As expected, individuals with higher levels of physical activity are less likely to have diabetes. This inverse relationship aligns with how the feature was constructed.


**Model Training and Evaluation**

The data pipeline begins with loading the dataset using `get_dataset_and_path.py`, followed by preprocessing in `dataset_manip.py`. Preprocessing steps include converting categorical data to numerical formats, handling missing values, feature engineering, outlier removal, data splitting, normalization, and model training. The trained model and scaler are saved for use in the graphical user interface.

If desired, confusion matrices and error analysis plots are generated for further evaluation. The `analyze_and_get_graphs.py` script handles visualization, while the graphical interface is launched via `graphic_interface.py`.

> *Note: Only the processed train and test dataframes are included in the archive. To run the full pipeline, the original dataset must be downloaded from Kaggle and placed in the project directory.*

The confusion matrix shows that the model more frequently misclassifies diabetes cases. Despite this, the model achieves a high overall accuracy of 97%, primarily due to the large dataset and the relatively low number of positive diabetes diagnoses. The model achieves high precision (0.99) for negative cases and lower precision (0.81) for positive cases.

**Classification Report**

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Not Diabetes | 0.99      | 0.98   | 0.99     | 18,248  |
| Diabetes     | 0.81      | 0.90   | 0.86     | 1,693   |
| **Accuracy** | -         | -      | **0.97** | 19,977  |
| Macro Avg    | 0.90      | 0.94   | 0.92     | 19,977  |
| Weighted Avg | 0.98      | 0.97   | 0.97     | 19,977  |


**Error Analysis**

The model demonstrates difficulty classifying common values of glucose, HbA1c, and BMI—likely because these fall within normal ranges, providing little diagnostic indication.

In the case of age, the engineered physical activity feature may have inadvertently biased the model toward predicting no diabetes in younger individuals. This influence lacks a strong scientific basis and reflects the trade-offs of feature engineering.


**Conclusion**

The model achieves a high overall accuracy (97%) in predicting diabetes. However, it underperforms when identifying positive cases, reflected in the lower precision for diabetes diagnoses. The performance reflects both the quality of the dataset and the influence of engineered features, such as `is_obese` and `physical_activity`. These additional features improve model performance but may introduce biases if not carefully validated.

