# LoanTap – Personal Loan Default Prediction

## 1. Business Problem

LoanTap aims to develop a machine learning model to assess the creditworthiness of individuals applying for personal loans.

The objective is to predict whether a borrower is likely to fully repay the loan or default (Charged Off), based on their financial, employment and credit-related attributes.

This is a binary classification problem.

The model should help LoanTap:
- Identify potentially risky borrowers.
- Reduce financial losses arising from loan defaults.
- Avoid rejecting creditworthy applicants.
- Support informed lending and risk-management decisions.

The model is intended to predict the likelihood of loan default at the underwriting stage, before loan approval and issuance. Only information available at or before this decision point should be considered as model input.

## 2. Initial Dataset Exploration

The dataset contains 396,030 records and 27 attributes.

The target variable is `loan_status`, which contains two categories:

- Fully Paid: 318,357 records (80.39%).
- Charged Off: 77,673 records (19.61%).

The target variable is imbalanced, with approximately 80% of observations belonging to the Fully Paid category.

Six features contain missing values. `mort_acc` has the highest missing percentage at 9.54%.

The dataset contains both numerical and categorical variables. Some categorical variables, particularly `emp_title`, `title` and `address`, have high cardinality and require further investigation before encoding.

## 3. Initial Observations

1. The dataset contains a substantial number of historical loan records.
2. Approximately one in five borrowers has a Charged Off loan status.
3. Missing values need to be investigated and treated before model building.
4. Class imbalance makes recall, precision and ROC-AUC important evaluation metrics.
5. Date-related and high-cardinality variables require appropriate feature engineering and preprocessing.

## 4. Univariate Analysis – Numerical Variables

### Continuous Variables

- Annual income exhibits significant positive skewness, indicating that a small proportion of borrowers report exceptionally high incomes.
- DTI and revolving balance also exhibit substantial positive skewness.
- Several variables contain potential outliers, particularly annual income, revolving balance, installment and DTI.
- These observations require further investigation before deciding on outlier treatment.

### Count-based Variables

- The majority of borrowers have zero public derogatory records.
- Public records are concentrated around zero, with fewer borrowers having multiple derogatory records.
- This variable may provide useful information for assessing borrower creditworthiness.

### Key Observations

The numerical variables exhibit different distributions and ranges. Several financial attributes are positively skewed and contain extreme observations. Median-based statistics may therefore be more representative than mean-based statistics for certain variables.

Outliers should not be removed blindly because extreme financial values may represent genuine borrower characteristics.

## 5. Univariate Analysis – Categorical Variables

The categorical variable analysis reveals the following:

1. **Loan Term:** 36-month loans account for approximately 76.26% of the dataset, indicating a preference for shorter loan tenures.

2. **Loan Grade:** Grade B is the most frequently assigned grade, followed by grades C and A.

3. **Sub-grade:** B3 is the most common sub-grade.

4. **Employment Length:** Borrowers with 10 or more years of employment form the largest employment-length category.

5. **Home Ownership:** MORTGAGE is the most common home ownership category, representing approximately 50.08% of borrowers. Home ownership provides additional information about the borrower's financial circumstances and potential existing housing obligations.

6. **Verification Status:** Verified is the most frequent individual category, followed by Source Verified and Not Verified.

7. **Loan Purpose:** Debt consolidation is the most common loan purpose, accounting for approximately 59.21% of records. Credit card repayment is the second most common purpose.

8. **Initial Listing Status:** The F category is more common than W.

9. **Application Type:** Individual applications dominate the dataset, representing approximately 99.82% of records.

### Overall Inference

The dataset primarily consists of individual borrowers with mortgage-based home ownership, relatively long employment histories and 36-month loan terms. Debt consolidation is the predominant reason for borrowing.

## 6. Bivariate Analysis 

### Loan Grade vs Loan Status

A cross-tabulation was performed to examine the relationship between LoanTap-assigned grades and loan repayment outcomes.

The analysis reveals a clear trend:

- Grade A has the highest repayment rate of 93.71% and the lowest charge-off rate of 6.29%.
- Grade G has the lowest repayment rate of 52.16% and the highest charge-off rate of 47.84%.
- The charge-off rate consistently increases as the loan grade moves from A to G.

**Inference:**

Loan grade demonstrates a strong association with loan repayment behaviour. Borrowers assigned better grades have historically shown higher repayment rates, while borrowers with lower grades have demonstrated higher default risk.

This indicates that LoanTap's internal grading system may be an important indicator of creditworthiness.

**Business Recommendation:**

LoanTap can use loan grades as one of the inputs in its credit risk assessment process. Applicants with lower grades may require additional scrutiny, stricter eligibility criteria or more conservative lending terms.

However, loan grade should not be the sole determinant of loan approval.

### Interest Rate vs Loan Status

The relationship between interest rate and loan repayment status was examined using box plots and descriptive statistics.

The findings are:

- Borrowers whose loans were Charged Off have a mean interest rate of 15.88%, compared with 13.09% for Fully Paid borrowers.
- The median interest rate for Charged Off borrowers is 15.61%, compared with 12.99% for Fully Paid borrowers.
- The difference in mean interest rates is 2.79 percentage points.

**Inference:**

Borrowers with Charged Off loans generally have higher interest rates than borrowers who fully repaid their loans. This suggests a positive association between interest rates and credit risk.

Higher interest rates may reflect the lender's assessment of borrower risk and may also increase the repayment burden.

However, the distributions overlap, indicating that interest rate alone cannot accurately distinguish between defaulters and non-defaulters.

**Business Recommendation:**

Interest rate should be considered alongside other borrower characteristics, such as income, DTI, loan amount and credit history, when assessing creditworthiness.

### Relationship Between Loan Amount and Monthly Installment

The scatterplot indicates a strong positive linear relationship between loan amount and monthly installment. The correlation coefficient is **0.9539**, suggesting that borrowers with higher loan amounts generally have higher monthly repayment obligations.

This relationship is expected because the monthly installment is calculated based on the loan amount, interest rate and repayment tenure.

**Inference:** Loan amount and monthly installment contain overlapping information. Since they are highly correlated, their combined use in Logistic Regression may introduce multicollinearity. This will be considered during feature selection and model building.

### Relationship Between Annual Income and Loan Status

The boxplot and summary statistics indicate that borrowers who fully paid their loans generally had higher annual incomes than borrowers who were charged off.

The median annual income for Fully Paid borrowers was $65,000, compared with $59,000 for Charged Off borrowers. The mean income was also higher for Fully Paid borrowers ($75,829.95) than for Charged Off borrowers ($67,535.54).

The boxplot shows several extreme income outliers in both groups. These values compress the central portion of the plot and make direct visual comparison difficult.

**Inference:** Annual income may be a useful predictor of loan repayment behaviour. Borrowers with higher incomes appear to have a somewhat lower incidence of charged-off loans. However, this relationship needs to be evaluated alongside other borrower and loan characteristics.

### Relationship Between Debt-to-Income Ratio (DTI) and Loan Status

The analysis indicates that borrowers who were charged off had higher DTI values than borrowers who fully paid their loans.

- Median DTI for Charged Off borrowers: 19.34
- Median DTI for Fully Paid borrowers: 16.34
- Mean DTI for Charged Off borrowers: 19.66
- Mean DTI for Fully Paid borrowers: 16.82

The maximum DTI values are unusually high in both groups, suggesting the presence of extreme observations that require further investigation.

**Inference:** DTI may be a useful predictor of loan repayment behaviour. Borrowers with higher debt obligations relative to their income appear to have a higher incidence of charged-off loans. This is an association and does not establish causation.

### Investigation of Extreme DTI Values

The DTI distribution contains extreme observations.

The 99th percentile is 36.43, and the 99.9th percentile is 39.86, whereas the maximum value is 9,999. This indicates an exceptionally large gap between the majority of observations and the maximum value.

Further investigation revealed that the record with DTI = 9,999 has an annual income of zero. Other unusually high DTI observations were also associated with very low annual incomes.

These observations may represent data-quality issues or unusual borrower circumstances. They require further investigation before deciding on an appropriate treatment.

**Preliminary inference:** DTI is potentially an informative predictor of loan default, but its extreme values may distort statistical summaries and affect model performance. The treatment of these observations will be addressed during data preprocessing.

### Analysis of Extreme DTI Observations

The investigation showed that extreme DTI values are very rare in the dataset.

- 296 records (0.0747%) have DTI above 40.
- 35 records (0.0088%) have DTI above 50.
- Only 8 records (0.0020%) have DTI above 100.

The record count indicates that extreme DTI values affect a very small proportion of the dataset.

**Decision:** These observations will not be removed at this stage. Their treatment will be evaluated during data preprocessing. The original DTI values will be retained so that the impact of any cleaning or outlier treatment can be assessed.

### Relationship Between Revolving Credit Utilisation and Loan Status

The analysis indicates that borrowers who were charged off had higher revolving credit utilisation than borrowers who fully paid their loans.

- Median utilisation for Charged Off borrowers: 59.3%
- Median utilisation for Fully Paid borrowers: 53.7%
- Mean utilisation for Charged Off borrowers: 57.87%
- Mean utilisation for Fully Paid borrowers: 52.80%

Both groups contain observations with utilisation above 100%. The maximum value of 892.3% in the Fully Paid group is particularly unusual and requires further investigation.

**Inference:** Revolving credit utilisation may be a useful predictor of loan repayment behaviour. Borrowers with higher utilisation appear to have a greater incidence of charged-off loans. However, the relationship is associative and does not establish causation.

### Relationship Between Home Ownership and Loan Status

The analysis indicates differences in loan repayment patterns across home ownership categories.

- RENT has the highest Charged Off percentage at 22.66%.
- NONE has a Charged Off percentage of 22.58%.
- OWN has a Charged Off percentage of 20.68%.
- MORTGAGE has a relatively lower Charged Off percentage of 16.96%.
- ANY shows 100% Fully Paid, but its sample size needs to be checked.

**Inference:** Home ownership may be a useful predictor of loan repayment behaviour. However, the differences across categories should be interpreted alongside the number of observations in each category and other borrower characteristics. Home ownership alone is not sufficient to assess credit risk.

### Relationship Between Loan Purpose and Loan Status

The analysis indicates differences in repayment patterns across loan-purpose categories.

- Small business loans have the highest Charged Off percentage at 29.45%.
- Moving and renewable energy loans have charged-off percentages of 23.48% and 23.40%, respectively.
- Wedding loans have the lowest Charged Off percentage at 12.09%.
- Debt consolidation loans have a Charged Off percentage of 20.74%.

**Inference:** Loan purpose may provide useful information for assessing credit risk. Small business loans show a comparatively higher incidence of charged-off loans, while wedding and car loans show lower incidences in this dataset.

These findings represent associations and should be interpreted alongside category sample sizes and other borrower characteristics.


### Loan Purpose – Category Distribution

The loan-purpose distribution is highly imbalanced.

Debt consolidation is the most common purpose, accounting for 234,507 loans (approximately 59.2% of the dataset), followed by credit card loans with 83,019 records.

Small business loans have a relatively high charged-off rate of 29.45%, based on 5,701 records. Renewable energy and educational loans have much smaller sample sizes of 329 and 257, respectively, so their repayment percentages should be interpreted cautiously.

**Inference:** Loan purpose may be useful for predicting repayment behaviour. However, both the default rate and the number of loans in each category should be considered when evaluating business risk.

### Relationship Between Employment Length and Loan Status

The analysis shows relatively small differences in charged-off percentages across employment-length categories.

- Borrowers with less than one year of employment have the highest charged-off percentage at 20.69%.
- Borrowers with 10+ years of employment have the lowest charged-off percentage at 18.42%.
- The charged-off percentage does not decrease consistently as employment length increases.

**Inference:** Employment length may provide some predictive information about loan repayment behaviour. However, the differences across categories are relatively small, suggesting that employment length alone may not be a strong predictor of default. It should be evaluated alongside other financial and borrower characteristics.

### Correlation Analysis of Numerical Features

A correlation analysis was performed to identify relationships among numerical variables and investigate potential multicollinearity.

The following highly correlated pairs were identified:

| Feature Pair | Correlation |
|---|---:|
| loan_amnt and installment | 0.9539 |
| pub_rec and pub_rec_bankruptcies | 0.6994 |
| open_acc and total_acc | 0.6807 |

The strongest relationship was observed between loan amount and installment, indicating substantial overlap in the information captured by these variables.

Other moderately correlated relationships included loan amount and annual income (0.3369), installment and annual income (0.3304), and total accounts and mortgage accounts (0.3811).

**Inference:** Potential multicollinearity exists among some numerical predictors. These relationships will be investigated further during feature selection and VIF analysis. Highly correlated features will not be removed automatically, as their individual predictive value and business relevance must also be considered.

### Categorical Feature Cardinality Analysis

The categorical variables were examined to identify their unique-value counts and determine appropriate encoding strategies.

**Low-cardinality features:** Variables such as term, grade, emp_length, home_ownership, verification_status, purpose, initial_list_status and application_type have a manageable number of categories and are candidates for One-Hot Encoding.

**High-cardinality features:** emp_title (173,105 unique values), title (48,816 unique values) and address (393,700 unique values) have very high cardinality. Direct One-Hot Encoding would create a large number of features and increase model complexity. These variables will be excluded from the initial model or investigated for meaningful aggregation.

**Date features:** issue_d and earliest_cr_line will be converted into date formats and examined for feature engineering opportunities, such as loan issue period and credit history length.

**Target variable:** loan_status will be converted into a binary variable, with Charged Off represented as 1 and Fully Paid as 0.

**Inference:** Appropriate encoding and feature engineering are essential to prepare the dataset for Logistic Regression while controlling dimensionality and avoiding unnecessary model complexity.

### Missing Value Analysis

Missing values were identified in six features.

| Feature | Missing Count | Missing Percentage |
|---|---:|---:|
| mort_acc | 37,795 | 9.54% |
| emp_title | 22,927 | 5.79% |
| emp_length | 18,301 | 4.62% |
| title | 1,756 | 0.44% |
| pub_rec_bankruptcies | 535 | 0.14% |
| revol_util | 276 | 0.07% |

The highest proportion of missing values was observed in mort_acc, followed by emp_title and emp_length.

**Preliminary treatment strategy:**

- mort_acc: Evaluate median imputation and related account information.
- emp_title: Exclude from the initial model due to high cardinality.
- emp_length: Retain missing information as an Unknown category or evaluate mode imputation.
- title: Exclude from the initial model due to high cardinality and potential overlap with loan purpose.
- pub_rec_bankruptcies: Evaluate median or mode imputation.
- revol_util: Evaluate median imputation.

**Inference:** Missing-value treatment will be selected based on feature type, business meaning and distribution. Missing values will not automatically be replaced with zero.

### Investigation of Missing Values – `mort_acc`

- `mort_acc` has 37,795 missing values (9.54%).
- Its correlation with `total_acc` is approximately **0.381**, indicating a moderate positive relationship.
- Group-wise medians show that borrowers with fewer total credit accounts generally have a median of zero mortgage accounts, while the median increases to one for some higher `total_acc` groups.
- However, `total_acc` alone does not reliably determine the number of mortgage accounts.

**Preliminary decision:** Consider median imputation for `mort_acc`. The final imputation value will be confirmed after examining its overall distribution. Imputation parameters will be calculated using the training data only during model preprocessing.

### Missing Value Treatment – `mort_acc`

**Missing values:** 37,795 (9.54%)

**Investigation:**
- The median number of mortgage accounts is 1.
- The mode is 0, indicating that zero mortgage accounts is the most frequently observed value.
- The distribution is right-skewed, with fewer borrowers having a large number of mortgage accounts.
- `mort_acc` has a moderate positive correlation (0.381) with `total_acc`, but this relationship is not sufficiently strong to reliably estimate missing values from total accounts.

**Decision:** Use median imputation for missing `mort_acc` values. The median is preferred over the mode because it is less influenced by the skewed distribution and extreme values.

**Implementation:** Calculate the imputation value from the training data only during model preprocessing.

### Missing Value Treatment – `emp_length`

**Missing values:** 18,301 (4.62%)

**Investigation:**

The relationship between missing employment length and loan repayment status was examined.

| Employment length | Charged Off | Fully Paid |
|---|---:|---:|
| Available | 19.23% | 80.77% |
| Missing | 27.53% | 72.47% |

Borrowers with missing employment length have a higher charge-off rate (27.53%) than borrowers with available employment information (19.23%).

**Decision:** Retain missing employment length as a separate `Unknown` category rather than imputing it with the mode.

**Reason:** Missingness itself may contain predictive information. Replacing missing values with an actual employment-length category could misrepresent the borrower's employment history.

**Implementation:** Convert missing values to `Unknown` during preprocessing. Evaluate its contribution to model performance later.

### Missing Value Treatment – `pub_rec_bankruptcies`

**Missing values:** 535 (0.14%)

**Investigation:**
- The median and mode are both 0.
- Approximately 88.5% of borrowers have zero public record bankruptcies.
- The distribution is strongly right-skewed.
- The correlation with `pub_rec` is approximately 0.699, indicating a fairly strong positive relationship.

**Decision:** Use median imputation (0) for missing values.

**Reason:** The median is robust to the right-skewed distribution, and the percentage of missing values is very small.

**Additional consideration:** Retain both `pub_rec` and `pub_rec_bankruptcies` initially because they represent related but distinct credit-history information. Evaluate potential multicollinearity during modelling.

### Missing Value Treatment – `revol_util`

**Missing values:** 276 (0.07%)

**Investigation:**
- Median: 54.8
- Mean: 53.79
- Mode: 0
- 99th percentile: 98
- 99.9th percentile: 102
- Maximum: 892.3

The distribution has a small number of unusually high values. Values above 100% are unusual but should not automatically be classified as invalid.

**Decision:** Use median imputation (54.8) for missing values.

**Reason:** Median imputation is robust to extreme observations and is preferable to mode imputation in this case.

**Additional consideration:** Investigate extreme values separately during outlier treatment. Do not remove or cap them without evaluating their impact.

### Outlier Investigation – `annual_inc`

**IQR analysis:**

| Statistic | Value |
|---|---:|
| Q1 | 45,000 |
| Q3 | 90,000 |
| IQR | 45,000 |
| Lower bound | -22,500 |
| Upper bound | 157,500 |
| Potential outliers | 16,700 |
| Outlier percentage | 4.22% |

**Further investigation:**

The potential outliers have:
- Median annual income of 199,350.
- 75th percentile of 245,000.
- 99th percentile of 735,066.
- Maximum annual income of 8,706,582.

Most observations above the IQR threshold may represent legitimate high-income borrowers. However, the maximum value is exceptionally high and warrants further inspection.

**Preliminary decision:** Do not delete or automatically cap the outliers. Investigate extreme observations and consider logarithmic transformation or robust scaling during preprocessing.

**Reason:** Income is an important indicator of repayment capacity, and removing legitimate high-income borrowers could result in information loss.

### Final Outlier Assessment – `annual_inc`

The IQR method identified 16,700 potential outliers (4.22%), with an upper bound of 157,500.

Further investigation identified 65 borrowers with annual incomes above 1 million, representing approximately 0.0164% of the dataset.

Their repayment distribution was:
- Fully Paid: 57 (87.69%)
- Charged Off: 8 (12.31%)

The charge-off rate in this small group is lower than the overall dataset charge-off rate of 19.61%. However, the sample size is too small to draw strong conclusions.

**Decision:** Retain the high-income observations without deletion or arbitrary capping. Consider logarithmic transformation or robust scaling during preprocessing and evaluate the impact on model performance.

### Outlier Investigation – `dti`

**IQR analysis:**

| Statistic | Value |
|---|---:|
| Q1 | 11.28 |
| Q3 | 22.98 |
| IQR | 11.70 |
| Lower bound | -6.27 |
| Upper bound | 40.53 |
| Potential outliers | 275 |
| Outlier percentage | 0.0694% |

**Further investigation:**

- The 99th percentile is 36.43, and the 99.9th percentile is 39.86.
- The maximum DTI is 9,999, which is substantially higher than the rest of the distribution.
- Eight observations have DTI above 100.
- Some extreme observations have very low reported annual income.
- The observation with DTI 9,999 also has annual income of zero, suggesting a possible data-quality or calculation issue.

**Preliminary decision:** Do not automatically delete all IQR outliers. Retain potentially meaningful high-DTI observations while separately investigating extreme values, particularly those above 100.

**Further action:** Examine records with zero annual income and evaluate an appropriate treatment for implausible DTI values during preprocessing.

### DTI – Additional Data Quality Investigation

Further investigation identified exactly one borrower with zero annual income. This borrower has a DTI of 9,999 and a loan amount of 3,700.

The distribution of extreme DTI observations is:

| DTI threshold | Count |
|---|---:|
| Above 40.53 | 275 |
| Above 50 | 35 |
| Above 100 | 8 |
| Above 500 | 2 |
| Above 1,000 | 2 |

**Interpretation:**

The number of observations decreases substantially at higher DTI thresholds. The DTI value of 9,999, associated with zero annual income, is particularly suspicious and may represent a data-quality or placeholder value.

**Preliminary decision:** Investigate all eight observations above 100 individually. Consider treating the 9,999 value as missing, retaining an anomaly indicator, and imputing it during model preprocessing. Do not automatically remove all high-DTI observations.

### Final Outlier Assessment – `dti`

The IQR method identified 275 potential outliers (0.0694%), with an upper bound of 40.53.

Further investigation revealed eight observations with DTI above 100.

Of these:
- One borrower has DTI = 9,999 and annual income = 0.
- One borrower has DTI = 1,622 and annual income = 600.
- Six other borrowers have DTI values between approximately 107 and 381.
- Seven of the eight borrowers are Fully Paid.

**Interpretation:**

The extreme values are unusual, particularly when considered alongside the reported annual incomes. However, the repayment outcomes show that high DTI values do not automatically imply loan default.

**Decision:**
- Treat DTI = 9,999 as a suspected anomalous value. Replace it with missing during preprocessing and retain an anomaly indicator.
- Preserve the remaining seven observations initially.
- Retain the other IQR outliers rather than automatically deleting or capping them.
- Evaluate the effect of extreme values on model performance during modelling.

The original dataset will remain unchanged, with these treatments applied through preprocessing.

### Outlier Analysis: Revolving Balance (`revol_bal`)

**Objective:** Identify unusually high revolving balances and determine whether they require treatment before modelling.

**IQR analysis:**
- Q1: 6,025
- Q3: 19,620
- IQR: 13,595
- Lower bound: -14,367.50
- Upper bound: 40,012.50
- Potential outliers: 21,259 (5.37% of the dataset)

**Distribution of potential outliers:**
- Median: 52,501
- Mean: 71,241
- 95th percentile: 160,993
- 99th percentile: 303,498
- Maximum: 1,743,266

**Observations:**

The revolving balance distribution is strongly right-skewed. The mean of the potential outliers is considerably higher than their median, indicating that a small number of exceptionally large balances influence the average.

The highest observations are unusually large but cannot be assumed to be erroneous without further evidence.

**Preliminary decision:**

- Retain the observations; do not automatically delete or cap them.
- Consider a `log1p` transformation during feature engineering.
- Evaluate the effect of transformation on model performance.
- Reassess the feature during model validation.

**Conclusion:** The IQR method identifies statistically unusual revolving balances, but these values may represent genuine borrower financial profiles. Further investigation and model-based evaluation are required before deciding on treatment.

### Outlier Analysis – Revolving Credit Features

**1. Revolving Balance (`revol_bal`)**

- IQR upper limit: 40,012.50
- Number of outliers: 21,259
- Percentage of dataset: 5.37%
- Maximum value: 1,743,266

The feature has a strongly right-skewed distribution, with a small proportion of borrowers reporting exceptionally high revolving balances.

**Decision:** Retain the observations initially. High revolving balances may represent genuine borrower characteristics. Consider transformations or robust scaling during preprocessing and evaluate their impact on model performance.

**2. Revolving Utilisation (`revol_util`)**

- IQR upper limit: 128.55
- Number of outliers: 12
- Percentage of dataset: approximately 0.003%
- Maximum value: 892.3

Twelve observations exceed the IQR upper limit. Eleven values range from 129.4 to 153.0, while one observation has an unusually high value of 892.3.

**Decision:** Retain the observations provisionally and investigate the exceptionally high value separately. Do not automatically delete or cap these records based only on the IQR method.

**Overall conclusion:** Statistical outliers are not necessarily invalid records. Final treatment will be determined during preprocessing, considering data quality, business context and model performance.

### Investigation of Extreme Revolving Utilisation Value

One observation (index 137211) was identified with a revolving utilisation of 892.3%, significantly higher than the other identified outliers.

On examining the complete record:
- Revolving balance: $2,677
- Annual income: $45,000
- Loan amount: $3,500
- Number of open accounts: 2
- Loan status: Fully Paid

The reported revolving balance and utilisation imply an unusually low revolving credit limit of approximately $300.

This value is therefore suspected to be a data-quality or reporting anomaly. However, the actual cause cannot be confirmed from the available dataset.

**Decision:** Retain the borrower record but flag the extreme `revol_util` value for special treatment during preprocessing. Consider replacing the value with missing and applying median imputation using training data only. Evaluate the impact on model performance before finalising the decision.

### Additional Investigation – Revolving Utilisation Outliers

Further examination of the 12 observations exceeding the IQR upper limit of 128.55% revealed:

- Median utilisation: 145.95%
- Maximum utilisation: 892.3%
- Median revolving balance: $13,765
- Maximum revolving balance: $27,233

Eleven observations have utilisation values between 129.4% and 153.0%. One observation, at index 137211, has an exceptionally high utilisation of 892.3%, despite a revolving balance of only $2,677.

This suggests that the 892.3% observation may be an isolated reporting or data-quality anomaly.

**Final preliminary decision:**
- Retain all borrower records.
- Retain `revol_bal` observations and evaluate transformations or robust scaling.
- Flag the 892.3% `revol_util` observation for special treatment.
- If treated as missing, apply median imputation using training data only.
- Validate the decision using model performance.

### Date Conversion and Credit History Feature Engineering

The columns `issue_d` and `earliest_cr_line` were initially stored as strings in the format `MMM-YYYY`.

Both columns were converted to datetime format using the `%b-%Y` format.

A new feature, `credit_history_months`, was derived by calculating the difference between the loan issue date and the earliest reported credit line date.

An additional feature, `credit_history_years`, was created by converting the month difference into years.

**Summary of credit history:**
- Number of records: 396,030
- Average credit history: 189.05 months (15.8 years)
- Median credit history: 172 months (14.3 years)
- Minimum: 36 months (3 years)
- Maximum: 842 months (approximately 70.2 years)
- Negative credit history records: 0

**Preliminary observation:** Borrowers have an average credit history of approximately 15.8 years. The maximum value of 842 months is unusually high and requires further investigation before finalising the feature.

**Potential modelling relevance:** Credit history length may provide information about a borrower's credit experience and could help explain differences in loan repayment outcomes.

### Investigation of Long Credit Histories

The 10 borrowers with the longest calculated credit histories were examined.

Their credit histories ranged from approximately 61 to 70 years, with earliest reported credit lines dating between 1944 and 1955.

These values are unusually high. However, the dataset does not contain borrower age or date of birth, so their validity cannot be independently established.

The records also contain both Fully Paid and Charged Off loans, indicating that long credit history alone does not determine repayment outcome.

**Decision:**
- Retain these observations provisionally.
- Do not apply arbitrary capping or deletion.
- Evaluate the feature during model development.
- Retain `credit_history_months` as the primary feature and exclude the redundant `credit_history_years` feature from the final predictor set.

**Conclusion:** Credit history length is a potentially useful predictor, but its actual contribution to default prediction must be established through model evaluation.

### Bivariate Analysis – Credit History and Loan Status

Credit history length was compared against loan repayment outcomes.

**Credit history by loan status:**

| Statistic | Charged Off | Fully Paid |
|---|---:|---:|
| Mean (months) | 184.34 | 190.20 |
| Median (months) | 167 | 174 |

Borrowers who fully repaid their loans had a slightly longer average and median credit history than borrowers who defaulted.

**Default rate by credit history group:**

| Credit history | Charged-off rate |
|---|---:|
| 0–5 years | 23.10% |
| 5–10 years | 21.26% |
| 10–15 years | 20.18% |
| 15–20 years | 18.77% |
| 20–30 years | 17.79% |
| 30–42 years | 17.99% |
| 42+ years | 23.65% |

**Observations:**
- Default rates generally decrease as credit history increases, up to the 20–30-year group.
- The 30–42-year group has a similar default rate.
- The 42+ years group shows a higher default rate, requiring further investigation of its sample size.

**Preliminary inference:** Credit history length appears to have a relationship with loan repayment outcomes. However, the relationship is not strictly linear. Longer credit history does not necessarily guarantee lower default risk.

The feature will be retained for model development, subject to further evaluation. This analysis identifies an association and does not establish causation.

### Bivariate Analysis – Credit History and Loan Status

Credit history length was compared against loan repayment outcomes using seven credit-history groups.

**Key findings:**

1. Borrowers with 0–5 years of credit history have a charged-off rate of 23.10%.
2. The default rate generally decreases as credit history increases, reaching 17.79% for borrowers with 20–30 years of credit history.
3. The 30–42-year group has a similar default rate of 17.99%.
4. The 42+ years group has a higher default rate of 23.65%, based on 2,025 borrowers (approximately 0.51% of the dataset).

The overall pattern suggests that borrowers with shorter credit histories have higher default rates. However, the relationship is not strictly linear.

**Decision:**
- Retain `credit_history_months` as a potential model feature.
- Exclude the redundant `credit_history_years` feature.
- Do not remove or cap unusually long credit histories based solely on these findings.
- Allow the model to evaluate the predictive contribution of credit history.

**Conclusion:** Credit history length appears to be associated with loan repayment outcomes and may provide useful information for default prediction. This association does not establish causation.

### Bivariate Analysis – Loan Issue Year and Loan Status

The loan issue year was extracted from `issue_d` to investigate whether loan repayment outcomes vary across different lending periods.

**Charged-off rates by issue year:**

| Issue year | Charged-off rate |
|---|---:|
| 2007 | 16.41% |
| 2008 | 15.81% |
| 2009 | 12.26% |
| 2010 | 13.21% |
| 2011 | 15.25% |
| 2012 | 16.48% |
| 2013 | 15.74% |
| 2014 | 23.11% |
| 2015 | 24.90% |
| 2016 | 13.13% |

**Observations:**
- The charged-off rate fluctuates across issue years.
- The highest observed rates occur in 2014 and 2015.
- The lowest observed rate occurs in 2009.
- The rate drops considerably in 2016.

**Preliminary inference:** Loan issue year appears to be associated with repayment outcomes. However, changes in borrower composition, lending conditions and the length of time available for loans to reach an outcome may influence these differences.

**Further investigation:** Examine the number of loans issued each year and consider differences in loan maturity before drawing conclusions.

**Potential modelling relevance:** Retain `issue_d` or an appropriate derived time feature for further evaluation, while considering a time-based validation strategy.

### Issue Year vs Loan Default – Effect of Loan Term

To investigate whether the relationship between issue year and loan default was influenced by loan duration, the analysis was repeated separately for 36-month and 60-month loans.

**Key observations:**

1. 60-month loans generally had higher charged-off percentages than 36-month loans across the years examined.
2. The charged-off rate increased during 2014 and 2015 for both loan-term categories.
3. Therefore, the higher default rates observed in 2014–2015 cannot be attributed solely to a change in the proportion of 60-month loans.
4. Both loan-term categories showed lower charged-off rates in 2016. However, this observation requires caution because the available data does not establish whether all loans had comparable outcome-observation periods.

**Conclusion:**

Loan term and issue year both appear to have associations with loan repayment outcomes. The year-wise pattern persists even after separating loans by term, suggesting that other borrower, loan or economic factors may also contribute to the observed differences.

These are exploratory associations and do not establish causation. Both `issue_year` and `term` will be considered for further evaluation during feature selection and model development.

### Issue Year vs Loan Default – Effect of Loan Term and Sample Size

To investigate whether the relationship between issue year and loan default was influenced by loan duration, the analysis was repeated separately for 36-month and 60-month loans. The number of loans in each year-term combination was also examined.

**Key observations:**

1. 60-month loans generally had higher charged-off percentages than 36-month loans across the years examined.
2. The charged-off rate increased during 2014 and 2015 for both loan-term categories.
3. The 2014 and 2015 observations are supported by substantial sample sizes, with 102,860 and 94,264 loans respectively.
4. The higher default rates observed in 2014–2015 cannot be attributed solely to a change in the proportion of 60-month loans.
5. The 2016 charged-off rate was lower for both loan-term categories. Although the dataset contains 28,088 loans from 2016, the reason for the lower rate cannot be established from the available information.

**Conclusion:**

Issue year and loan term both show associations with loan repayment outcomes. The higher charged-off rates in 2014–2015 persist even when loans are analysed separately by term, suggesting that other borrower, loan or economic factors may also contribute.

The analysis is exploratory and does not establish causation. The availability of only Fully Paid and Charged Off outcomes, without an explicit observation or closure date, also limits our ability to assess whether all loans had comparable outcome-observation periods.

Both `issue_year` and `term` will be considered for further evaluation during feature selection and model development.

### Verification Status vs Loan Default

The relationship between `verification_status` and `loan_status` was examined using the percentage distribution of loan outcomes within each verification category.

**Key observations:**

1. Not Verified loans had the lowest charged-off rate at 14.64%.
2. Source Verified loans had a charged-off rate of 21.47%.
3. Verified loans had the highest charged-off rate at 22.32%.
4. The difference between the Not Verified and Verified categories was 7.68 percentage points.
5. All three categories had substantial sample sizes, making the observed differences worth further investigation.

**Conclusion:**

Verification status shows an association with loan repayment outcomes in this dataset. However, this does not imply that verification itself influences the likelihood of default. Differences in borrower profiles, loan characteristics and other risk factors may explain some of the observed variation.

`verification_status` will be retained for further evaluation during model development.

### Sub-grade vs Loan Default

The relationship between `sub_grade` and `loan_status` was examined by calculating the percentage of charged-off and fully paid loans within each sub-grade.

**Key observations:**

1. The charged-off rate generally increased as the sub-grade moved from A1 towards G5.
2. A1 had the lowest charged-off rate at 2.87%.
3. The rate increased consistently from A1 (2.87%) to E5 (40.31%).
4. Higher-risk sub-grades, particularly F and G, showed substantially higher charged-off rates.
5. G3 had the highest observed charged-off rate at 51.09%.
6. A few minor reversals occurred in the F and G categories, but the overall upward trend remained clear.

**Conclusion:**

Sub-grade shows a strong association with loan repayment outcomes. Borrowers in lower-quality sub-grades had substantially higher charged-off rates in this dataset.

Since `grade` and `sub_grade` represent overlapping classifications, their combined use may introduce redundancy in the model. Both will be evaluated during feature selection before deciding which representation to retain.

### Initial List Status vs Loan Default

The relationship between `initial_list_status` and `loan_status` was examined by calculating the percentage of charged-off and fully paid loans within each category.

**Key observations:**

1. Loans with initial list status `f` had a charged-off rate of 19.31%.
2. Loans with initial list status `w` had a charged-off rate of 20.08%.
3. The difference between the two categories was only 0.77 percentage points.
4. Both categories had substantial sample sizes.

**Conclusion:**

`initial_list_status` shows only a small difference in repayment outcomes across its two categories. It may have limited predictive value on its own, although its contribution should ultimately be assessed during model development.

The variable will be retained for initial model evaluation rather than excluded solely on the basis of this exploratory analysis.

### Application Type vs Loan Default

The relationship between `application_type` and `loan_status` was examined by calculating the percentage of charged-off and fully paid loans within each application category.

**Key observations:**

1. Individual applications accounted for 395,319 loans, approximately 99.82% of the dataset, with a charged-off rate of 19.61%.
2. Joint applications accounted for only 425 loans and had a charged-off rate of 12.71%.
3. Direct Pay applications accounted for only 286 loans and had a charged-off rate of 35.66%.
4. Although the Joint and Direct Pay categories showed different default rates from Individual applications, their sample sizes were very small.

**Conclusion:**

The dataset is heavily dominated by Individual applications. The observed default rates for Joint and Direct Pay applications should be interpreted cautiously because of their limited representation.

`application_type` will be retained for initial model evaluation, but the contribution of its minority categories will be assessed carefully during preprocessing and model development.

**Duplicate Record Check**

- Total exact duplicate rows: 0
- Duplicate percentage: 0%
- No records were identified as exact duplicates across all available columns.
- No rows need to be removed for exact duplication.

**Limitation:** The dataset does not contain an explicitly identifiable loan ID column. Therefore, the absence of exact duplicates does not establish that every record represents a unique loan.

# Data Leakage Assessment

## 1. Prediction Scenario

The objective is to predict the probability of loan default during the underwriting process, before the loan is approved and issued.

The model should use information available at the intended prediction point. Information generated later in the lending process or outcomes of previous risk assessments should not be used indiscriminately.

## 2. Feature-wise Assessment

### 2.1 Interest Rate (`int_rate`)

**Business meaning:** Annual interest rate charged on the loan.

**Assessment:**
- Interest rates are generally determined as part of the lender's pricing and underwriting process.
- The rate may reflect the lender's earlier assessment of borrower risk.
- It may not be available at our chosen early underwriting prediction point.

**Decision:** Exclude from the primary pre-approval model.

**Further consideration:** A separate comparison model could include this feature if the prediction is made after proposed pricing has been determined.

### 2.2 Monthly Installment (`installment`)

**Business meaning:** Expected monthly loan repayment amount.

**Assessment:**
- The installment is calculated using the loan amount, interest rate and repayment term.
- It may be available after proposed loan terms have been calculated.
- Its availability depends on the exact prediction point.

**Decision:** Exclude from the primary model if predicting before proposed loan terms are calculated.

**Further consideration:** It may be retained in a separate model if the prediction occurs after the proposed installment is known but before final approval.

### 2.3 Loan Grade (`grade`)

**Business meaning:** Lender-assigned risk classification, ranging from A to G.

**Assessment:**
- The grade represents an existing lender's assessment of borrower risk.
- It has a strong association with loan outcomes in our EDA.
- Using it in an independent pre-approval risk model could make the model dependent on an existing risk assessment.

**Decision:** Exclude from the primary model.

### 2.4 Sub-grade (`sub_grade`)

**Business meaning:** More granular risk classification within each loan grade, ranging from A1 to G5.

**Assessment:**
- It is an extension of the lender-assigned grade.
- It has a strong association with loan outcomes.
- It overlaps with the information represented by `grade`.

**Decision:** Exclude from the primary model for the same reasons as `grade`.

## 3. Summary of Decisions

| Feature | Proposed treatment | Reason |
|---|---|---|
| int_rate | Exclude | Pricing information may not be available at the chosen prediction point |
| installment | Exclude initially | Depends on proposed loan terms being calculated |
| grade | Exclude | Existing lender risk assessment |
| sub_grade | Exclude | Existing lender risk assessment; overlaps with grade |

**Important:** These are feature-selection decisions for the primary model, not decisions to delete columns from the original dataset. The original data will be preserved.

## 4. Outstanding Considerations

- Confirm the precise timing of interest-rate and installment assignment in the source lending process.
- Maintain a clear distinction between an independent pre-approval model and a post-pricing risk model.
- Evaluate alternative feature sets only when their information would genuinely be available at the relevant prediction point.

### 2.5 Verification Status (`verification_status`)

**Business meaning:** Indicates whether the applicant's income information has been verified by the lender.

**Assessment:**
- Income verification is part of the underwriting process.
- The verification status is generated or updated after the application is submitted.
- Our chosen prediction point is immediately after application submission, before income verification begins.
- Therefore, this information is unavailable at the intended prediction point.

**Decision:** Exclude `verification_status` from the primary model.

**Further consideration:** It may be suitable for a separate model that makes predictions after income verification has been completed.

### 2.6 Initial Listing Status (`initial_list_status`)

**Business meaning:** Indicates how a loan is initially offered to investors.

- `f`: Fractional loan – offered to investors in fractional portions.
- `w`: Whole loan – offered to investors as a whole loan.

**Assessment:**
- This feature describes the investor-side listing classification.
- It is not a direct borrower or applicant characteristic.
- It relates to the loan listing process, which occurs after the initial underwriting assessment.
- It is not available at our chosen prediction point immediately after application submission.

**EDA observation:** The Charged Off rates were 19.31% for fractional loans and 20.08% for whole loans, a relatively small difference of 0.77 percentage points.

**Decision:** Exclude from the primary pre-approval model because it is unavailable at the intended prediction point and does not represent an applicant characteristic.

**Further consideration:** Retain in the original dataset for historical analysis.

### 2.7 Issue Date (`issue_d`) and Issue Year (`issue_year`)

**Business meaning:** `issue_d` represents the date on which the loan was issued. `issue_year` is derived from this date.

**Assessment:**
- The issuance date is generated only after the loan has been approved and issued.
- Our prediction point is immediately after application submission, before approval.
- Therefore, neither the actual issuance date nor its derived year is available at the intended prediction point.

**Decision:** Exclude both `issue_d` and `issue_year` from the primary model.

**Further consideration:** Retain them in the original dataset for historical analysis, including loan-vintage and default-rate trends.

### 2.8 Earliest Credit Line (`earliest_cr_line`)

**Business meaning:** Represents the date on which the borrower's oldest recorded credit account was opened.

**Assessment:**
- This information can be obtained from the applicant's credit bureau report.
- It does not require the applicant to be an existing LoanTap customer.
- Under our project assumption, LoanTap obtains the credit report during application assessment.
- Therefore, the feature is potentially available at the intended prediction point.

**Decision:** Retain `earliest_cr_line` as a potential predictor.

**Feature engineering consideration:**
- The previously derived `credit_history_months` uses `issue_d` as its reference date.
- Since the actual issuance date is unavailable at application submission, this calculation is not suitable for our primary model.
- Recalculate credit history duration using the application date when available.

**Status:** Retain the original credit-history date; recalculate duration using an appropriate application-date reference.

### 2.9 Residential Address (`address`)

**Business meaning:** Applicant's residential address.

**Assessment:**
- Residential address is generally collected during the loan application process.
- It is potentially available at our prediction point.
- However, the column has extremely high cardinality, with 393,700 unique values across 396,030 records (approximately 99.41%).
- Direct categorical encoding would create a very large feature space.
- Address information is also sensitive personal information.

**Decision:** Exclude `address` from the primary model.

**Further consideration:** Geographic aggregation could be explored separately if a valid business justification and appropriate privacy safeguards exist.

### 2.10 Employment Title (`emp_title`)

**Business meaning:** Applicant's reported job title or occupation.

**Assessment:**
- Potentially available at application submission.
- Contains 173,105 unique values, indicating extremely high cardinality.
- Job-title descriptions may be inconsistent and difficult to standardise.
- Direct categorical encoding would create a very large feature space.
- Annual income is already available as a direct numerical indicator of income.

**Decision:** Exclude `emp_title` from the initial model.

**Important consideration:** High cardinality and modelling complexity are the primary reasons for exclusion. A relationship between job title and annual income does not, by itself, establish multicollinearity.

**Further consideration:** Occupational categories could be explored separately through text standardisation and grouping if required.

### 2.11 Loan Title (`title`)

**Business meaning:** Applicant-provided description or title associated with the loan application.

**Assessment:**
- Potentially available at application submission.
- Contains 48,816 unique values, indicating high cardinality.
- It is free-text information supplied by the applicant and may be vague, inconsistent or not fully representative of the actual purpose.
- The structured `purpose` column already captures the broad reason for borrowing using 14 standardised categories.

**Decision:** Exclude `title` from the primary prediction model and retain `purpose`.

**Further consideration:** The title field may still be useful for exploratory or text-based analysis of how applicants describe their borrowing requirements.

### Feature: `emp_length` – Employment Length

**Availability at prediction time:** Available

Employment length is potentially collected from the applicant during loan application and can therefore be considered available at our chosen prediction point: immediately after application submission.

**EDA observations:**
- The feature contains 11 categories, including `10+ years` and `<1 year`.
- The largest category is `10+ years`, with 126,041 records.
- Missing values: 18,301 (4.62%).
- The Charged Off rate for records with missing employment length is approximately 27.53%, compared with 19.23% for records with reported employment length.

**Decision: Retain**

We will retain `emp_length` as a predictor because it is potentially available at application submission and has a manageable number of structured categories.

**Missing-value treatment:**
- Retain all records.
- Convert missing values into a separate `Unknown` category.
- Do not impute missing values with the most frequent category (`10+ years`), as this would make an unsupported assumption about the applicant's employment history.

**Reasoning:** Treating missing values as a separate category preserves information and allows the model to learn whether missing employment information is associated with default risk. The observed association should not be interpreted as causation.

**Preprocessing note:** Implement the missing-value treatment during preprocessing, without modifying the original raw dataset.

### Feature: `home_ownership` – Home Ownership Status

**Availability at prediction time:** Available

Home ownership status is an applicant-provided feature collected during the loan application process. It is therefore potentially available at our chosen prediction point: immediately after application submission.

**EDA observations:**

| Category | Records | Charged Off Rate |
|---|---:|---:|
| MORTGAGE | 198,348 | 16.96% |
| RENT | 159,790 | 22.66% |
| OWN | 37,746 | 20.68% |
| OTHER | 112 | 14.29% |
| NONE | 31 | 22.58% |
| ANY | 3 | 0.00% |

**Decision: Retain**

The feature is relevant to an applicant's housing situation and is potentially available at application submission.

**Preprocessing consideration:**

- Retain `MORTGAGE`, `RENT`, and `OWN` as separate categories.
- Consider grouping the very low-frequency categories `OTHER`, `NONE`, and `ANY` into a single `Other` category.
- Apply categorical encoding during preprocessing.

**Reasoning:** The three major categories account for nearly all records. Grouping the rare categories can reduce unnecessary complexity and help avoid unstable estimates from very small groups.

The observed differences in default rates indicate an association with loan repayment outcomes, but do not establish causation.

### Feature: `annual_inc` and `log_annual_inc` – Annual Income

**Availability at prediction time:** Available

Annual income is applicant-provided information collected during the loan application process. It is potentially available at our chosen prediction point.

**EDA observations:**
- `annual_inc` is highly right-skewed.
- Charged Off applicants have a lower mean and median annual income than Fully Paid applicants.
- Mean annual income:
  - Charged Off: 67,535.54
  - Fully Paid: 75,829.95
- Median annual income:
  - Charged Off: 59,000
  - Fully Paid: 65,000

**Feature engineering:** Created `log_annual_inc` to reduce the impact of the highly skewed income distribution.

**Decision:**
- Retain `log_annual_inc` as the model input.
- Exclude raw `annual_inc` from the initial model to avoid retaining two representations of the same underlying information.
- Preserve `annual_inc` in the original dataset for reference, EDA and interpretation.

**Reasoning:** The logarithmic transformation compresses extreme values and can make the income distribution more suitable for modelling. We will evaluate the transformation during model development.

**Preprocessing note:** Ensure that the transformation is valid for all income values. Since income can be zero, `np.log1p(annual_inc)` is appropriate. Any invalid or negative values must be investigated before transformation.

### Feature: `loan_amnt` – Loan Amount

**Availability at prediction time:** Available

`loan_amnt` represents the amount requested by the applicant. It is known at the time of application submission and is therefore available at our chosen prediction point.

**EDA observations:**
- The feature represents the applicant's requested borrowing amount.
- It is relevant to assessing the size of the financial obligation.
- Its relationship with repayment outcomes can be further evaluated during modelling.

**Decision: Retain**

**Reasoning:** The requested loan amount is an important financial characteristic and may help the model estimate default risk.

**Preprocessing consideration:** Examine its distribution and outliers during preprocessing. Scaling may be considered depending on the selected algorithm.

### Feature: `term` – Loan Repayment Term

**Availability at prediction time:** Available

`term` represents the repayment duration selected by the applicant, such as 36 months or 60 months.

**EDA observations:**
- The dataset contains two repayment-term categories: 36 months and 60 months.
- Approximately 76.25% of loans have a 36-month term.
- Repayment term influences the monthly installment, along with loan amount and interest rate.

**Decision: Retain**

**Reasoning:** Since the repayment term is selected by the applicant and is known at application submission, it is available at our chosen prediction point. It is also relevant to repayment capacity and loan default risk.

**Preprocessing consideration:** Treat `term` as a categorical feature and encode it during preprocessing.

**Important distinction:** Although `term` influences `installment`, this does not make it a leakage feature. Its availability at the prediction point determines whether it can be used.

### Feature: `purpose` – Loan Purpose

**Availability at prediction time:** Available

`purpose` represents the reason for which the applicant is requesting the loan. It is applicant-provided information and is potentially available at application submission.

**EDA observations:**
- The feature contains 14 structured categories.
- `debt_consolidation` is the most common category, with 234,507 records.
- `small_business` has a relatively high Charged Off rate of 29.45%.

**Decision: Retain**

**Reasoning:** Loan purpose provides information about the applicant's borrowing requirement and may help identify differences in repayment risk across categories.

**Preprocessing consideration:**
- Treat `purpose` as a categorical feature.
- Encode the categories during preprocessing.
- Review category frequencies before encoding and consider grouping genuinely rare categories if necessary.

**Important:** Differences in default rates across purposes indicate associations, not causation. Rates from categories with small sample sizes should be interpreted cautiously.

### Feature: `dti` – Debt-to-Income Ratio

**Availability at prediction time:** Available, assuming it is calculated using applicant-provided financial information and/or information obtained during the application-stage credit assessment.

**EDA observations:**
- Mean DTI for Charged Off applicants: 19.66
- Mean DTI for Fully Paid applicants: 16.82
- Maximum observed DTI: 9,999, an extreme value requiring investigation.
- Only a small proportion of records have DTI above 40.

**Decision: Retain**

**Reasoning:** DTI measures the applicant's existing debt burden relative to income and is relevant to assessing repayment capacity.

**Preprocessing considerations:**
- Investigate extreme values, particularly the maximum value of 9,999.
- Check whether extreme values are data-entry issues, special values or genuine observations.
- Do not automatically delete extreme observations without investigation.
- Decide on appropriate treatment during preprocessing.
- Consider scaling depending on the selected algorithm.

**Important:** Confirm that the DTI value represents information available at application submission and was not calculated using post-application or post-approval information.

### Feature: `open_acc` – Number of Open Credit Accounts

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the number of currently open credit accounts.
- Shows a positive correlation with `total_acc` (approximately 0.68).

**Decision: Retain**

**Reasoning:** The number of open accounts provides information about the applicant's existing credit exposure and borrowing activity. It may help the model assess repayment risk.

**Preprocessing considerations:**
- Examine the distribution and identify unusually high values.
- Treat as a numerical feature.
- Consider scaling depending on the selected algorithm.

**Additional consideration:** Since `open_acc` is correlated with `total_acc`, we will examine their relationship during feature analysis. Correlation alone is not sufficient reason to exclude either feature.

### Feature: `total_acc` – Total Number of Credit Accounts

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the total number of credit accounts in the applicant's credit history, including open and closed accounts.
- Correlation with `open_acc`: approximately 0.68.
- Correlation with `mort_acc`: approximately 0.38.

**Decision: Retain**

**Reasoning:** `total_acc` provides a broader view of the applicant's credit history and borrowing activity. It complements `open_acc`, which represents only currently open accounts.

**Preprocessing considerations:**
- Treat as a numerical feature.
- Examine its distribution and unusually high values.
- Consider scaling depending on the selected algorithm.
- Evaluate its relationship with `open_acc` during feature analysis.

**Important:** The correlation with `open_acc` does not automatically make the two features redundant. Their individual contribution should be evaluated during modelling.

### Feature: `pub_rec` – Derogatory Public Records

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the number of derogatory public records in the applicant's credit history.
- The most common value is 0, indicating that most applicants have no reported derogatory public records.
- It has a correlation of approximately 0.70 with `pub_rec_bankruptcies`.

**Decision: Retain**

**Reasoning:** This feature provides information about adverse credit history and may help identify applicants with a history of credit-related difficulties.

**Preprocessing considerations:**
- Treat as a numerical count feature.
- Examine its distribution and frequency of non-zero values.
- Consider scaling depending on the selected algorithm.
- Evaluate its relationship with `pub_rec_bankruptcies` during feature analysis.

**Important:** Correlation with `pub_rec

### Feature: `pub_rec_bankruptcies` – Public Record Bankruptcies

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the number of public record bankruptcies reported for an applicant.
- Has approximately 0.70 correlation with `pub_rec`.
- Contains 535 missing values (0.14%).

**Decision: Retain**

**Reasoning:** Bankruptcy history is a relevant indicator of an applicant's credit history and may contribute to default risk assessment.

**Missing-value treatment:**
- Proposed imputation: Replace missing values with 0.
- Reasoning: Avoid assigning an arbitrary positive bankruptcy count that could incorrectly indicate bankruptcy history.
- Assumption: Missing values are treated as no reported bankruptcies for this initial modelling exercise.
- Limitation: Missing does not necessarily mean zero. Validate this assumption against the source data definitions where possible.

**Multicollinearity consideration:**

`pub_rec_bankruptcies` has approximately 0.70 correlation with `pub_rec`. Retain both initially and evaluate their relationship using VIF and model behaviour, particularly for Logistic Regression.

**Preprocessing note:** Apply imputation during preprocessing using training data only.

### Feature: `mort_acc` – Number of Mortgage Accounts

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the number of mortgage accounts reported in the applicant's credit history.
- Missing values: 37,795 (9.54%).
- Correlation with `total_acc`: approximately 0.38.

**Decision: Retain**

**Reasoning:** Mortgage account history may provide additional information about an applicant's credit profile and existing financial commitments.

**Missing-value treatment:** To be investigated during preprocessing. We should examine the relationship

### Feature: `revol_bal` – Revolving Credit Balance

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents the total outstanding balance on revolving credit accounts.
- The feature contains extreme values that require further investigation.
- It provides information about the applicant's existing revolving debt.

**Decision: Retain**

**Reasoning:** Existing revolving debt is relevant to assessing an applicant's overall financial obligations and potential repayment capacity.

**Preprocessing considerations:**
- Examine the distribution and investigate extreme values.
- Consider a logarithmic transformation if the feature is highly right-skewed and values are non-negative.
- Consider scaling depending on the selected algorithm.
- Do not automatically remove extreme values without investigation.

**Important:** Preserve the original feature in the raw dataset and perform any transformations during preprocessing.

### Feature: `revol_util` – Revolving Credit Utilisation

**Availability at prediction time:** Available, assuming LoanTap obtains the applicant's credit bureau report during application assessment.

**EDA observations:**
- Represents revolving credit utilisation, generally expressed as a percentage.
- Maximum observed value: 892.3, which is suspicious and requires investigation.
- Missing values: 276 (0.07%).

**Decision: Retain**

**Reasoning:** Credit utilisation provides information about how much of the applicant's available revolving credit is currently being used. It may help assess existing financial pressure and credit risk.

**Preprocessing considerations:**
- Investigate the unusually high values, particularly observations above 100%.
- Do not automatically cap or remove these values without understanding their origin.
- Handle the small number of missing values during preprocessing.
- Consider scaling depending on the selected algorithm.

**Important:** Preserve the original values for reference. Any correction, capping or imputation must be documented and applied during preprocessing.

### Feature: `application_type` – Application Type

**Availability at prediction time:** Available

`application_type` indicates whether the loan application was submitted by an individual or jointly by multiple applicants.

**EDA observations:**
- Categories: `Individual` and `Joint`.
- The feature provides information about the application structure.

**Decision: Retain**

**Reasoning:** Individual and joint applications may differ in their financial profiles and repayment capacity. This feature may help the model identify differences in default risk.

**Preprocessing considerations:**
- Treat as a categorical feature.
- Encode the categories during preprocessing.
- Evaluate its relationship with loan default during modelling.

# Feature Availability and Data Leakage Assessment – Derived Credit History Features

## 1. Background

During Exploratory Data Analysis (EDA), we created three derived features to represent the applicant's credit history:

- `credit_history_months`
- `credit_history_years`
- `credit_history_group`

These features were derived using `earliest_cr_line` and `issue_d`.

However, our agreed model prediction point is:

**Immediately after loan application submission, before underwriting, verification, loan approval and issuance.**

Therefore, every feature must be evaluated based on whether it would have been available at that point in time.

## 2. Feature Assessment

| Feature | Description | Assessment |
|---|---|---|
| `credit_history_months` | Duration of credit history in months | Exclude |
| `credit_history_years` | Duration of credit history in years | Exclude |
| `credit_history_group` | Categorised credit history duration | Exclude |
| `earliest_cr_line` | Date of the applicant's oldest recorded credit account | Retain in raw dataset |

## 3. Reason for Exclusion

The three derived features were calculated using `issue_d`, which represents the actual loan issuance date.

The issue date is known only after the loan has passed through the approval and issuance process. It is therefore unavailable at our chosen prediction point.

Using the issuance date to calculate credit-history duration would introduce information from a later stage into a model intended to make predictions at application submission.

This creates a prediction-time data leakage concern.

## 4. Treatment of `earliest_cr_line`

We will retain `earliest_cr_line` in the original dataset.

Under our assumption that LoanTap obtains the applicant's credit bureau report during application assessment, this date would be available at the prediction point.

However, the dataset does not contain the actual loan application date.

Consequently, we cannot accurately calculate the applicant's credit-history duration as of application submission using the available dates.

If the actual application date becomes available, we can derive a new feature:

`credit_history_duration = application_date - earliest_cr_line`

This would represent the applicant's credit history as it existed at the time of application.

## 5. Final Decision

- Exclude `credit_history_months` from the primary model.
- Exclude `credit_history_years` from the primary model.
- Exclude `credit_history_group` from the primary model.
- Retain `earliest_cr_line` in the raw dataset for reference.
- Do not derive a replacement credit-history duration until a valid application date is available.

## 6. Important Data Limitation

The dataset contains the actual loan issuance date (`issue_d`) but does not provide the application submission date.

This limits our ability to reconstruct certain applicant characteristics precisely as they existed at the intended prediction point.

This limitation will be documented as part of the project's assumptions and data constraints.

**Conclusion:** The three derived credit-history features will not be used in the primary model because their calculation relies on a date that is unavailable at the intended prediction point. This decision helps maintain consistency between the model's intended real-world use and the information supplied during model training.

# Final Feature Decision Register
## LoanTap – Personal Loan Default Prediction

### 1. Objective

The purpose of this register is to document the final decisions regarding which features will be used in the primary loan default prediction model.

Every feature has been assessed based on:

- Availability at the intended prediction point.
- Relevance to loan default prediction.
- Data quality and missing values.
- Potential data leakage.
- Redundancy and feature relationships.
- Preprocessing requirements.

### 2. Agreed prediction scenario

**Model prediction point:** Immediately after loan application submission, before underwriting, verification, loan approval and loan issuance.

The model should independently estimate the applicant's probability of default using information available at this point.

**Key assumptions:**
1. Applicant-provided information is available at application submission.
2. LoanTap obtains the applicant's credit bureau report during application assessment.
3. Verification has not yet begun at the prediction point.
4. The actual application date is unavailable in the dataset.
5. The raw dataset will remain unchanged. Exclusions and transformations will be applied to a separate modelling dataset.

### 3. Features retained for the primary model

| Feature | Description | Decision / Preprocessing |
|---|---|---|
| `loan_amnt` | Requested loan amount | Retain; examine distribution and outliers |
| `term` | Applicant-selected repayment term | Retain; categorical encoding |
| `emp_length` | Employment length | Retain; missing values as `Unknown` |
| `home_ownership` | Home ownership status | Retain; group rare categories |
| `log_annual_inc` | Log-transformed annual income | Retain; exclude raw `annual_inc` |
| `purpose` | Loan purpose | Retain; categorical encoding |
| `dti` | Debt-to-income ratio | Retain; investigate extreme values |
| `open_acc` | Number of open credit accounts | Retain; numerical feature |
| `total_acc` | Total number of credit accounts | Retain; examine relationship with `open_acc` |
| `pub_rec` | Derogatory public records | Retain; numerical count |
| `pub_rec_bankruptcies` | Public record bankruptcies | Retain; proposed missing-value imputation with 0 |
| `mort_acc` | Number of mortgage accounts | Retain; missing-value treatment pending |
| `revol_bal` | Revolving credit balance | Retain; investigate extreme values |
| `revol_util` | Revolving credit utilisation | Retain; investigate suspicious values and handle missing values |
| `application_type` | Individual or joint application | Retain; categorical encoding |

**Important assumptions and considerations:**
- `pub_rec_bankruptcies`: Zero imputation is a working assumption, not a confirmed interpretation of missing values. Validate against the source definition where possible.
- `pub_rec` and `pub_rec_bankruptcies`: Retain both initially. Evaluate multicollinearity, particularly for Logistic Regression.
- `open_acc` and `total_acc`: Retain both initially and evaluate their individual contributions.
- `mort_acc`: Do not assume missing means zero.
- `home_ownership`: Proposed grouping of `OTHER`, `NONE` and `ANY` into `Other`.

### 4. Features excluded from the primary model

| Feature | Reason for exclusion |
|---|---|
| `int_rate` | Interest rate is determined during loan pricing/underwriting and is unavailable at initial application submission |
| `installment` | Monthly installment is calculated using proposed loan terms and interest rate; unavailable under the strict initial application prediction scenario |
| `grade` | Lender-assigned risk classification; represents an existing risk assessment |
| `sub_grade` | More granular lender-assigned risk classification |
| `verification_status` | Verification takes place during underwriting, after our prediction point |
| `initial_list_status` | Investor-side loan listing classification, not applicant-provided information |
| `emp_title` | Very high cardinality and inconsistent free-text values |
| `title` | Applicant-provided free-text description; less structured than `purpose` |
| `address` | Extremely high cardinality, privacy concerns and poor generalisation potential |
| `annual_inc` | Raw income excluded in favour of `log_annual_inc` |
| `issue_year` | Derived from actual loan issuance date, unavailable at application submission |
| `credit_history_months` | Calculated using actual issuance date |
| `credit_history_years` | Calculated using actual issuance date |
| `credit_history_group` | Derived from credit history duration calculated using actual issuance date |

### 5. Features retained only for EDA or reference

| Feature | Treatment | Reason |
|---|---|---|
| `loan_status` | Retain as target variable, not a predictor | Outcome to be predicted |
| `issue_d` | Retain for EDA/reference, exclude as predictor | Actual loan issuance date; unavailable at application submission |
| `issue_year` | Retain for historical analysis, exclude as predictor | Derived from actual issuance date |
| `earliest_cr_line` | Retain in raw dataset for reference; not included as a predictor in the current model | Application date is missing, so valid application-time credit history duration cannot be calculated |
| `annual_inc` | Retain in raw dataset for reference and EDA | Use `log_annual_inc` as the model representation |
| `emp_title`, `title`, `address` | Preserve in raw dataset | Excluded from primary model; may support other analyses |

### 6. Target variable treatment

The target variable is `loan_status`.

| Original value | Binary encoding |
|---|---:|
| Fully Paid | 0 |
| Charged Off | 1 |

The target must be separated from predictor variables before preprocessing.

### 7. Important data limitations

**Missing application date**

The dataset contains `issue_d`, the actual issuance date, but not the application submission date. This prevents accurate calculation of credit-history duration at the intended prediction point.

**Credit bureau availability**

The use of credit bureau variables assumes LoanTap obtains the report during application assessment. This assumption should be validated against the intended real-world workflow.

**Missing-value assumptions**

The decision to impute missing `pub_rec_bankruptcies` values with zero requires validation. Other missing-value treatments, particularly for `mort_acc`, will be decided during preprocessing.

### 8. Preprocessing principles

1. Preserve the original raw dataset.
2. Create a separate modelling dataset containing only approved predictors and the target.
3. Handle missing values using documented strategies.
4. Investigate outliers before deciding on treatment.
5. Apply categorical encoding where required.
6. Scale numerical features depending on the selected algorithm.
7. Fit preprocessing transformations using training data only to avoid data leakage.
8. Evaluate multicollinearity for models sensitive to correlated predictors.
9. Keep the test dataset untouched until model evaluation.

### 9. Final decision

The 15 features listed in Section 3 form our initial predictor shortlist. `loan_status` is the target variable.

Excluded features will remain in the original dataset wherever appropriate, but will not be passed into the primary model.

These decisions establish the feature-selection baseline for the next phase: data preprocessing and preparation for modelling.

The feature shortlist may be revisited if later validation reveals data-quality issues, redundancy or poor predictive contribution.

# LoanTap Personal Loan Default Prediction
## Modelling – Baseline Logistic Regression

### 1. Objective of the modelling stage

The objective is to develop a classification model that estimates the probability that a personal loan applicant will default.

The target variable is `loan_status`, encoded as:

- `0` – Fully Paid
- `1` – Charged Off

The model is intended to support LoanTap's underwriting and risk assessment process by identifying applicants who may have a higher probability of default.

The prediction point was defined as immediately after loan application submission, before underwriting, verification, approval and loan issuance. Accordingly, features that would only become available after underwriting or loan issuance were excluded from the primary modelling dataset.

### 2. Modelling dataset

The modelling dataset was prepared using the feature-selection and preprocessing decisions documented in the earlier sections.

The modelling predictors included:

- Loan and applicant information: `loan_amnt`, `term`, `emp_length`, `home_ownership`, `log_annual_inc`, `purpose`, `application_type`
- Debt and credit information: `dti`, `open_acc`, `total_acc`, `pub_rec`, `pub_rec_bankruptcies`, `mort_acc`, `revol_bal`, `revol_util`
- Missingness information: `mort_acc_missing`

The target variable was `loan_status`.

The final encoded predictor dataset contained 41 columns after one-hot encoding.

### 3. Train-test split

The data was divided into training and testing sets using an 80:20 split.

Stratification was applied using the target variable to preserve the original class distribution in both datasets.

| Dataset | Records |
|---|---:|
| Training | 316,824 |
| Testing | 79,206 |
| Total | 396,030 |

Target distribution:

| Target | Meaning | Approximate proportion |
|---|---|---:|
| 0 | Fully Paid | 80.39% |
| 1 | Charged Off | 19.61% |

The test dataset was kept separate from model training and preprocessing.

### 4. Feature encoding and scaling

Categorical predictors were converted into numerical variables using one-hot encoding.

The categorical variables included:

- `term`
- `emp_length`
- `home_ownership`
- `purpose`
- `application_type`

The training and testing datasets were aligned to ensure that both contained the same encoded columns.

Continuous numerical features were standardised using `StandardScaler`.

The scaler was fitted only on the training data and then used to transform the test data. This avoided data leakage from the test set into the training process.

The continuous features scaled were:

`loan_amnt`, `log_annual_inc`, `dti`, `open_acc`, `total_acc`, `pub_rec`, `pub_rec_bankruptcies`, `mort_acc`, `revol_bal`, `revol_util`.

Binary indicator and one-hot encoded columns were not standardised.

### 5. Baseline Logistic Regression model

Logistic Regression was selected as the first baseline classification algorithm.

It estimates the probability of a borrower belonging to the Charged Off class.

The model was trained using:

- Maximum iterations: 1,000
- Random state: 42
- Default classification threshold: 0.50

The model converged after 59 iterations.

The training accuracy was approximately 80.53%.

### 6. Baseline model evaluation

The model was evaluated on the test dataset.

#### 6.1 Confusion matrix

| Actual / Predicted | Fully Paid (0) | Charged Off (1) |
|---|---:|---:|
| Fully Paid (0) | 63,127 | 544 |
| Charged Off (1) | 14,866 | 669 |

Interpretation:

- 63,127 fully paid loans were correctly classified.
- 544 fully paid loans were incorrectly flagged as charged off.
- 669 charged-off loans were correctly identified.
- 14,866 charged-off loans were incorrectly classified as fully paid.

The model identified only a small proportion of actual defaults at the default threshold of 0.50.

#### 6.2 Classification report

| Metric | Fully Paid (0) | Charged Off (1) |
|---|---:|---:|
| Precision | 0.81 | 0.55 |
| Recall | 0.99 | 0.04 |
| F1-score | 0.89 | 0.08 |
| Support | 63,671 | 15,535 |

Overall accuracy: **80.54%**

Interpretation:

- The recall for Fully Paid loans is 99%, meaning the model identifies almost all fully paid loans.
- The recall for Charged Off loans is only 4%, meaning the model identifies very few actual defaults.
- The model's default-class precision is 55%, but this is based on a relatively small number of predicted defaults.
- The overall accuracy is not sufficient to judge model quality because the target classes are imbalanced.

A simple model that predicts every loan as Fully Paid would already achieve approximately 80.39% accuracy. Therefore, the baseline model's 80.54% accuracy represents only a marginal improvement over that naive benchmark.

#### 6.3 ROC-AUC

ROC-AUC: **0.6879**

ROC-AUC measures how well the model ranks positive cases above negative cases across different classification thresholds.

A score of 0.6879 indicates modest discriminatory ability. The model has some capacity to distinguish between Fully Paid and Charged Off loans, but its ranking performance is not particularly strong.

### 7. Baseline model assessment

The initial Logistic Regression model has an important limitation: it predicts very few borrowers as likely defaulters when using the default threshold of 0.50.

This is reflected in the Charged Off recall of only 4%.

However, this does not necessarily mean that the model cannot identify higher-risk borrowers. The default threshold of 0.50 may be too conservative for this imbalanced classification problem.

The probability estimates should therefore be examined at different classification thresholds.

### 8. Next modelling activity: threshold optimisation

The next step is to investigate alternative classification thresholds using a validation dataset.

The test dataset must not be used to select the threshold. Threshold selection will be performed using a validation split created from the existing training dataset.

The planned process is:

1. Split the training dataset into a sub-training set and a validation set.
2. Fit a fresh scaler using only the sub-training data.
3. Train a separate Logistic Regression model on the sub-training data.
4. Generate default probabilities for the validation data.
5. Evaluate different thresholds using precision, recall, F1-score and confusion matrices.
6. Select a threshold based on the business trade-off between identifying defaulters and incorrectly flagging borrowers who would repay.

After selecting a threshold, the final model will be trained using the full original training dataset and evaluated on the test dataset.

The test dataset will remain excluded from threshold selection.

### 9. Current conclusion

The first Logistic Regression model establishes a baseline for the LoanTap default prediction problem.

The model achieves 80.54% accuracy and an ROC-AUC of 0.6879. However, its default-class recall is only 4% at the default threshold of 0.50.

The next priority is to investigate whether a more suitable classification threshold can improve default identification while maintaining an acceptable level of precision.

No final model-selection or deployment decision has been made at this stage.

# LoanTap Personal Loan Default Prediction
## Modelling – Logistic Regression: Baseline, Threshold Selection and Final Evaluation

### 1. Modelling objective

Develop a binary classification model to estimate the probability that a personal loan applicant will default.

**Target variable:** `loan_status`

- 0 – Fully Paid
- 1 – Charged Off

**Prediction point:** Immediately after application submission, before underwriting, verification, approval and loan issuance.

The objective is to identify potentially high-risk applicants while balancing the identification of actual defaulters against incorrectly flagging borrowers who would repay.

### 2. Modelling dataset

The modelling predictors were:

`loan_amnt`, `term`, `emp_length`, `home_ownership`, `log_annual_inc`, `purpose`, `dti`, `open_acc`, `total_acc`, `pub_rec`, `pub_rec_bankruptcies`, `mort_acc`, `revol_bal`, `revol_util`, `application_type`, `mort_acc_missing`.

The target was `loan_status`.

After one-hot encoding, the predictor dataset contained 41 columns.

### 3. Train-test split

The original dataset was divided using an 80:20 stratified split.

| Dataset | Records |
|---|---:|
| Training | 316,824 |
| Testing | 79,206 |
| Total | 396,030 |

Target distribution:

| Target | Meaning | Proportion |
|---|---|---:|
| 0 | Fully Paid | 80.39% |
| 1 | Charged Off | 19.61% |

Stratification preserved the target distribution in both datasets.

### 4. Preprocessing

Categorical predictors were converted using one-hot encoding with `drop_first=True`.

The categorical variables were:

- `term`
- `emp_length`
- `home_ownership`
- `purpose`
- `application_type`

Continuous numerical predictors were standardised using `StandardScaler`.

The scaler was fitted on the training data and applied to the test data. Binary indicators and one-hot encoded columns were not scaled.

The 10 continuous predictors were:

`loan_amnt`, `log_annual_inc`, `dti`, `open_acc`, `total_acc`, `pub_rec`, `pub_rec_bankruptcies`, `mort_acc`, `revol_bal`, `revol_util`.

### 5. Baseline Logistic Regression

A Logistic Regression model was trained with:

- `max_iter=1000`
- `random_state=42`
- Default classification threshold = 0.50

The model converged after 59 iterations.

Training accuracy: approximately 80.53%.

#### 5.1 Baseline test evaluation

Confusion matrix:

| Actual / Predicted | Fully Paid | Charged Off |
|---|---:|---:|
| Fully Paid | 63,127 | 544 |
| Charged Off | 14,866 | 669 |

Classification metrics:

| Metric | Fully Paid | Charged Off |
|---|---:|---:|
| Precision | 0.81 | 0.55 |
| Recall | 0.99 | 0.04 |
| F1-score | 0.89 | 0.08 |

- Test accuracy: 80.54%
- ROC-AUC: 0.6879

**Interpretation:**

The baseline model identified only 4.14% of actual defaulters at the default threshold of 0.50. Although accuracy was approximately 80.54%, this was only marginally higher than the 80.39% accuracy achievable by predicting every loan as Fully Paid.

ROC-AUC of 0.6879 indicated modest discriminatory ability.

The baseline therefore demonstrated that accuracy alone was not a suitable measure of success for this imbalanced classification problem.

### 6. Threshold optimisation using validation data

#### 6.1 Validation split

To investigate alternative classification thresholds without using the test dataset, the original training data was further divided into:

| Dataset | Records | Shape |
|---|---:|---|
| Sub-training | 253,459 | (253459, 41) |
| Validation | 63,365 | (63365, 41) |

Both datasets retained the same class distribution:

- Fully Paid: 80.39%
- Charged Off: 19.61%

A fresh `StandardScaler` was fitted only on the sub-training data and applied to the validation data.

A separate Logistic Regression model was trained on the sub-training data. It converged after 64 iterations.

The validation model generated default probabilities using `predict_proba()`.

#### 6.2 Threshold experiments

Thresholds between 0.10 and 0.50 were evaluated using:

- Precision
- Recall
- F1-score
- True positives
- False positives
- True negatives
- False negatives

The initial threshold comparison showed that reducing the threshold substantially increased Charged Off recall.

Selected results:

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.10 | 0.2226 | 0.9370 | 0.3597 |
| 0.15 | 0.2621 | 0.7891 | 0.3935 |
| 0.20 | 0.3048 | 0.6106 | 0.4066 |
| 0.25 | 0.3483 | 0.4571 | 0.3953 |
| 0.30 | 0.3874 | 0.3270 | 0.3546 |
| 0.50 | 0.5593 | 0.0414 | 0.0770 |

A finer search between 0.15 and 0.25 was then performed.

The exact F1-score comparison for the two closest rounded candidates was:

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.19 | 0.29655885 | 0.64628259 | 0.40656003 |
| 0.20 | 0.30481240 | 0.61055681 | 0.40662344 |

A further search using `precision_recall_curve()` evaluated all distinct validation probability thresholds.

#### 6.3 Selected threshold

The maximum-F1 threshold found was:

**0.1909335296 (approximately 0.19093)**

Validation metrics at this threshold:

| Metric | Result |
|---|---:|
| Precision – Charged Off | 29.78% |
| Recall – Charged Off | 64.40% |
| F1-score – Charged Off | 0.4073 |
| Validation accuracy | Approximately 63% |
| Applicants flagged | 26,875 |
| Percentage flagged | 42.41% |

Validation confusion matrix:

| Actual / Predicted | Fully Paid | Charged Off |
|---|---:|---:|
| Fully Paid | 32,066 | 18,871 |
| Charged Off | 4,424 | 8,004 |

**Threshold-selection objective:** Balance precision and recall, using F1-score as the primary selection guide.

**Decision:** Select the threshold of approximately 0.19093 because it produced the highest observed F1-score on validation data.

This is a modelling decision rather than a confirmed LoanTap business policy. The actual financial costs of missed defaults and false positives, as well as operational review capacity, have not been quantified.

### 7. Final evaluation on the test dataset

The selected threshold was applied to the original Logistic Regression model trained on the full training dataset.

The test dataset was not used to select or modify the threshold.

#### 7.1 Test confusion matrix

| Actual / Predicted | Fully Paid | Charged Off |
|---|---:|---:|
| Fully Paid | 40,096 | 23,575 |
| Charged Off | 5,565 | 9,970 |

#### 7.2 Test classification metrics

| Metric | Fully Paid | Charged Off |
|---|---:|---:|
| Precision | 0.88 | 0.30 |
| Recall | 0.63 | 0.64 |
| F1-score | 0.73 | 0.41 |
| Support | 63,671 | 15,535 |

Additional metrics:

| Metric | Result |
|---|---:|
| Test accuracy | 63.21% |
| Test ROC-AUC | 0.68785 |
| Applicants flagged | 33,545 |
| Percentage flagged | 42.35% |

#### 7.3 Interpretation of final results

The model identified 9,970 of the 15,535 actual Charged Off loans, giving a default recall of approximately 64.18%.

It missed 5,565 actual defaulters.

It also flagged 23,575 Fully Paid borrowers as potential defaulters. These are false positives.

Of the 33,545 applicants flagged by the model, approximately 29.72% actually defaulted.

The selected threshold therefore increases default identification substantially compared with the baseline threshold, but results in a significant number of false positives.

The flagged population represents 42.35% of the test dataset. This is a meaningful operational consideration if flagged applicants require manual review.

### 8. Comparison of validation and test performance

| Metric | Validation | Test |
|---|---:|---:|
| Charged Off precision | 29.78% | 29.72% |
| Charged Off recall | 64.40% | 64.18% |
| Charged Off F1-score | 0.4073 | Approximately 0.409 |
| Applicants flagged | 42.41% | 42.35% |
| Accuracy | Approximately 63% | 63.21% |
| ROC-AUC | Not separately reported | 0.68785 |

The validation and test results are close, indicating reasonably consistent performance at the selected threshold.

### 9. Understanding the change in accuracy

The baseline model achieved 80.54% test accuracy at a threshold of 0.50.

The model achieved 63.21% test accuracy at the selected threshold of 0.19093.

This reduction is expected because lowering the threshold causes more applicants to be classified as potential defaulters.

The objective of threshold selection was to improve default identification, not maximise accuracy.

ROC-AUC remained approximately 0.6879 because changing the classification threshold does not change the underlying probability rankings. It changes only the conversion of probabilities into class predictions.

### 10. Business interpretation and limitations

The model can be considered a preliminary risk-screening tool, rather than an automatic loan rejection mechanism.

Potential uses include:

- Identifying applicants for additional risk assessment.
- Prioritising manual underwriting review.
- Supporting risk segmentation.
- Providing an estimated default probability to assist decision-making.

Important limitations:

1. Default precision is approximately 30%, so many flagged borrowers would have repaid.
2. Approximately 36% of actual defaulters are still missed.
3. The flagged population is approximately 42% of applicants, which could create a substantial review workload.
4. ROC-AUC of approximately 0.688 indicates only modest ranking performance.
5. The F1-based threshold does not incorporate actual financial loss, loan exposure or operational review costs.
6. Additional model development and comparison are required before considering deployment.

### 11. Current conclusion

The baseline Logistic Regression model demonstrated poor default recall at the default threshold of 0.50.

Threshold optimisation using validation data identified 0.19093 as the maximum-F1 operating point.

When applied to the independent test dataset, the model achieved approximately:

- 29.72% Charged Off precision
- 64.18% Charged Off recall
- 0.409 Charged Off F1-score
- 0.68785 ROC-AUC

The validation and test results are consistent, suggesting that the threshold selection generalises reasonably well.

However, the model's modest ROC-AUC and high false-positive count indicate that further improvements are needed.

**Status:** Baseline Logistic Regression, validation threshold selection and final test evaluation completed. Further model development and comparison remain pending.


