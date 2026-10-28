# IBM HR Analytics Employee Attrition — Data Dictionary

**Dataset Reference:** IBM HR Analytics Employee Attrition & Performance  
**Total Records:** 1,470 employees  
**Total Raw Attributes:** 35 columns  
**Target Variable:** `Attrition` (Binary: Yes / No)  
**Cleaned Modeling Features:** 30 features (23 numerical + 7 categorical)  
**Dropped Columns:** 4 columns (1 identifier + 3 constant columns)

---

## 1. Target Variable

| Attribute | Data Type | Description | Values / Domain | Modeling Role |
| :--- | :--- | :--- | :--- | :--- |
| **Attrition** | Categorical (String) | Indicates whether the employee left the organization | `Yes` (16.12%, 237 records)<br>`No` (83.88%, 1233 records) | **Target Variable** (mapped to 1 for Yes, 0 for No) |

---

## 2. Demographic & Personal Characteristics

| Attribute | Data Type | Description | Values / Range | Preprocessing / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Age** | Numerical (Integer) | Age of the employee in years | 18 to 60 | Scaled via `StandardScaler` |
| **Gender** | Categorical (String) | Gender of the employee | `Male`, `Female` | One-Hot Encoded (`OneHotEncoder`) |
| **MaritalStatus** | Categorical (String) | Marital status of the employee | `Single`, `Married`, `Divorced` | One-Hot Encoded |
| **Education** | Numerical (Ordinal) | Education level of the employee | `1`: Below College<br>`2`: College<br>`3`: Bachelor<br>`4`: Master<br>`5`: Doctor | Scaled via `StandardScaler` |
| **EducationField** | Categorical (String) | Field of formal education | `Life Sciences`, `Medical`, `Marketing`, `Technical Degree`, `Human Resources`, `Other` | One-Hot Encoded |
| **DistanceFromHome** | Numerical (Integer) | Commute distance from home to workplace (in miles) | 1 to 29 | Scaled via `StandardScaler` |

---

## 3. Job Role & Organizational Placement

| Attribute | Data Type | Description | Values / Range | Preprocessing / Role |
| :--- | :--- | :--- | :--- | :--- |
| **Department** | Categorical (String) | Operating department within the company | `Sales`, `Research & Development`, `Human Resources` | One-Hot Encoded |
| **JobRole** | Categorical (String) | Specific organizational role | `Sales Executive`, `Research Scientist`, `Laboratory Technician`, `Manufacturing Director`, `Healthcare Representative`, `Manager`, `Sales Representative`, `Research Director`, `Human Resources` | One-Hot Encoded |
| **JobLevel** | Numerical (Ordinal) | Hierarchical job seniority level | 1 (Entry) to 5 (Executive) | Scaled via `StandardScaler` |
| **BusinessTravel** | Categorical (String) | Frequency of company-required travel | `Non-Travel`, `Travel_Rarely`, `Travel_Frequently` | One-Hot Encoded |
| **OverTime** | Categorical (String) | Whether employee works beyond contracted hours | `Yes`, `No` | One-Hot Encoded |

---

## 4. Compensation & Financial Metrics

| Attribute | Data Type | Description | Values / Range | Preprocessing / Role |
| :--- | :--- | :--- | :--- | :--- |
| **MonthlyIncome** | Numerical (Integer) | Gross monthly salary in USD | \$1,009 to \$19,999 | Scaled via `StandardScaler` |
| **DailyRate** | Numerical (Integer) | Daily billing or compensation rate | \$102 to \$1,499 | Scaled via `StandardScaler` |
| **HourlyRate** | Numerical (Integer) | Hourly billing rate | \$30 to \$100 | Scaled via `StandardScaler` |
| **MonthlyRate** | Numerical (Integer) | Monthly allocated rate | \$2,094 to \$26,999 | Scaled via `StandardScaler` |
| **PercentSalaryHike** | Numerical (Integer) | Percentage salary increase at last review | 11% to 25% | Scaled via `StandardScaler` |
| **StockOptionLevel** | Numerical (Ordinal) | Company equity / stock option tier | 0 (None) to 3 (Highest) | Scaled via `StandardScaler` |

---

## 5. Employee Satisfaction & Engagement (Likert Scales)

| Attribute | Data Type | Description | Values / Scale | Preprocessing / Role |
| :--- | :--- | :--- | :--- | :--- |
| **EnvironmentSatisfaction** | Numerical (Ordinal) | Satisfaction with workplace physical/cultural environment | 1: Low, 2: Medium, 3: High, 4: Very High | Scaled via `StandardScaler` |
| **JobSatisfaction** | Numerical (Ordinal) | Direct satisfaction with assigned role | 1: Low, 2: Medium, 3: High, 4: Very High | Scaled via `StandardScaler` |
| **JobInvolvement** | Numerical (Ordinal) | Degree of active involvement & ownership in work | 1: Low, 2: Medium, 3: High, 4: Very High | Scaled via `StandardScaler` |
| **RelationshipSatisfaction** | Numerical (Ordinal) | Satisfaction with workplace colleagues & peers | 1: Low, 2: Medium, 3: High, 4: Very High | Scaled via `StandardScaler` |
| **WorkLifeBalance** | Numerical (Ordinal) | Perceived balance between work commitments and personal life | 1: Bad, 2: Good, 3: Better, 4: Best | Scaled via `StandardScaler` |
| **PerformanceRating** | Numerical (Ordinal) | Performance evaluation rating | 3: Excellent, 4: Outstanding (ratings 1 & 2 not present in data) | Scaled via `StandardScaler` |

---

## 6. Career Tenure & Work History

| Attribute | Data Type | Description | Values / Range | Preprocessing / Role |
| :--- | :--- | :--- | :--- | :--- |
| **TotalWorkingYears** | Numerical (Integer) | Cumulative career working experience | 0 to 40 years | Scaled via `StandardScaler` |
| **NumCompaniesWorked** | Numerical (Integer) | Number of prior employers | 0 to 9 | Scaled via `StandardScaler` |
| **YearsAtCompany** | Numerical (Integer) | Total tenure at current employer | 0 to 40 years | Scaled via `StandardScaler` |
| **YearsInCurrentRole** | Numerical (Integer) | Years in current assigned role | 0 to 18 years | Scaled via `StandardScaler` |
| **YearsSinceLastPromotion** | Numerical (Integer) | Years elapsed since last formal promotion | 0 to 15 years | Scaled via `StandardScaler` |
| **YearsWithCurrManager** | Numerical (Integer) | Years working under current direct manager | 0 to 17 years | Scaled via `StandardScaler` |
| **TrainingTimesLastYear** | Numerical (Integer) | Formal training sessions attended in previous year | 0 to 6 | Scaled via `StandardScaler` |

---

## 7. Dropped Attributes (Justification)

| Attribute | Data Type | Value Distribution | Reason for Removal |
| :--- | :--- | :--- | :--- |
| **EmployeeNumber** | Numerical (Integer) | 1 to 2068 (arbitrary sequence) | Unique identifier; holds no generalizable relationship to attrition behavior. |
| **EmployeeCount** | Numerical (Integer) | Constant = 1 for all 1,470 records | Zero variance ($\sigma^2 = 0$); carries zero predictive information. |
| **Over18** | Categorical (String) | Constant = `Y` for all 1,470 records | Zero variance; legal compliance field with no discriminatory value. |
| **StandardHours** | Numerical (Integer) | Constant = 80 for all 1,470 records | Zero variance; fixed bi-weekly pay standard for all records. |
