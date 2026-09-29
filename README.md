# District-Level Anemia Analysis in India Using NFHS-5 Data

## Team Information

**Team Number:** TEAM-5

### Team Members

| S.No |      Name         |   Roll Number   |          Responsibilities             |
|------|-------------------|-----------------|---------------------------------------|
1.       M.N.B. Sasirekha      25B11CS528	    Data Analysis and Machine learning 
2.       P. Naga Maheswari     25B11CS691       Data visualization and documentation
3.       M. Pavan              25B11CS535       Data extraction and cleaning
4.       M. Yaswanth           25B11CS527	    Data loading and preprocessing 

---

## 1. Project Overview

Anemia is an important public health concern, particularly among women and children. This project analyzes district-level health and socioeconomic indicators using data from the National Family Health Survey (NFHS-5).

The project investigates the prevalence of anemia among women and children and examines its relationship with factors such as women's literacy, schooling, teenage pregnancy, iron-folic acid consumption, institutional births, low BMI, sanitation, clean cooking fuel, and health insurance coverage.

The project follows a complete data analysis pipeline, beginning with dataset loading and preprocessing and continuing through exploratory analysis, statistical analysis, visualization, risk-score generation, and machine learning.

## 2. Problem Statement

The objective of this project is to analyze district-level anemia prevalence in India and identify socioeconomic, nutritional, healthcare, and household factors associated with anemia.

The analysis aims to answer questions such as:

- What is the prevalence of anemia among women across districts?
- How does anemia prevalence vary between states?
- Which districts have higher levels of anemia?
- What factors are associated with women's anemia?
- How do indicators such as literacy, schooling, IFA consumption, BMI, sanitation, and health insurance relate to anemia?
- Can machine learning models be used to analyze/predict women's anemia prevalence based on selected indicators?

## 3. Objectives

1. Collect and load the NFHS-5 dataset.
2. Inspect the structure and quality of the dataset.
3. Handle missing and invalid values.
4. Remove or identify duplicate records where applicable.
5. Filter relevant variables for the analysis.
6. Rename columns for easier analysis.
7. Extract location, anemia, and contributing-factor data.
8. Perform grouping, sorting, filtering, and aggregation using Pandas.
9. Analyze district- and state-level anemia prevalence.
10. Calculate a composite anemia risk score.
11. Visualize important patterns and comparisons.
12. Apply a machine learning model using selected indicators.
13. Summarize findings, limitations, and conclusions.

## 4. Scope of the Project

The project focuses on district-level analysis of anemia-related indicators in India using NFHS-5 data.

The analysis includes:

- Women aged 15–49 years
- Adolescent women
- Children aged 6–59 months
- Pregnant women
- Non-pregnant women
- Women's literacy and schooling
- Iron-folic acid consumption
- Institutional births
- BMI
- Sanitation
- Clean cooking fuel
- Health insurance coverage

The project is intended for data analysis and educational purposes.

## 5. Dataset

### Dataset Source

The project uses data from the National Family Health Survey (NFHS-5), which provides district-level health and demographic indicators for India.

**Dataset:** NFHS-5_DATASET

**Source:** "https://www.kaggle.com/datasets/kmldas/india-national-family-health-survey-nfhs"

> Replace the link above with the exact source URL used by the team.

### Dataset Description

The dataset contains district-level indicators covering demographic, health, nutrition, education, sanitation, and household characteristics.

Important variables used in this project include:

| Variable | Description |
|----------|-------------|
| District | District name |
| State | State/UT name |
| Women_Literacy | Percentage of literate women aged 15–49 |
| Women_Schooling | Percentage of women aged 15–49 with 10+ years of schooling |
| Teenage_Pregnancy | Percentage of women aged 15–19 who were already mothers or pregnant |
| IFA_Consumption | Percentage of mothers consuming IFA for 100+ days |
| Institutional_Births | Percentage of institutional births |
| Low_BMI | Percentage of women with BMI below normal |
| Improved_Sanitation | Percentage of population using improved sanitation |
| Clean_Cooking_Fuel | Percentage of households using clean cooking fuel |
| Health_Insurance | Percentage of households covered by health insurance/financing |
| Children_Anemia | Percentage of children aged 6–59 months who are anaemic |
| NonPregnant_Anemia | Percentage of non-pregnant women aged 15–49 who are anaemic |
| Pregnant_Anemia | Percentage of pregnant women aged 15–49 who are anaemic |
| Women_Anemia | Percentage of all women aged 15–49 who are anaemic |
| Teen_Women_Anemia | Percentage of women aged 15–19 who are anaemic |

## 6. Project Pipeline

```text
Data Loading
      ↓
Data Acquisition & Filtering
      ↓
Data Validation & Cleaning
      ↓
Data Extraction
      ↓
Data Aggregation & Representation
      ↓
Data Analysis
      ↓
Machine Learning
      ↓
Data Visualization
```

## 7. Project Files

| Notebook | Purpose |
|----------|---------|
| 01_DataLoading.ipynb | Loads and initially inspects the dataset |
| 02_Data_Acquisition_and_Filtering.ipynb | Selects relevant data and variables |
| 03_Data_Validation_and_Cleaning.ipynb | Validates and cleans the dataset |
| 04_Data_Extraction.ipynb | Extracts location, anemia, and factor datasets |
| 05_Data_Aggregation_and_Representation.ipynb | Performs grouping, aggregation, and risk-score analysis |
| 06_Data_Analysis.ipynb | Performs statistical and exploratory analysis |
| 07_ML_Model.ipynb | Applies machine learning |
| 08_Data_Visualization.ipynb | Creates charts and visual representations |

## 8. Data Preprocessing

The preprocessing stage includes:

- Checking dataset dimensions
- Inspecting column names
- Checking data types
- Identifying missing values
- Handling invalid numeric values
- Removing unnecessary columns
- Converting numerical columns to appropriate numeric types
- Renaming columns for easier analysis
- Saving the cleaned dataset

## 9. Data Analysis

Pandas operations used in the project include:

- `head()`
- `info()`
- `describe()`
- `isnull()`
- `groupby()`
- `agg()`
- `sort_values()`
- `value_counts()`
- Data filtering
- Column selection
- Correlation analysis

The analysis compares anemia prevalence across districts and states and examines relationships between anemia and selected socioeconomic, nutritional, and healthcare indicators.

## 10. Risk Score

A composite Risk Score is calculated using selected anemia-related indicators.

The score incorporates:

- Women's anemia prevalence
- Low BMI
- Improved sanitation
- Women's literacy
- IFA consumption

The resulting score is used to rank districts and categorize them according to relative risk levels.

The Risk Score is an analytical indicator created for this project and should not be interpreted as a clinical diagnosis or official government risk classification.

## 11. Machine Learning

The machine learning stage uses selected socioeconomic, nutritional, healthcare, and demographic indicators as features.

### Features

- Women_Literacy
- Women_Schooling
- Teenage_Pregnancy
- IFA_Consumption
- Institutional_Births
- Low_BMI
- Improved_Sanitation
- Clean_Cooking_Fuel
- Health_Insurance

### Target

- Women_Anemia

The machine learning notebook contains the model implementation and evaluation.

## 12. Data Visualization

The project includes visualizations showing:

- Overall anemia prevalence
- State-wise anemia prevalence
- District-level anemia prevalence
- Risk Score distribution
- Risk categories
- Relationships between selected indicators

Each visualization includes appropriate titles and axis labels.

## 13. Key Findings

The major findings from the analysis will be documented here after the final execution of the notebooks.

> Update this section with the actual numerical findings obtained from the final notebook execution.

## 14. Results

The generated charts, tables, and analysis outputs are available in the `results/` directory.

## 15. Limitations

1. The analysis is based on available NFHS-5 district-level data.
2. The analysis identifies associations and patterns and does not by itself establish causation.
3. The Risk Score is a project-specific analytical measure and is not an official medical classification.
4. Dataset limitations and missing information may affect the analysis.
5. Machine learning performance depends on the available variables and dataset quality.

## 16. Conclusion

This project demonstrates a complete data analysis workflow using NFHS-5 district-level data.

The project covers data collection, preprocessing, validation, extraction, aggregation, statistical analysis, visualization, risk-score generation, and machine learning.

The analysis provides a data-driven view of anemia prevalence and its relationship with selected socioeconomic, nutritional, healthcare, and household indicators.

## 17. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Google Colab
- GitHub

## 18. How to Run the Project

### Step 1: Clone the repository

```bash
git clone [YOUR_GITHUB_REPOSITORY_URL]
cd Anemia-Analysis-NFHS
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Open the notebooks

The notebooks can be opened using Jupyter Notebook, JupyterLab, or Google Colab.

### Step 4: Run the notebooks in order

```text
01_DataLoading
02_Data_Acquisition_and_Filtering
03_Data_Validation_and_Cleaning
04_Data_Extraction
05_Data_Aggregation_and_Representation
06_Data_Analysis
07_ML_Model
08_Data_Visualization
```

## 19. Presentations

The Review-1 and Review-2 presentations are included in the `presentations/` folder.

## 20. Team Contributions

Each team member contributed to different components of the project.

The detailed responsibilities are listed in the Team Information section above.

## 21. Academic Project

This repository was created as part of the **Data Analysis Essentials** course/project.
