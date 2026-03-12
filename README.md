# Healthcare Insurance Cost Analysis

**Healthcare Insurance Cost Analysis** is a comprehensive data analysis project designed to identify key factors influencing healthcare insurance premiums and provide data-driven insights for cost prediction and risk assessment. The project uses  data science tools such as Feature Engineering, descriptive and predictive statistical analysis and Statistical Correlation methods to uncover patterns in insurance costs across personal attributes such as age and BMI (Body Mass Index) and geographical variables.

# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)


## Dataset Content

The dataset comprises 1,338 healthcare insurance records with the following characteristics:

* **Size**: 1,337 records (rows) (after duplicate removal) with 7 variables
* **Data Types**: 4 numerical variables and 3 categorical variables
* **Numerical Variables**: 
  - Age (measured in years)
  - BMI (Body Mass Index)
  - Children (number of children covered)
  - Charges (annual healthcare insurance premium in dollars)
* **Categorical Variables**: 
  - Sex (male/female)
  - Smoker (yes/no)
  - Region (northeast, northwest, southeast, southwest)
* **Data Quality**: 
  - Zero missing values
  - 2 duplicate records identified and removed
  - Significant outliers detected in BMI and Charges variables
* **Source**: Insurance dataset from public health analytics repository (insurance.csv).


## Business Requirements

1. **Identify Primary Cost Drivers**: Determine which variables (age, BMI, smoking status, family size, demographics) have the strongest influence on insurance premium costs.
2. **Understand Price Variation Across Different facets of the general population**: Analyze how insurance charges vary across different customer segments (sex, geographical region)
3. **Detect Outliers and Data Inadequacies**: Identify and handle extreme outlier values in healthcare costs (charges) and BMI (bmi) to ensure data quality for analysis.
4. **Quantify Relationships**: Measure correlation strength between personal attributes and insurance costs using both linear (Pearson) and non-linear (Spearman) methods
5. **Generate Actionable Insights**: Provide data-driven recommendations for pricing strategy, risk assessment, and targeted customer programs
6. **Support Predictive Modeling**: Clean and prepare data for future machine learning models that predict insurance costs


## Hypothesis

**Primary Hypothesis**: Smoking status (smoker) is the strongest predictor of insurance costs, followed by age, while demographic factors (sex, region) have minimal impact on pricing (charges).

**Statistical Methods**:
1. **Descriptive statistics**: Providing the descriptive summaries of both the numerical (mean, median (50%),count, 75th and 25th quantile, standard deviation, max and min values) and  categorical (mode(freq),top and unique values) variables.
1. **Categorical and Numerical Data Analysis**: Visualization of the variable behaviours or distribution (spread) through histograms, pie charts, and grouped distributions to identify patterns.
2. **Correlation Analysis**: 
   - Pearson correlation to measure linear relationships
   - Spearman correlation to measure monotonic (non-linear) relationships
3. **Comparative/Grouping Statistics**: Group comparisons using numerical distributions ('age',bmi, children and charges) grouped by categorical variables (smoking habits, geographical region and sex).
4. **Outlier Detection**: IQR-based boxplot analysis to identify outlier values which was capped using the Winsorizer capper from the feature engine library.

**Expected Outcomes**:
- Smoking status will show highest correlation with charges
- Age will show moderate positive correlation with charges
- Sex and region will show weak or no correlation with charges
- Non-linear relationships may be stronger than linear ones (evidenced by higher Spearman vs Pearson values) 

## Project Plan

### Steps:
#### Extract, Transformation and Loading of the raw dataset (ETL pipeline):

1. **Data Collection & Exploration**: Load insurance.csv and perform initial data inspection using `.info()`  method.
2. **Data Quality Assessment**: 
   - Check for missing values using `.isnull().sum()`
   - Identify duplicates using `.duplicated()`
   - Detect outliers using IQR method (Q1, Q3, IQR calculation)
3. **Data Cleaning & Transformation**:
   - Remove duplicate records
   - Apply Feature Engine's Winsorizer for outlier capping (IQR method, fold=1.5, tail='both')
   - Cap outliers in BMI and Charges variables
4. **Data Export**: Save cleaned dataset to `Cleaned_Insurance_Data.csv`
5. **Data Visualization**:
   - Distribution plots (histograms with KDE) for numerical variables (age,bmi, charges and children)
   - Pie charts for categorical variables (region, sex and smoking habits)
   - Grouped histograms and KDE  plots by categorical variables with color-coding by smoker status
6. **Statistical Analysis**:
   - Calculate Pearson correlation (linear relationships)
   - Calculate Spearman correlation (monotonic relationships)
   - Create correlation heatmaps for visual interpretation
7. **Insights & Conclusions**: Document findings and provide recommendations

### Data Management Throughout Pipeline:

- **Collection**: Raw data sourced from public repository (insurance.csv)
- **Processing**: Pandas for data manipulation, Feature Engine for outlier capping, scikit-learn Pipeline for reproducible transformation
- **Analysis**: NumPy for statistical calculations, SciPy for correlation methods
- **Visualization**: Matplotlib and Seaborn for exploratory plots and heatmaps
- **Output**: Cleaned CSV file stored in `data file/Clean_Data/` directory

### Methodology Justification:

- **Winsorizer vs Other Methods**: IQR-based capping chosen over deletion because it preserves data volume while reducing outlier values' influence unlike the Arbitrary capper or Outlier Trimmer which reduces the data volume. Fold=1.5 is business-standard for moderate outlier treatment
- **Pearson + Spearman**: Using both methods reveals whether relationships are linear or non-linear, providing more complete picture than single correlation method
- **Grouped Visualizations**: Breaking down by categorical variables shows how relationships affect demographic segments.
- **Pipeline Structure**: Ensures reproducible, scalable data transformation suitable for automation in production environments.

## The rationale to map the business requirements to the Data Visualisations

| Business Requirement | Data Visualisation | Rationale |
|---|---|---|
| Identify primary cost drivers | Figure 5 (Grouped histograms by category, colored by smoker) | Shows clear visual differentiation between smokers and non-smokers across all variables, making smoking's impact on charges immediately apparent |
| Understand price variation across demographics | Figure 5 (Distribution plots grouped by sex and region) | Demonstrates whether charges vary by demographic segment; reveals that sex and region have minimal impact on cost distribution |
| Detect outliers and anomalies | Figure 1 (Boxplots before capping) | IQR-based boxplots visually identify extreme values in BMI and Charges variables |
| Quantify relationships - linear perspective | Figure 6 (Pearson correlation heatmap) | Heatmap with color-coded correlation values (0.31 for age-charges) shows linear relationships at a glance |
| Quantify relationships - non-linear perspective | Figure 7 (Spearman correlation heatmap) | Reveals stronger correlation (0.53) than Pearson, indicating non-linear monotonic relationships |
| Verify data cleaning effectiveness | Figure 3 (Distribution plots after capping) | Side-by-side comparison shows reduced outlier influence and normalized distributions post-transformation |
| Understand categorical distributions | Figure 4 (Pie charts for sex, smoker, region) | Shows that ~79% of dataset are non-smokers but smokers pay disproportionately higher costs |

## Analysis techniques used

### Data Analysis Methods:

1. **Descriptive Statistics**: Mean, median, standard deviation, quantiles, and frequency distributions to summarize data characteristics
2. **Outlier Detection & Treatment**: 
   - IQR (Interquartile Range) method for identifying outliers
   - Winsorizing (IQR-based capping) with fold=1.5 on both tails for robust outlier handling
3. **Correlation Analysis**:
   - Pearson Correlation: Measures linear relationships (range: -1 to 1)
   - Spearman Rank Correlation: Measures monotonic (non-linear) relationships
4. **Exploratory Data Visualization**: Histograms, KDE plots, pie charts, boxplots, grouped distributions, and heatmaps
5. **Comparative Group Analysis**: Segmented analysis across categorical variables (sex, region, smoker status)

### Data Analysis Structure Justification:

1. **Top-Down Flow**: Started with data overview → quality assessment → cleaning → visualization → statistical testing. This prevents analysis bias from incomplete data understanding
2. **Dual Correlation Methods**: Spearman + Pearson comparison revealed that age-charges relationship is non-linear (r=0.31 vs ρ=0.53), suggesting curved or exponential patterns rather than straight-line relationships
3. **Grouped Analysis Before Aggregation**: Examined patterns within demographic subgroups before drawing overall conclusions, preventing Simpson's Paradox (where overall and subgroup trends might contradict)

### Data Limitations & Alternative Approaches:

**Limitations Encountered**:
- Limited feature set (only 7 variables) restricts comprehensive predictive modeling
- No temporal dimension; analysis is cross-sectional snapshot rather than longitudinal
- Categorical variables have imbalanced distributions (79% non-smokers vs 21% smokers)

**Alternative Approaches Considered**:
- **Log Transformation**: Could have been applied to Charges variable given its right-skewed distribution, but Spearman (which doesn't require normality) already captured non-linear relationship
- **Polynomial Features**: For future predictive models, polynomial features (age², age×bmi) could capture non-linear relationships identified in current analysis
- **Statistical Testing**: T-tests or ANOVA could formally test whether smokers' charges differ significantly from non-smokers, complementing correlation analysis
- **Clustering**: K-means or hierarchical clustering could identify natural customer segments if distinct groups exist

### Use of Data Science Libraries:

**Why This Structure?**:
- Feature Engine's Winsorizer is purpose-built for robust outlier handling, eliminating need for manual capping logic
- scikit-learn's Pipeline ensures reproducible, scalable data transformation (can be saved and reused on new data)
- Pandas' groupby operations efficiently segment data for comparative analysis
- Seaborn provides statistical visualization (KDE, hue) with cleaner syntax than raw Matplotlib

## Ethical considerations

### Data Privacy & Bias Issues:

1. **Individual Privacy**: The dataset contains personal health information (BMI, medical charges, smoking status). While not publicly identifiable in this format, proper data handling practices were followed.
   - **Mitigation**: Data aggregated at group level for analysis; no individual records exposed in conclusions
   
2. **Selection Bias**: Dataset may represent only specific geographic region, age group, or socioeconomic status
   - **Mitigation**: Regional analysis (Figure 5) included to identify if patterns vary geographically. Acknowledged in limitations that findings may not generalize globally
   
3. **Algorithmic Fairness**: Analysis reveals smoking status as primary cost driver, but could perpetuate discrimination
   - **Consideration**: While statistically valid, pricing based solely on smoking status may disproportionately affect lower-income groups who smoke at higher rates
   - **Recommendation**: Cost differential should account for smoking cessation programs and support services
   
4. **Protected Characteristics**: Age included in analysis; age-based pricing may have legal implications in some jurisdictions
   - **Mitigation**: Age analysis contextualized; recommended for risk assessment rather than sole pricing factor

### Legal & Societal Considerations:

1. **Fair Lending/Insurance Practices**: Insurance regulators in many jurisdictions restrict which factors can be used for pricing
   - **Solution**: Recommend stakeholder review with legal/compliance teams before implementing insights in pricing
   
2. **Health Equity**: Analysis shows smokers charged 3-4x more, which may price out vulnerable populations
   - **Recommendation**: Develop sliding scale or wellness incentive programs to maintain access while managing risk
   
3. **Data Minimization**: Collected only variables necessary for analysis; no extraneous demographic data that could enable discrimination

## Dashboard Design

**Note**: Current project deliverable is a Jupyter Notebook with integrated visualizations and analysis rather than an interactive dashboard. However, proposed dashboard structure for future implementation:

### Dashboard Pages & Content:

**Page 1: Overview & Summary Metrics**
- Total records analyzed (1,338)
- Average insurance charge (aggregated and by smoker status)
- Key statistics cards (min/max charges, age range, BMI distribution)
- High-level insight banner: "Smoking increases average charges by X%"

**Page 2: Variable Distribution Analysis**
- Widgets: Four distribution plots (Age, BMI, Children, Charges) with KDE overlays
- Toggle: Before/After outlier capping comparison
- Interactive histogram with adjustable bin sizes

**Page 3: Demographic Breakdown**
- Pie charts for categorical variables (Sex, Smoker, Region)
- Tabbed interface for quick category switching
- Filter buttons to drill down into specific segments

**Page 4: Relationship & Correlation Analysis**
- Side-by-side Pearson and Spearman heatmaps with correlation values
- Interactive heatmap allowing hover to see exact correlation coefficients
- Color legend explaining correlation strength (red=positive, blue=negative)

**Page 5: Grouped Analysis & Insights**
- Dropdown selector for numerical variable (Age, BMI, Children, Charges)
- Display grouped histograms (2×3 grid) automatically generated for selected variable
- Color legend showing smoker (red) vs non-smoker (blue) distinction

**Page 6: Deep Dive Analysis**
- Filterable tables showing exact charges by demographic combinations
- Statistical summary cards (mean, median, std dev) by selected group
- Export buttons for filtered data

### Design Approach for Different Audiences:

**For Executive/Non-Technical Stakeholders**:
- Summary metrics and key finding cards prominently displayed
- Visualizations use intuitive color schemes (red for high-risk/high-cost, green for low-cost)
- Narrative insights boxes above charts explaining "what this means"
- Avoid technical terminology; use business language ("smoking status is primary cost driver")

**For Data Scientists/Technical Teams**:
- Detailed heatmaps with exact correlation coefficients
- Option to download processed datasets
- Statistical test results and p-values
- Methodology documentation accessible via info buttons

**For Insurance Actuaries/Risk Assessors**:
- Segmentation tables with charge distributions by risk category
- IQR and outlier treatment parameters visible
- Group comparison tools (e.g., "smoker 50yo vs non-smoker 50yo")
- Predictability indicators (correlation strength as proxy for model potential) 

## Unfixed Bugs

**Current Status**: No known unfixed bugs. All analysis code executes successfully end-to-end.

### Challenges Resolved:

1. **Data Path Issues**: 
   - **Problem**: Initial notebook contained hardcoded path `/mnt/chromeos/MyFiles/Downloads/archive.zip` (specific to Chromebook environment)
   - **Solution**: Updated to relative path `data file/Raw_Data/insurance.csv` for portability across systems
   - **Lesson**: Always use relative paths in shared projects for reproducibility

2. **Outlier Capping Trade-offs**:
   - **Challenge**: Deciding between deletion vs capping for outliers. Deletion would lose 5-10% of data; capping preserves sample size but slightly distorts distribution
   - **Decision**: Chose capping (Winsorizing) as it maintains data integrity while reducing extreme value influence
   - **Validation**: Compared pre/post distributions to confirm capping achieved intended smoothing effect

3. **Correlation Method Selection**:
   - **Challenge**: Initial analysis used only Pearson correlation, which showed weak age-charges relationship (r=0.31)
   - **Discovery**: Spearman correlation revealed stronger relationship (ρ=0.53), indicating non-linear pattern
   - **Gap Addressed**: Added comprehensive explanation of why Spearman is more appropriate for this dataset

### Knowledge Gaps & How They Were Addressed:

1. **Feature Engineering for Outliers**:
   - **Gap**: Initially unfamiliar with Feature Engine library's Winsorizer
   - **Resolution**: Consulted library documentation; tested fold parameter (1.5 is standard); verified output against manual IQR calculations
   - **Outcome**: Successfully implemented robust outlier handling without manual coding

2. **Non-Linear Correlation Concepts**:
   - **Gap**: Understood Pearson but needed to deepen knowledge of Spearman and monotonic relationships
   - **Resolution**: Researched difference between linear vs monotonic relationships; practiced interpreting Spearman ρ values
   - **Outcome**: Can now identify when non-linear methods are more appropriate than linear ones

3. **Pipeline & Reproducibility**:
   - **Gap**: Initially wrote ad-hoc transformation code; needed to structure it for reuse
   - **Resolution**: Restructured using scikit-learn Pipeline; this enables saving fitted transformers for applying to new data
   - **Outcome**: Improved code maintainability and prepared foundation for model deployment

## Development Roadmap

### Challenges Faced & Strategies to Overcome:

1. **Challenge: Data Quality Issues**
   - **Problem**: Dataset contained 2 duplicate records and significant outliers in BMI (range: 16-54) and Charges (range: $1,122-$63,770)
   - **Strategy Used**: 
     - Implemented systematic duplicate detection using `.duplicated()` method
     - Applied IQR-based outlier identification before deciding on capping vs deletion
     - Selected Winsorizer for robust handling while preserving data volume
   - **Outcome**: Clean, analysis-ready dataset with 1,337 valid records

2. **Challenge: Weak Linear Correlation Results**
   - **Problem**: Pearson correlation showed age-charges correlation of only 0.31 (weak), contradicting domain knowledge that age strongly influences insurance costs
   - **Strategy Used**: 
     - Investigated by running Spearman correlation (captures non-linear relationships)
     - Spearman showed ρ=0.53 (moderate), indicating curved/exponential relationship rather than linear
     - Visualized age-charges distribution to understand the non-linearity visually
   - **Outcome**: Deeper understanding that cost increases with age but not in straight-line fashion

3. **Challenge: Interpreting Grouped Visualizations**
   - **Problem**: Initial distribution plots didn't effectively show smoker-status impact across demographic segments
   - **Strategy Used**: 
     - Restructured plots to use 2×3 grid (6 subplots per numerical variable)
     - Added color-coding by smoker status using seaborn hue parameter
     - Included categorical grouping (sex and region combinations)
   - **Outcome**: Clear visual demonstration that smoking is primary cost driver across all demographics

4. **Challenge: Reproducibility & Code Portability**
   - **Problem**: Initial notebook had hardcoded paths specific to Chromebook environment
   - **Strategy Used**: 
     - Converted to relative paths
     - Structured code in functions for reuse
     - Documented assumptions in markdown cells
   - **Outcome**: Notebook now runs on any system with correct folder structure

### Next Skills & Tools to Learn:

1. **Machine Learning Modeling** (Next Priority)
   - Learn regression models (Linear, Polynomial, Random Forest) to predict insurance costs
   - Apply cross-validation and hyperparameter tuning techniques
   - Compare model performance to establish baseline for current correlation insights
   - **Why**: Current analysis identifies relationships; modeling quantifies predictive power

2. **Interactive Dashboarding** (Medium Priority)
   - Master Plotly/Dash for web-based interactive visualizations
   - Build web application allowing stakeholders to filter and explore data
   - Implement real-time updates if data sources are added
   - **Why**: Static visualizations are valuable; interactive dashboards enable business users to explore independently

3. **Statistical Hypothesis Testing** (High Priority)
   - Learn t-tests, ANOVA, chi-square tests for formal statistical validation
   - Understand p-values, confidence intervals, effect sizes
   - Test whether observed differences (e.g., smoker vs non-smoker charges) are statistically significant vs random variation
   - **Why**: Current correlations are descriptive; hypothesis tests add inferential rigor

4. **Advanced Data Visualization** (Medium Priority)
   - Learn animated visualizations showing data over time
   - Master faceted/small-multiples plots for comparing many segments
   - Create publication-quality figures with proper legends, annotations, fonts
   - **Why**: Insights are only valuable if effectively communicated

5. **Feature Engineering & Domain Expertise** (High Priority)
   - Learn to create composite features (e.g., age groups, BMI categories)
   - Add domain knowledge about insurance (actuarial methods, risk adjustment formulas)
   - Engineer cyclical features if temporal data becomes available
   - **Why**: More variables improve predictive models; domain context improves interpretation

6. **Deployment & Automation** (Lower Priority - Future)
   - Learn to containerize analysis with Docker
   - Automate periodic report generation as new data arrives
   - Deploy predictive model as API for real-time cost estimation
   - **Why**: Transitions analysis from research to production operational tool 

## Deployment

### Current Deployment Status

The current project is a **Jupyter Notebook-based analysis** with the following deployment considerations:

**For Local Execution**:
1. Clone the repository: `git clone https://github.com/chizoalban2003-beep/Project1.git`
2. Navigate to project directory: `cd Project1`
3. Activate virtual environment: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Launch Jupyter Notebook: `jupyter notebook jupyter_notebooks/Notebook_Template.ipynb`
6. Run cells sequentially (top-to-bottom) to generate analysis and visualizations

### Future Deployment Options:

**Option 1: Heroku Web Application (If Dashboard is Built)**
- *When*: After converting Jupyter analysis into interactive Plotly/Dash dashboard
- *Steps*:
  1. Create `app.py` with Dash application code
  2. Create `Procfile` with: `web: gunicorn app:server`
  3. Set Python version in `runtime.txt` to Heroku-20 supported version (e.g., `python-3.11.4`)
  4. Commit all files including `requirements.txt` to GitHub
  5. Log into Heroku and create new App
  6. Connect GitHub repository to Heroku
  7. Enable automatic deployment for main branch
  8. Monitor deployment in Heroku dashboard; access live app at `https://[app-name].herokuapp.com/`

**Option 2: Streamlit Cloud (Recommended for Quick Deployment)**
- *Advantages*: Simpler than Heroku, free tier, Streamlit optimized for data science
- *Steps*:
  1. Convert notebook to Streamlit script (`app.py`)
  2. Push to GitHub public repository
  3. Connect repository to Streamlit Cloud
  4. Live deployment in seconds

**Option 3: GitHub Pages (Static Visualizations)**
- *Advantages*: Free, simple, good for sharing static reports
- *Use case*: Generate HTML report from Jupyter notebook, push to gh-pages branch
- *Command*: `jupyter nbconvert --to html notebook.ipynb`


## Main Data Analysis Libraries

### Core Libraries & Usage Examples:

**1. Pandas (v2.3.3)**
- **Purpose**: Data manipulation, cleaning, and exploratory analysis
- **Usage Examples**:
  ```python
  import pandas as pd
  df = pd.read_csv('data file/Raw_Data/insurance.csv')  # Load data
  df.info()  # Display data types and missing values
  df.isnull().sum()  # Check for missing data
  df[df.duplicated(keep=False)]  # Find duplicate rows
  df.drop_duplicates(keep='first', inplace=True)  # Remove duplicates
  df.describe(include='all')  # Descriptive statistics
  ```
- **Why Chosen**: Industry standard for data manipulation; clean API for grouping, filtering, and aggregation

**2. Scikit-Learn (v1.8.0) - Feature Engine (v1.9.3)**
- **Purpose**: Outlier detection and capping using Winsorizer
- **Usage Examples**:
  ```python
  from feature_engine.outliers import Winsorizer as W
  from sklearn.pipeline import Pipeline
  
  # Create pipeline with Winsorizer
  P = Pipeline([('Capper', W(capping_method='iqr', tail='both', fold=1.5, variables=['bmi', 'charges']))])
  
  df_transformed = P.fit_transform(df)  # Apply transformation
  P['Capper'].left_tail_caps_  # View capping limits
  ```
- **Why Chosen**: Feature Engine provides production-ready outlier handling; Pipeline ensures reproducibility

**3. NumPy (v2.4.1)**
- **Purpose**: Numerical calculations, statistical operations
- **Usage Examples**:
  ```python
  import numpy as np
  Q1 = df['age'].quantile(0.25)  # 25th percentile
  Q3 = df['age'].quantile(0.75)  # 75th percentile
  IQR = Q3 - Q1  # Interquartile range for outlier detection
  ```
- **Why Chosen**: Efficient vectorized operations; foundation for pandas and scipy

**4. Matplotlib (v3.10.8)**
- **Purpose**: Core plotting library for creating figures
- **Usage Examples**:
  ```python
  import matplotlib.pyplot as plt
  
  fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
  axes[0].set_title("Figure Title")
  axes[0].set_xlabel("X-axis Label")
  fig.suptitle("Overall Figure Title")
  plt.tight_layout()
  plt.show()
  ```
- **Why Chosen**: Fine-grained control over plot appearance; foundation for seaborn

**5. Seaborn (v0.13.2)**
- **Purpose**: Statistical visualization with enhanced aesthetics
- **Usage Examples**:
  ```python
  import seaborn as sns
  
  # Distribution plot with kernel density estimate
  sns.histplot(data=df, x='age', kde=True, ax=ax1)
  
  # Grouped histogram colored by category
  sns.histplot(data=df_subset, x='charges', kde=True, hue='smoker', ax=ax2, palette='Set1')
  
  # Correlation heatmap
  sns.heatmap(df_corr, annot=True, cmap='coolwarm', fmt=".2f")
  
  # Boxplot for outlier visualization
  sns.boxplot(data=df, x='bmi', ax=ax3)
  ```
- **Why Chosen**: Built on Matplotlib but with statistical intelligence; cleaner syntax for common plots

**6. SciPy (v1.17.0)**
- **Purpose**: Statistical calculations, including Spearman correlation
- **Usage Examples**:
  ```python
  from scipy.stats import spearmanr
  
  # Spearman correlation directly (used internally by pandas.corr(method='spearman'))
  df_corr_spearman = df[numerical_cols].corr(method='spearman')
  ```
- **Why Chosen**: Provides non-parametric statistical tests; more appropriate for non-normal distributions

**7. Statsmodels (v0.14.6)**
- **Purpose**: Advanced statistical modeling and hypothesis testing
- **Note**: Imported but not actively used in current analysis; prepared for future statistical testing
- **Future Use**: T-tests (differences between mean of grouped variables for a better sense of correlation), ANOVA, regression diagnostics

### Library Interaction Flow:

```
Data Input (Pandas)
    ↓
Outlier Detection (NumPy + Feature Engine)
    ↓
Data Cleaning/Transformation (Pandas + Pipeline)
    ↓
Statistical Analysis (SciPy + Pandas)
    ↓
Visualization (Matplotlib + Seaborn)
    ↓
Output (CSV + Figures)
```


## Credits 

### Content & Analysis

- **Dataset Source**: Healthcare Insurance dataset from publicly available health analytics repository (insurance.csv)
- **Data Science Methods**: Methodology drawn from industry best practices in exploratory data analysis and statistical correlation analysis
- **Outlier Treatment Approach**: Winsorizing technique based on Feature Engineer library documentation and statistical textbooks
- **Correlation Analysis**: Both Pearson and Spearman methods implemented following statistical best practices from SciPy documentation

### Code & Technical Resources

- **Feature Engine Library**: https://feature-engine.readthedocs.io/ - Used for robust Winsorizer implementation
- **Pandas Documentation**: https://pandas.pydata.org/docs/ - Data manipulation and descriptive statistics
- **Seaborn Documentation**: https://seaborn.pydata.org/ - Statistical visualization best practices
- **Scikit-Learn Pipeline**: https://scikit-learn.org/stable/modules/compose.html - Data transformation pipeline pattern
- **Matplotlib Visualization Guide**: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html - Subplot creation and figure management
- **Git & GitHub**: Version control and repository management for project tracking

### Learning & Development

- **Python Virtual Environments**: Standard practice for dependency isolation in Python projects
- **Jupyter Notebook Best Practices**: Documentation and code organization patterns
- **Data Analysis Workflow**: ETL (Extract, Transform, Load) pipeline structure widely used in industry



## Acknowledgements (optional)

* **Code Institute**: Gratitude to the Code Institute program and course administrators (Vasi Pavaloi, Mark, Tom Cohen, Niel and my Bootcamp mates) for guidance and support throughout the project despite my linux issues.
* **Data Science Community**: The pandas, scikit-learn, and seaborn communities for building excellent open-source tools that enabled this analysis.
* **Feature Engine Developers**: Special thanks to the Feature Engine library creators for providing robust outlier handling tools.
* **Educational Resources**: Jupyter documentation and various data science tutorials that informed best practices for notebook organization and analysis workflows.
* **GitHub & Version Control**: Appreciation for Git version control system that enabled tracking of analysis iterations and recovery of previous states.
