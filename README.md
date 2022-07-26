# Feature Engineering and all that


In this project, we will learn:
- what is feature engineering 
- basic feature engineering techniques
- what is a feature store
- compare existing feature store solutions
- feature management 


# 1. Feature engineering 

**Feature engineering** is a machine learning technique that takes a data set and constructs new `explanatory variables` (features)
that aren’t in the original training set. 

The goal of Feature engineering is to simplify and speed up data transformations while also enhance model accuracy.


This technique can be used for both `supervised and unsupervised learning`, a terrible feature will have a direct impact 
on your model accuracy regardless of your data and model.

For instance, in a house price data, you may have two column: 
- total_area: Total surface of the house
- living_area: Usable living surface of the house

The linear model will have difficulty to learn with the two column. But if we build a new feature:

`living_ratio=living_area/total_area`

With the domain knowledge of a real estate agent, we know the living_ratio has much more sense to predict the price of a house

## 1.1 Various feature engineering techniques

Feature engineering consists of various feature engineering techniques:

- **Feature Creation**: Creating features involves creating new variables by mixing existing features via operations (e.g. 
  addition, subtraction, multiplication, ratio, etc). The new derived features should have greater predictive power. 
  This is a subjective process that requires human intervention and creativity.   

- **Transformations**: Feature transformation involves manipulating the predictor variables to improve model 
    performance: 
    - ensuring the model is flexible in the variety of data it can ingest
    - ensuring variables are on the same scale, making the model easier to understand (min_max_normalization, z_score_normalization) 
    - improving accuracy; 
    - avoiding computational errors by ensuring all features are within an acceptable range for the model. 

- **Feature Extraction** (aka dimension reduction): Feature extraction is the process of extracting features from a data set to identify 
              useful information. Without distorting the original relationships or significant information, this 
              compresses the amount of data into manageable quantities for algorithms to process. Some feature extraction 
              methods include `cluster analysis, text analytics, edge detection algorithms, and principal components analysis`.

- **Feature Selection**: Feature selection algorithms essentially analyze, judge, and rank various features to 
            determine which features are irrelevant and should be removed, which features are redundant and should be 
            removed, and which features are most useful for the model and should be prioritized.

## 1.2 Steps in Feature Engineering

The art of feature engineering may vary among data scientists, however steps for how to perform feature engineering for 
most machine learning algorithms include the following:

- **Data Preparation**: This preprocessing step involves the manipulation and consolidation of raw data from 
                        different sources into a standardized format so that it can be used in a model. Data 
                        preparation may entail **data augmentation, cleaning, delivery, fusion, ingestion, and/or loading**. 
- **Exploratory Analysis**: This step is used to identify and summarize the main characteristics in a data set 
                        through data analysis and investigation. Data science experts use data visualizations to 
                       better understand how best to manipulate data sources, to determine which statistical 
                        techniques are most appropriate for data analysis, and for choosing the right features for a model. 
- **Benchmark**: Benchmarking is setting a baseline standard for accuracy to which all variables are compared. This 
                is done to reduce the rate of error and improve a model’s predictability. Experimentation, testing 
                and optimizing metrics for benchmarking is performed by data scientists with domain expertise and business users.

## 1.3 Importance Of Feature Engineering

Feature Engineering is a very important step in machine learning. It's a major factor that can impact the accuracy of 
a model. Below figure shows in general how much time a Data scientists spend on their project with data:

![ml_time_spent.PNG](img/ml_time_spent.PNG)

# 2. Feature engineering techniques 

[feature_engineering_techniques](docs/Feature_engnieering_intro.md)


# 3. What is a feature store?

A feature store creates a central place where different teams within an organization can share, build, and manage 
features – preventing the need to rebuild the same features. This allows organizations to save time, resources, 
ensure consistency of information, and scale their AI.


## 3.1 Feature store functionalities

### 3.1.1 Core Features

- Feature Registry/Search
- Feature Schema Versioning
- Offline Feature Store
- Online Feature Store
- Time Travel
- External feature group 
- Generate Training dataset with various File Formats (e.g. csv, parquet, avro, etc.)

### 3.1.2 Feature computation

- support feature engineer/computation framework (e.g. python, R, spark, etc.)
- streaming feature computation
- integrated feature data validation
- CI/CD Support
- 3rd Party tool Orchestration

### 3.1.3  Governance

- Authentication
- Access control
- Custom Metadata
- Feature Statistics
- Lineage

### 3.1.4 Data Ingestion
- Ingestion from various Streaming Sources
- Ingestion from various Batch Sources (e.g. s3, )

### 3.1.5 User Experience
- Web UI
- Feature Visualization


## 3.2 When to use a feature store?

Below are major challenges in ml 

- Training serving skew: The training data is preprocessed(e.g. normalization, feature splitting, etc.). If the 
                 incoming predication request (serving data) are not transformed with the same preprocessed step.
                 Otherwise, the accuracy of the predication will be corrupted. 




# 4. Existing feature store


## 4.1 Feast

## 4.2 Hopsworks

## 4.
## 4.x Intenal implementations

- Palette: (Uber implementation)   
- Zipline: (Airbnb)
- Time Travel: Netflix


# 5. Feature management

You can consider a common feature store as a feature management system,   this means defining a feature creation lifecycle. If you unpack this, it includes defining:

A proper folder structure
Naming conventions and tagging
Feature versioning
Feature Services for feature consumption instead of purely Feature Views. This opens up interesting use-cases like Shadow Deployments.
Safely creating and updating features with authorization and automation (CI/CD)