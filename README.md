# FeatureEngineering


In this project, we will learn:
- what is feature engineering
- basic feature engineering techniques
- what is a feature store
- compare existing feature store solutions

# 1. What is feature engineering?

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
    - ensuring variables are on the same scale, making the model easier to understand; 
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

The art of feature engineering may vary among data scientists, however steps for how to perform feature engineering for most machine learning algorithms include the following:

Data Preparation: This preprocessing step involves the manipulation and consolidation of raw data from different sources into a standardized format so that it can be used in a model. Data preparation may entail data augmentation, cleaning, delivery, fusion, ingestion, and/or loading. 
Exploratory Analysis: This step is used to identify and summarize the main characteristics in a data set through data analysis and investigation. Data science experts use data visualizations to better understand how best to manipulate data sources, to determine which statistical techniques are most appropriate for data analysis, and for choosing the right features for a model. 
Benchmark: Benchmarking is setting a baseline standard for accuracy to which all variables are compared. This is done to reduce the rate of error and improve a model’s predictability. Experimentation, testing and optimizing metrics for benchmarking is performed by data scientists with domain expertise and business users.
Exploratory Data Analysis : Exploratory data analysis (EDA) is a powerful and simple tool that can be used to improve your understanding of your data, by exploring its properties. The technique is often applied when the goal is to create new hypotheses or find patterns in the data. It’s often used on large amounts of qualitative or quantitative data that haven’t been analyzed before.
Benchmark : A Benchmark Model is the most user-friendly, dependable, transparent, and interpretable model against which you can measure your own. It’s a good idea to run test datasets to see if your new machine learning model outperforms a recognised benchmark. These benchmarks are often used as measures for comparing the performance between different machine learning models like neural networks and support vector machines, linear and non-linear classifiers, or different approaches like bagging and boosting. To learn more about feature engineering steps and process, check the links provided at the end of this article. Now, let’s have a look at why we need feature engineering in machine learning.