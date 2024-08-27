# Optimising Earnings for Uber Drivers In New York City


## Executive summary
This project uses K-means clustering to identify high-fare regions in NYC for Uber drivers. The analysis, with a moderate silhouette score of 0.48, clusters NYC into four regions. It helps drivers optimize routes and schedules to maximize earnings, especially during surge pricing.



## Data Preprocessing
The dataset used for this project was sourced from Kaggle's [Uber Fares Dataset](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset/data). 
This dataset was chosen due to its fare amount, latitude, longitude making it suitable for this project to visualise areas in NYC where an Uber Driver can traverse to maximise their earnings.

### Loading and Initial Exploration
The CSV dataset is imported into a Pandas DataFrame using the `pd.read_csv()` function.
Insights was gained into its structure and contents. 

![df_info](screenshots/df.info.PNG)

### Data Cleaning 

#### Removing missing values

Rows containing missing values were dropped to data integrity and accuracy in further analysis. Using `df = df.dropna()`.

![null_values](screenshots/null_values.PNG)

#### Removing irrelevant data
Relevant columns were selected through `usecols = ['pickup_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']. Before and after ![selectcolumns](selecting_columns.PNG)



#### Transforming 






## Exploratory Data Analysis (EDA)




## Analysis 
