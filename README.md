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
Relevant columns were selected through `usecols = ['pickup_datetime','pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']. Before and after: ![select_columns](screenshots/selecting_columns.PNG)



#### Removing outliers
Fare amounts can not be negative, hence removed and stored in uber_df. The longitude and latitude are restricted to New York as outliers outside this location was found (discovered later). 

`uber_df = df[(df['pickup_longitude'] >= -74.2591) &
                 (df['pickup_longitude'] <= -73.7004) &
                 (df['pickup_latitude'] >= 40.4772) &
                 (df['pickup_latitude'] <= 40.774) & (df['fare_amount'] <= 300) & (df['fare_amount'] > 0)]`

![negativefares](screenshots/negative_fares.PNG)

### Data Transformation 

Columns for weekday, day, minute, month, and hour are created from the PICKUP_DATETIME column by converting it from object to datetime datatype.
`df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])`
`df['pickup_datetime'] = df['pickup_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')`
`df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])`



`df['weekday']=df['pickup_datetime'].dt.day_name()`
`df['day']=df['pickup_datetime'].dt.day`
`df['minute']=df['pickup_datetime'].dt.minute`
`df['month']=df['pickup_datetime'].dt.month`
`df['hour']=df['pickup_datetime'].dt.hour`
`df.head()`

![transformed](screenshots/transformed_columns.PNG)


## Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was conducted to gain an understanding of the dataset and extract insights prior to modeling. It was found that the average fare amount was $11.44, with the fare distribution exhibiting a positive skew. Uber pick-up frequency showed a significant increase after 8 PM, with Friday being the busiest day for trips. Outliers in the pick-up latitude and pick-up longitude were identified outside the NYC boundaries, prompting the dataset to be restricted to latitude and longitude values within NYC.A density heatmap of fare prices revealed that Upper Manhattan is a hotspot for relatively higher fare prices.


![describe](screenshots/describe.PNG)

![f](screenshots/fare_amount_histogram.PNG)

![h](screenshots/distribution_of_hours.PNG)

![h](screenshots/busiest_day.PNG)

![h](screenshots/density_heatmap.PNG)





## Results and Analysis  
