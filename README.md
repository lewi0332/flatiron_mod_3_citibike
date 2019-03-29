
# CitiBike Destination Predictor

This project is a collaboration between [Elena Morais](https://github.com/elenasm7) and [Derrick Lewis](https://github.com/lewi0332)

The project requirements are to find and clean data to be fit to a classification algorithm to make predictions on the outcome.

We chose to use data from CitiBike of all trips in 2018. This data includes time, start/stop locations, and some basic demographic identifiers. To aid in our perdiction we added daily weather information of New York City.
- Trip Duration (seconds)
- Start Time and Date
- Stop Time and Date
- Start Station Name
- End Station Name
- Station ID
- Station Lat/Long
- Bike ID
- User Type (Customer = 24-hour pass or 3-day pass user; Subscriber = Annual Member)
- Gender (Zero=unknown; 1=male; 2=female)
- Year of Birth

The goal is to perdict the destination neighborhood of each journey based on the available variables. This is a multi-classification problem with a high number of outcomes. We determined that there are 51 qualifying neighborhoods in New York, which presents a sigificant challenge. While we have a very large amount of instances of this data to use, the variables involved do not have any obviously significant indication value. 

We will learn if the massive data size available can overcome our assumptions of predictability. Additionally, 

# Data Collection and Cleaning

First we collected 1 year worth of trip data from Cibibike.com. We learn that there are 17million records in the 2018 data, which is too large for our processing capbabilities. Thus we import each month of data, and randomly sample 10% from each month to create a more manageable, yet representative list.


 - NAN values
There are just 260 NaN values missing the station start and end points in the file of nearly 1.8million. This information is crucial to the prediction and thus can not be substituted. We have decidied to simply remove these rows.

- Convert to DateTime.
Start and End times are saved in a format that is not readable by Pandas or our future models. We will convert the columns with this information into a standard date format, then split the relevant information into individual columns to be used as a specific independant variable.

- Insert Weather Data
Next we connect daily weather information. The hypothesis is that this may aide in predicting a destination as fair weather might increase trips to parks and beaches. Data was collected from [https://www.weather.gov/okx/centralparkhistorical]

- Insert Neighborhood Name
Connect Neighborhood name data to our 'End Station Latitude/Longitude' to use as labels for predictions. In the most widely used context there are 51 neighborhoods in New York City.

- Convert usertype to categories
Our CitiBike data includes a label for each trip to determine if it was made by a rider who is an annual subscriber or someone who has purchased a temporary pass. We assume that there may be destinations more likely chosen by tourist users.


- Smooth '1969' birth year
Lorem Ipsum

- Label Encode for Morning versus Evening.
To better fit a model, using a split day rather than by hour might aide in determining a destination. The idea is that many nieghborhoods are destinations for commuters going to work in the morning and alternatively heading home in the evening.


Remove redundant rows -
Now that our primary categories are set we will remove features that are not needed or redundant and we are ready to run models on a tidy data set. We save the pandas dataframe as a csv file from here in order to work quickly on future workbooks. 

# EDA 
View Correlation between varibles.

dist plots
value counts 
ETS

# Models 
 
 2 class test
  acc % 
  CF plot
  
  
 Random Forest & ROC curve
 SVM
 XGBoost
 
# Results 
This was a very challenging question with a prediction accuracy that is understandibly difficult. Given the large number of classifications to predict and the limited information available about each user, our expectation is to beat a random guess and or the mean probability estimate. 


  ROC Cross
  Accuracy
  CFMT 
  
# Next Steps 
