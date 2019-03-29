
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

https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/citibike_data_cleaning_and_eda.ipynb

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

![Age graph](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/original_age_dist.png)

![fixed age](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/fixed_age_dist.png)

- Label Encode for Morning versus Evening.
To better fit a model, using a split day rather than by hour might aide in determining a destination. The idea is that many nieghborhoods are destinations for commuters going to work in the morning and alternatively heading home in the evening.


Remove redundant rows -
Now that our primary categories are set we will remove features that are not needed or redundant and we are ready to run models on a tidy data set. We save the pandas dataframe as a csv file from here in order to work quickly on future workbooks. 

# Exploratory Data Analysis 

View Correlation between varibles.

dist plots
value counts 
ETS


# Models 
 
Once we have tidy data to work with we start testing the accuracy of a variety of models to determine which type of algorithm produces the best results for our data. 

When running models we will primarily look for a high accuracy score. In addition, we will observe a Confusion Matrix for each trial to determine where the predictions are being made. Lastly we will peroiodally check our Feature Importance for each class to verify each variables contribution to the final prediction. 

---

## 2 class test

https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/citibike_two_class.ipynb

In this notebook we will take in our data from CitiBike and organize down to just one start station and two end neighborhoods. The goal is to test if we can better predict on a question with fewer target classes.

We will test with the Xgboost classifier and use the same variables we have for other trials.

  
|   | Two Class Test |
| ------------- | :-------------: |
| Training Accuracy  | 86.19%  |
| Content Cell  | 65.4%  |
  
  CF plot:
  
  ![Confusion Matrix for a Two Class Trial](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/two_class_CFM.png)
  
The accuracy here is not execptional, however the results are better than a random guess between the two target neighborhoods. Addtionally, our training accuracy is much higher leading us to assume that with some tuning this predction could be more accurate. We will use the time to tune on other models containing all target classes. 

---

## Random Forest & ROC curve

https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/citibike_rand_forrest.ipynb

Our first model on the full data will be run with the Random Forrest Classifier. 

To find the best tuning for our model we start with a Grid Search function.  The Grid Search will iterate over our Random Forrest Classifier changing the hyperparameters each time looking for the best performing model. We start with the following grid of parameters to test: 

|  Grid Search CV  | Settings |  | |
| ------------- | :-------------: | :-------------: | :-------------: |
| criterion  | 'gini'  | 'entropy'  |   | 
| max_depth  | 2 | 3  | 4 |
| min_child_weight | 4 | 5 |  |
| n_estimators | 300 | 400 | 500 |

This Grid Search produced best parameters of: 

|   | Best Parameters  |
| ------------- | :-------------: |
| criterion  | 'gini  |
| max_depth  | 3  |
| n_estimators | 400 |

Then we run our model with these parameters: 


|   | Random Forrest  |
| ------------- | :-------------: |
| Training Accuracy  | 8.24%  |
| Validation Accuracy  | 6.96%  |

Confusion Matrix for Random Forest with 

![Random Forest CFM](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/random_forrest_dfm.png)

Feature Importance for Random Forrest: 

![Random Forrest Feature Importance](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/rand_f_feat.png)

### ROC Curve for our Random Forest Model

explain roc curve!

![Roc Curve Graph](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/roc_curve.png)


---

## Support Vector Machines

https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/citibike_svm.ipynb

We then ran our data through a Suport Vector Classification algorithim. In this iteration we will run three styles of the support vector classification algorithm. The first is a linear classification of the data, the second is an RBF kernel style classfication and lastly the Poly Kernel.

Parameter Search - 
We begin each trial with a parameter search similar to the grid search above. This will determine the best parameters to fit our data. 

* Linear Model

|   | Best Parameters  |
| ------------- | :-------------: |
| c  | 1 |
| Gamma  | 0.001 |


|   | Support Vector Machine Test |
| ------------- | :-------------: |
| Training Accuracy  | 7.6%  |
| Validation Accuracy  | 6.56%  |


![svm linear](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/svm_linear_cfm.png)


* RBF Model

|  RBF | Best Parameters  |
| ------------- | :-------------: |
| c  | 1 |
| Gamma  | 0.1 |



| RBF | Support Vector Machine Test |
| ------------- | :-------------: |
| Training Accuracy  | 7.573%  |
| Validation Accuracy  | 6.667%  |


![svm rbf](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/svm_rbf_cfm.png)


* Poly Kernel Model

| POLY  | Best Parameters  |
| ------------- | :-------------: |
| c  | 0.001 |
| Gamma  | 1 |



|  POLY | Support Vector Machine Test |
| ------------- | :-------------: |
| Training Accuracy  | 6.658%  |
| Validation Accuracy  | 6.133%  |


![svm poly](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/svm_poly_sfm.png)

---

## XGBoost

https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/citibike_xgboost.ipynb

In this trial we used XGBoost. This algorithim allows for a quicker processing of a Gradient Boosted Decision Tree. Thus we quickly realized that we could see better results with a larger volume of data. XGBoost was able to process this on our machines in a timely fashion. 

For this trial we used the largest sample set thus far and produced the best results thus far. 


|  Grid Search CV  | Settings | XGBoost | |
| ------------- | :-------------: | :-------------: | :-------------: |
| learning_rate  | 0.5  | 0.7  | 0.1  | 
| max_depth  | 2 | 3  | 4 |
| min_child_weight | 4 | 5 |  |
| n_estimators | 300 | 400 | 500 |

This Grid Search resulted in the following best parameters:

|  XGBoost | Best Parameters  |
| ------------- | :-------------: |
| learning_rate  | 0.5  |
| max_depth  | 2 |
| min_child_weight | 4 |
| n_estimators | 300 |


|   | XGBoost Test |
| ------------- | :-------------: |
| Training Accuracy  | 10.83%  |
| Validation Accuracy  | 9.767%  |

XGBoost Confusion Matrix:

![500k xgboost](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/500k_xgboost.png)

XGBoost Feature Importance: 

![500k XgBoost](https://github.com/lewi0332/flatiron_mod_3_citibike/blob/master/Visualizations/500k_xgB_feat.png)


---

## Smote 

link 

Explain

Results

Conclusions

## Random Forrest with Reframed Question: 

In this trial we reframed the question to truncate our start stations into the same parameters as our detination neighborhoods. This reduced this prominent feature from **750** categories to just **51**. This can provide more meaning perdiction. 

Thus, given the new question of Given the starting *neighborhood* from which a rider started, what is the most likely destination neighborhood.

|  Grid Search CV  | Settings |  | |
| ------------- | :-------------: | ------------- | ------------- |
| criterion |   | 'entropy'  |   | 
| max_depth  | None | 2  | 5 |
| min_smalpes_split | 2 | 5 | 12 |
| min_smaple_leaf | 1 | 2 | |
| n_estimators | 50 | 100 | 150 |


|  Grid Search CV  | BEst Parameters | 
| ------------- | :-------------: | 
| criterion |   | 'entropy'  |   
| max_depth  | None | 5 |
| min_samples_split | 2 | 
| min_sample_leaf | 2 |
| n_estimators | 50 |



|   | Random Forrest Test |
| ------------- | ------------- |
| Training Accuracy  | 40.68%  |
| Validation Accuracy  | 14.76%  |


---

# Results 
This was a very challenging question with a prediction accuracy that is understandibly difficult. Given the large number of classifications to predict and the limited information available about each user, our expectation is to beat a random guess and or the mean probability estimate. 


  ROC Cross
  Accuracy
  CFMT 
 
---

# Next Steps 
