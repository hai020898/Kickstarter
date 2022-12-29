# Kickstarter

Final Project for INSY 662 as part of MMA program at McGill University

For the final project I was tasked with develop a classification model and a clustering model using the Kickstarter dataset. 

First, exploratory data analysis was conducted to understand the distribution of variables and missing values. I then performed data cleaning to remove irrelevant values and combine several columns, as well as dummifying categorical variables. Afterwards, I standardized the predictors and split the dataset into training and test set.

For the classification model, I use accuracy score as a measure of efficiency. After trying different model and hyperparameter tuning, I chose the Gradient Boosting Model with 6 samples required to split. 

For the clustering model, I used the K-Means clustering model. I used the silhouette score and the elbow method to determine the best number of clusters (n=4). 

More detail on the methodology could be found in final_report.pdf

Below are a list of the files in the repository:

* **Classification_final.py**: Contains the code to perform data cleaning, feature engineering, implementing the classification model.

* **clustering_final.py**: Contains the code to perform data cleaning, feature engineering, implementing the clustering model.

* **Kickstarter.xlsx**: Contains the data of projects from Kickstarter

* **Kickstarter-Grading-Sample.xlsx**: Test data to test the classification model

* **final_report.pdf**:  The final report outlining the significance of the problem, the methodology for the analysis and insights gained from the results.
