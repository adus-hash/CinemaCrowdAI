## Project goals

Project goals are: 

1. Use machine learning algorithms to predict attendance in cinema

2. Learn and compare different ML algorithms (Linear regression, Random Forest, Decision Tree, Support Vector Machine)

## Dataset

It is artificially made, consist of 7 variables and first 5 samples look like this:

<img width="667" alt="Snímka obrazovky 2024-12-27 o 22 24 42" src="https://github.com/user-attachments/assets/05212e09-6990-44ff-bb10-0fcf1ae1bd85" />

Our target variable is "Attendance (%)" which we aim to predict with our model using the other variables, hoever here are steps we need to do at first:
  - We need to encode string data into numeric
  - Fill missing datas with 0's
  - Covariance, Correlation
  - Choose best variables that correlate with target variable

We encode strings and fill missing datas with these piece of code
```Python
encoder = OrdinalEncoder()
df[["Month of the Year", "Genre"]] = encoder.fit_transform(df[["Month of the Year", "Genre"]])

df.fillna(0, inplace=True)
```
Then we calculate covariance and correlation using .cov() and .corr() methods, if we look at covariance, it tells us how variables behave linearly with respect to each other. In our case, the two most linearly related variables are 'Release Year' and 'Attendance (%)', as well as 'Movie Duration' and 'Attendance (%)'. When the release year of a movie increases, cinema attendance rises sharply, whereas when the movie duration increases, attendance decreases. For better interpretation of covariance and dependencies, we will use correlation, if we look at the table (+1 - Perfect positive linear relationship, -1 Perfect negative linear relationship, 0 - No linear relationship) 'Release Year' and 'Rating' have the highest correlation with attendance, so we will select them as input variables for our model.

|    Covariance   |   Attendance  |
|-----------------|---------------|
|Genre            |     -3.182462 |
|Release Year     |    183.212470 |                      
|Rating           |     20.773803 |
|Duration         |    -93.818008 |
|Ticket Price (€) |     -8.729587 |
|Month of the Year|    -11.000959 |
                                                         

|   Correlation   |  Attendance  |
|-----------------|--------------|
|Attendance (%)   |     1.000000 |
|Release Year     |      0.368998|
|Rating           |      0.305451|
|Genre            |     -0.042216|
|Duration         |     -0.112921|
|Ticket Price (€) |    -0.118620 |
|Month of the Year|   -0.124997  |
