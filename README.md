## Project goals

Project goals are: 

1. Use machine learning algorithms to predict attendance in cinema

2. Learn and compare different ML algorithms (Linear regression, Random Forest, Decision Tree, Support Vector Machine)

## Dataset

It is artificially made, consist of 7 variables and first 5 samples look like this:

|     |    Title   |     Genre | Release Year |    Rating | Duration | Ticket Price (€) | Month of the Year | Attendance (%) |
|-----|------------|-----------|--------------|-----------|----------|------------------|-------------------|----------------|
|0    |  Film 1    | Romance   |      2007    | 3.70      |     77   |          5.11    |        April      |       41       |
|1    |  Film 2    |  Sci-Fi   |      1963    | 2.30      |     71   |          6.73    |     November      |       10       |
|2    |  Film 3    |  Comedy   |      1979    | 7.80      |    158   |         11.07    |         July      |       28       |
|3    |  Film 4    |  Horror   |      1978    | 3.27      |    119   |          5.35    |         July      |       57       |
|4    |  Film 5    |  Action   |      2002    | 3.55      |     99   |          8.28    |      January      |       56       |

Our target variable is "Attendance (%)" which we aim to predict with our model using the other variables, however here are steps we need to do at first:
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
Then we calculate covariance and correlation using .cov() and .corr() methods, if we look at covariance, it tells us how variables behave linearly with respect to each other. In our case, the two most linearly related variables are 'Release Year' and 'Attendance (%)', as well as 'Movie Duration' and 'Attendance (%)'. When the release year of a movie increases, cinema attendance rises sharply, whereas when the movie duration increases, attendance decreases. 
For better interpretation of covariance and dependencies, we will use correlation, if we look at the table (+1 - Perfect positive linear relationship, -1 Perfect negative linear relationship, 0 - No linear relationship) 'Release Year' and 'Rating' have the highest correlation with attendance, so we will select them as input variables for our model.

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

For additional information about the 'Rating' and 'Release Year', we can use the .describe() method:
|            | Rating | Release Year |
|------------|--------|--------------|
|count       | 1000.0 |  1000.0      |
|mean        | 5.608  |  1991.413    |
|std         | 2.606  |  19.031      |
|min         | 1.0    |  1960.0      |
|25%         | 3.460  |  1974.0      |
|50%         | 5.630  |  1991.0      |
|75%         | 7.90   |  2008.0      |
|max         | 10.0   |  2024.0      |

## Machine learning algorithms

Very simply, machine learning algorithms work by finding patterns in data and using those patterns to make predictions, in all cases the algorithm "learns" by analyzing past data (training) and then uses what it learned to predict new outcomes
In our project, we used 4 algorithms: Linear regression, Random Forest, Decision Tree, Support Vector Machine. Let's describe them

## Linear regression
It fits a straight line (or hyperplane in our case) to minimize the error (mean squared error in our case)

The equation it models is:

    - y = w1x1 + x2x2 + .. + wnxn + b

### When to use

    - When you expect a linear relationship between inputs and outputs

    - When you have a simple dataset

### Limitations

    - Struggles with complex non-linear relationships


## Decision Tree
A decision tree splits the dataset into smaller subsets based on feature values, forming a tree-like structure, the leaves represent the predictions, while the internal nodes represent decisions (splits)

### When to use

    - For non-linear relationships or datasets with complex structures

    - Suitable for datasets with both numerical and string features

### Limitations

    - Prone to overfitting, especially with deep trees

    - Sensitive to small changes in data (high variance)


## Random Forest
Random forest builds multiple decision trees and combines their outputs, each tree is trained on a random subset of the data and features (a process called bagging)

### When to Use

    - For large datasets with complex non-linear relationships

    - To reduce overfitting compared to a single decision tree

### Limitations

    - Computationally expensive for large datasets


## Support Vector Machine (SVM)
Finds the optimal hyperplane that maximizes the margin (distance) between classes (for classification) or minimizes errors (for regression, called SVR), it uses kernel functions (e.g., linear, polynomial, radial basis function (RBF)) to handle non-linear relationships by transforming data into a higher-dimensional space

### When to Use

    - For datasets with small to medium sizes and complex non-linear relationships

    - Works well in high-dimensional spaces (many features)

### Limitations

    - Computationally expensive for large datasets

    - Harder to interpret compared to simpler models like linear regression


## Project results
When evaluating the results we will consider the metrics Mean Squared Error, Mean Absolute Error, one sample from the dataset, and four newly created samples as {Rating, Release Year}.

### Mean Squared Error (MSE)
MSE measures the average squared difference between the predicted values and the actual values, smaller MSE indicates that the model's predictions are closer to the actual values

### Mean Absolute Error (MAE)
Measures the average absolute difference between the predicted values and the actual values, simply it tells you how far your predictions are from the true values on average

### Results
|           | Rating | Year | Linear Regression | Decision Tree   | Random Forest  |  SVM  |
|-----------|--------|------|-------------------|-----------------|----------------|-------|
|   MSE     |        |      |       527         |        16       |      97        |  692  |
|   MAE     |        |      |       19          |        0        |       7        |   21  |
|  Film 1   |   2.3  | 1963 |      16.96        |      10.0       |     15.13      | 36.90 |
|New film 1 |   8    | 2015 |      59.77        |      40.0       |     42.49      | 37.20 |
|New film 2 |   5    | 1999 |      42.89        |      45.0       |     36.08      | 37.11 |
|New film 3 |   2    | 2023 |      45.88        |      62.0       |     46.54      | 37.25 |
|New film 4 |   9    | 1974 |      42.38        |      26.0       |     29.64      | 36.97 |

In this table we can compare the MSE values, the worst result is from the SVM model, which may be caused by insufficient optimization of the model, the second worst is Linear Regression likely due to the lack of sufficiently linearly dependent variables in the dataset, on the other hand, the Decision Tree stands out possibly because it is overfitted, the Random Forest performed reasonably well providing the most reliable results with minimal deviations. 


### Conclusion
In this project our goals were predicting cinema attendance and comparing machine learning algorithms. Through dataset analysis we identified 'Rating' and 'Release Year' as the two most highly correlated variables. However, despite that it wasn't enough good as we can see in the Mean Squared Error of the models. The best-performing algorithms were Decision Tree and Random Forest (though the Decision Tree may be overfitted), while Linear Regression and Support Vector Machine performed the worst. For this specific prediction and dataset I would personally choose Random Forest, as it provides the most reliable results with minimal deviations.
