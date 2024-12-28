## Project goals

Project goals are: 

1. Use machine learning algorithms to predict attendance in cinema

2. Learn and compare different ML algorithms (Linear regression, Random Forest, Decision Tree, Support Vector Machine)

## Dataset

It is artificially made, consist of 7 variables and first 5 samples look like this:

|     |    Title   |     Genre | Release Year |    Rating | Duration | Ticket Price (€) | Month of the Year | Attendance (%) |
|-----|------------|-----------|--------------|-----------|----------|------------------|-------------------|----------------|
|0    |  Film 1    | Romance   |      2007    | 3.700000  |     77   |          5.11    |        April      |       41       |
|1    |  Film 2    |  Sci-Fi   |      1963    | 2.300000  |     71   |          6.73    |     November      |       10       |
|2    |  Film 3    |  Comedy   |      1979    | 7.800000  |    158   |         11.07    |         July      |       28       |
|3    |  Film 4    |  Horror   |      1978    | 3.277309  |    119   |          5.35    |         July      |       57       |
|4    |  Film 5    |  Action   |      2002    | 3.557263  |     99   |          8.28    |      January      |       56       |

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

    - Struggles with complex, non-linear relationships


## Decision Tree
A decision tree splits the dataset into smaller subsets based on feature values, forming a tree-like structure, the leaves represent the predictions, while the internal nodes represent decisions (splits)

### When to use

    - For non-linear relationships or datasets with complex structures

    - Suitable for datasets with both numerical and string features

### Limitations

    - Prone to overfitting, especially with deep trees

    - Sensitive to small changes in data (high variance
