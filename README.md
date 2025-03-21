# Tuning-diabetes-prediction-model

This project aims to improve the performance of a diabetes prediction model by focusing on two key concepts: **cross-validation** and **hyperparameter tuning**. The goal is to achieve the best possible accuracy by tuning the model using various values of k for the K-Nearest Neighbors (KNN) algorithm.






## Model

We define a function `get_kneighbors_score()` to perform cross-validation using different values of k for the **K-Nearest Neighbors (KNN)** classifier. 


```
parameters = [1, 3, 5, 8, 10, 12, 15, 18, 20, 25, 30, 50,60,80,90,100]

def get_kneighbors_score(k):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X, y, cv=4)
    return scores.mean()


ACC_dev = []

for k in parameters:
    scores=get_kneighbors_score(k)
    ACC_dev.append(scores)
    
ACC_dev

```





## Visualizing the Results:

To better understand how k affects model performance, the validation curve of  **(accuracy vs. k)** is plotted


![validation curve of accuracy vs. k](https://github.com/RawanAlsaedi/INE-Data-Science-Bootcamp/blob/main/Tuning%20diabetes%20prediction%20model/img/validation%20curve%20of%20accuracy%20vs.%20k.png)



## Results

After testing various k values for the K-Nearest Neighbors (KNN) classifier, the following results were obtained:

- The **best value** of k was found to be 12, which resulted in an accuracy of 0.7491.

- The **validation curve** showed that accuracy improved as k increased from 1 to 12. However, after k = 12, the accuracy plateaued, suggesting no further improvement with larger values of k.
