---
title: "Practical Machine Learning - Week 4 Project"
author: "Wenju Cui"
date: "July 6, 2022"
output: 
  html_document: 
  #pdf_document:
    keep_md: yes
---



## 1. Loading and preprocessing the data

### 1.1 Load the data and packages
View data and convert "#DIV/0!" or blanks to NA value


```r
train <- read.csv("pml-training.csv", na.strings = c("#DIV/0!", "NA", ""))
test  <- read.csv("pml-testing.csv",  na.strings = c("#DIV/0!", "NA", ""))

library(caret)
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## randomForest 4.7-1.1
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

### 1.2 Drop columns
Drop colums with NA and columns irrelevant to modeling such as "user_name", "X", etc. 
Apply same treatment to the test data


```r
No_na_cols <- sapply(train, function(x)all(!is.na(x)))
train_clean_1 <- train[, No_na_cols]
train_clean_2 <- train_clean_1[, c(8:60)]

test_clean_1 <- test[, No_na_cols]
test_clean_2 <- test_clean_1[, c(8:60)]
```

### 1.3 Split training data


```r
set.seed(1234)

inTrain <- createDataPartition(y=train_clean_2$classe, p=0.75, list=FALSE)
train_train <- train_clean_2[inTrain,]
train_test  <- train_clean_2[-inTrain,]

# check level of classe variable
table(train_train$classe)
```

```
## 
##    A    B    C    D    E 
## 4185 2848 2567 2412 2706
```


## 2. Modeling

### 2.1 Decision Tree

```r
# Fit model
modfit_tree <- train(classe ~., method='rpart', data=train_train)

# Prediction
tree_Prediction <- predict(modfit_tree, train_test)

# Confusion matrix
confusionMatrix(tree_Prediction,as.factor(train_test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1274  390  401  386  141
##          B   18  344   36  131  127
##          C  100  215  418  287  244
##          D    0    0    0    0    0
##          E    3    0    0    0  389
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4945          
##                  95% CI : (0.4804, 0.5086)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3385          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9133  0.36249  0.48889   0.0000  0.43174
## Specificity            0.6244  0.92111  0.79106   1.0000  0.99925
## Pos Pred Value         0.4915  0.52439  0.33070      NaN  0.99235
## Neg Pred Value         0.9477  0.85758  0.87995   0.8361  0.88652
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2598  0.07015  0.08524   0.0000  0.07932
## Detection Prevalence   0.5285  0.13377  0.25775   0.0000  0.07993
## Balanced Accuracy      0.7688  0.64180  0.63997   0.5000  0.71550
```

```r
# Plot the tree
plot(modfit_tree$finalModel, uniform=TRUE, main="Classfication Tree")
text(modfit_tree$finalModel, use.n=TRUE, all=TRUE, cex=0.8)
```

![](PML-week4_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

### 2.2 Random Forest

```r
# Fit model
modfit_rf2 <- randomForest(as.factor(train_train$classe) ~ ., data=train_train, method="class")
# tried using train(classe ~ ., data =train_train, method='rf'), running time is too long

# Prediction
rf2_Prediction <- predict(modfit_rf2, train_test)

# Confusion matrix
confusionMatrix(rf2_Prediction, as.factor(train_test$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  943    8    0    0
##          C    0    1  846    6    0
##          D    0    0    1  798    0
##          E    0    0    0    0  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9957          
##                  95% CI : (0.9935, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9946          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9937   0.9895   0.9925   1.0000
## Specificity            0.9986   0.9980   0.9983   0.9998   1.0000
## Pos Pred Value         0.9964   0.9916   0.9918   0.9987   1.0000
## Neg Pred Value         1.0000   0.9985   0.9978   0.9985   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1923   0.1725   0.1627   0.1837
## Detection Prevalence   0.2855   0.1939   0.1739   0.1629   0.1837
## Balanced Accuracy      0.9993   0.9958   0.9939   0.9961   1.0000
```


## 3. Conclusion
Modeling using Random Forest has a way much better performanc than modeling using decision tree. Random Forest yeilds an accuracy = 0.9957, whereas decision tree accuracy = 0.4945. Random Forest model is chosen and applied to the test data for new prediction. 


## 4. New Prediction

```r
predict <- predict(modfit_rf2, test_clean_2)
predict
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
