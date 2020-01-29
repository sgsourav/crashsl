# =======================================================
# Classification : Simple Logistic Regression
# =======================================================

# Load the dataset and explore
# install.packages("ISLR")
library("ISLR")
attach(Default)
help(Default)
defData <- data.frame(Default)
str(defData)
summary(defData)


# -------------------------------------------------------
# Characteristics of Default vs Predictors/Features

# Default vs Balance
plot(defData$default, defData$balance,
     xlab = "Default", ylab = "Balance")
plot(defData$balance, defData$default, 
     pch = 20, lwd = 0.1,
     col = as.integer(defData$default),
     xlab = "Balance", ylab = "Default (YES / NO)")

# Default vs Income
plot(defData$default, defData$income,
     xlab = "Default", ylab = "Income")
plot(defData$income, defData$default, 
     pch = 20, lwd = 0.1,
     col = as.integer(defData$default),
     xlab = "Income", ylab = "Default (YES / NO)")

# Default vs Student
plot(defData$student, defData$default,
     xlab = "Student", ylab = "Default")
plot(defData$default, defData$student,
     xlab = "Default", ylab = "Student")


# -------------------------------------------------------
# Fit logistic model for Default ~ Balance

# Default vs Balance
plot(defData$default, defData$balance,
     xlab = "Default", ylab = "Balance")
plot(defData$balance, defData$default, 
     pch = 20, lwd = 0.1,
     col = as.integer(defData$default),
     xlab = "Balance", ylab = "Default (YES / NO)")

# Logistic Model
logFit <- glm(default ~ balance, 
              data = defData,                    # fit on the full data
              family = binomial(link='logit'))   # link function: logit
summary(logFit)


# -------------------------------------------------------
# Predict using the model

probData <- predict(logFit, type = "response", newdata = defData)

# Convert probabilities to predictions
predData <- rep("No", nrow(defData))
predData[probData > 0.5] = "Yes"      # set threshold 0.5
predData <- as.factor(predData)

# Confusion matrix for predictions
table(defData$default, predData)
summary(defData$default)


# -------------------------------------------------------
# Performance measures for Prediction

cm <- table(defData$default, predData)
TP <- cm[2,2]  # True Positive (Yes predicted as Yes)
TN <- cm[1,1]  # True Negative (No predicted as No)
FP <- cm[1,2]  # False Positive (No predicted as Yes) -- Type I error
FN <- cm[2,1]  # False Negative (Yes predicted as No) -- Type II error

# Classification Accuracy
(TN + TP) / (TN + TP + FN + FP)    # Correct Classification / Total
mean(predData == defData$default)  # Classification accuracy (alt.)

# False Positive Rate (fpr) / Type I error
FP / (TN + FP)      # False Positive / Total Negative
1 - FP / (TN + FP)  # Specificity = 1 - (Type I error)

# True Positive Rate / Sensitivity / Recall / Power
TP / (TP + FN)      # True Positive / Total Positive
1 - TP / (TP + FN)  # Type II error = 1 - Sensitivity


# -------------------------------------------------------
# Different thresholds for "probability to prediction"

threshold <- 0.01                    # set threshold and experiment

probData <- predict(logFit, type = "response", newdata = defData)
predData <- rep("No", nrow(defData))
predData[probData > threshold] = "Yes"      
predData <- as.factor(predData)

table(defData$default, predData)    # confusion matrix for predictions
summary(defData$default)            # actual distribuion of response


# -------------------------------------------------------
# Check Receiver Operating Characteristic (ROC)

# install.packages("ROCR")
library(ROCR)
probData <- predict(logFit, type = "response", newdata = defData)
predData <- prediction(probData, defData$default)
perfData <- performance(predData, measure = "tpr", x.measure = "fpr")
plot(perfData, main = "ROC Curve for Logistic Regression", colorize = TRUE)
abline(a = 0, b= 1)


# -------------------------------------------------------
# Optimal threshold for "probability to prediction"

# Check individual cutoffs vs corresponding FPR/TPR
str(perfData)
cutoffs <- data.frame(cut = perfData@alpha.values[[1]], 
                      fpr = perfData@x.values[[1]], 
                      tpr = perfData@y.values[[1]])
head(cutoffs)

# Function to find the "optimal" cutoff
# The one closest to (FPR = 0, TPR = 1)
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

# Find optimal cutoff for the model
optThresh <- opt.cut(perfData, predData)
optThresh

# Set the "optimal" cutoff as thereshold
threshold <- optThresh["cutoff",]
probData <- predict(logFit, type = "response", newdata = defData)
predData <- rep("No", nrow(defData))
predData[probData > threshold] = "Yes"      
predData <- as.factor(predData)

table(defData$default, predData)    # confusion matrix for predictions
summary(defData$default)            # actual distribuion of response
mean(predData == defData$default)   # classification accuracy (alt.)


# =======================================================
# Classification : Multiple Logistic Regression
# =======================================================

# Load the dataset and explore
# install.packages("ISLR")
library("ISLR")
attach(Default)
help(Default)
defData <- data.frame(Default)
str(defData)
summary(defData)


# -------------------------------------------------------
# Characteristics of Default vs Predictors/Features

plot(defData$default, defData$balance,
     xlab = "Default", ylab = "Balance")
plot(defData$balance, defData$default, 
     pch = 20, lwd = 0.1,
     col = as.integer(defData$default),
     xlab = "Balance", ylab = "Default (YES / NO)")

plot(defData$default, defData$income,
     xlab = "Default", ylab = "Income")
plot(defData$income, defData$default, 
     pch = 20, lwd = 0.1,
     col = as.integer(defData$default),
     xlab = "Income", ylab = "Default (YES / NO)")

plot(defData$student, defData$default,
     xlab = "Student", ylab = "Default")
plot(defData$default, defData$student,
     xlab = "Default", ylab = "Student")


# -------------------------------------------------------
# Fit logistic model for Default ~ Balance + Income + Student

logFit <- glm(default ~ balance + income + student, 
              data = defData,                    # fit on the full data
              family = binomial(link='logit'))   # link function: logit
summary(logFit)


# -------------------------------------------------------
# Predict using the model

probData <- predict(logFit, type = "response", newdata = defData)

# Convert probabilities to predictions
predData <- rep("No", nrow(defData))
predData[probData > 0.5] = "Yes"      # set threshold 0.5
predData <- as.factor(predData)

# Confusion matrix for predictions
table(defData$default, predData)
summary(defData$default)


# -------------------------------------------------------
# Check Receiver Operating Characteristic (ROC)

# install.packages("ROCR")
library(ROCR)
probData <- predict(logFit, type = "response", newdata = defData)
predData <- prediction(probData, defData$default)
perfData <- performance(predData, measure = "tpr", x.measure = "fpr")
plot(perfData, main = "ROC Curve for Logistic Regression", colorize = TRUE)
abline(a = 0, b= 1)


# =======================================================
# Classification : Comparing Logistic Models
# =======================================================

# Load the dataset and explore
# install.packages("ISLR")
library("ISLR")
attach(Default)
help(Default)
defData <- data.frame(Default)
str(defData)
summary(defData)


# -------------------------------------------------------
# Train different logistic models on the full dataset

# Model 0 : default ~ 1 : NULL model
logFit0 <- glm(default ~ 1, data = defData, family = binomial(link='logit'))
summary(logFit0)

# Model 1 : default ~ balance
logFit1 <- glm(default ~ balance, data = defData, family = binomial(link='logit'))
summary(logFit1)

# Model 2 : default ~ income
logFit2 <- glm(default ~ income, data = defData, family = binomial(link='logit'))
summary(logFit2)

# Model 3 : default ~ student
logFit3 <- glm(default ~ student, data = defData, family = binomial(link='logit'))
summary(logFit3)

# Model 4 : default ~ balance + income
logFit4 <- glm(default ~ balance + income, data = defData, family = binomial(link='logit'))
summary(logFit4)

# Model 5 : default ~ balance + student
logFit5 <- glm(default ~ balance + student, data = defData, family = binomial(link='logit'))
summary(logFit5)

# Model 6 : default ~ income + student
logFit6 <- glm(default ~ income + student, data = defData, family = binomial(link='logit'))
summary(logFit6)

# Model 7 : default ~ balance + income + student : FULL model
logFit7 <- glm(default ~ balance + income + student, data = defData, family = binomial(link='logit'))
summary(logFit7)


# -------------------------------------------------------
# Check Receiver Operating Characteristic (ROC)

# install.packages("ROCR")
library(ROCR)

# Model 0 : default ~ 1 : NULL model
probData0 <- predict(logFit0, type = "response", newdata = defData)
predData0 <- prediction(probData0, defData$default)
perfData0 <- performance(predData0, measure = "tpr", x.measure = "fpr")
plot(perfData0, main = "ROC Curve for Logistic Regression")
perf.auc0 <- performance(predData0, measure = "auc")
perf.auc0@y.values[[1]]

# Model 1 : default ~ balance
probData1 <- predict(logFit1, type = "response", newdata = defData)
predData1 <- prediction(probData1, defData$default)
perfData1 <- performance(predData1, measure = "tpr", x.measure = "fpr")
plot(perfData1, colorize = TRUE, add = TRUE)
perf.auc1 <- performance(predData1, measure = "auc")
perf.auc1@y.values[[1]]

# Model 2 : default ~ income
probData2 <- predict(logFit2, type = "response", newdata = defData)
predData2 <- prediction(probData2, defData$default)
perfData2 <- performance(predData2, measure = "tpr", x.measure = "fpr")
plot(perfData2, colorize = TRUE, add = TRUE)
perf.auc2 <- performance(predData2, measure = "auc")
perf.auc2@y.values[[1]]

# Model 3 : default ~ student
probData3 <- predict(logFit3, type = "response", newdata = defData)
predData3 <- prediction(probData3, defData$default)
perfData3 <- performance(predData3, measure = "tpr", x.measure = "fpr")
plot(perfData3, colorize = TRUE, add = TRUE)
perf.auc3 <- performance(predData3, measure = "auc")
perf.auc3@y.values[[1]]

# Model 4 : default ~ balance + income
probData4 <- predict(logFit4, type = "response", newdata = defData)
predData4 <- prediction(probData4, defData$default)
perfData4 <- performance(predData4, measure = "tpr", x.measure = "fpr")
plot(perfData4, colorize = TRUE, add = TRUE)
perf.auc4 <- performance(predData4, measure = "auc")
perf.auc4@y.values[[1]]

# Model 5 : default ~ balance + student
probData5 <- predict(logFit5, type = "response", newdata = defData)
predData5 <- prediction(probData5, defData$default)
perfData5 <- performance(predData5, measure = "tpr", x.measure = "fpr")
plot(perfData5, colorize = TRUE, add = TRUE)
perf.auc5 <- performance(predData5, measure = "auc")
perf.auc5@y.values[[1]]

# Model 6 : default ~ balance + student
probData6 <- predict(logFit6, type = "response", newdata = defData)
predData6 <- prediction(probData6, defData$default)
perfData6 <- performance(predData6, measure = "tpr", x.measure = "fpr")
plot(perfData6, colorize = TRUE, add = TRUE)
perf.auc6 <- performance(predData6, measure = "auc")
perf.auc6@y.values[[1]]

# Model 7 : default ~ balance + income + student : FULL model
probData7 <- predict(logFit7, type = "response", newdata = defData)
predData7 <- prediction(probData7, defData$default)
perfData7 <- performance(predData7, measure = "tpr", x.measure = "fpr")
plot(perfData7, colorize = TRUE, add = TRUE)
perf.auc7 <- performance(predData7, measure = "auc")
perf.auc7@y.values[[1]]
