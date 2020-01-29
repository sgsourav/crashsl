# =======================================================
# Linear Regression : Part 1 : Introduction
# =======================================================

# Install and Load Packages
# install.packages("vioplot")
# install.packages("corrplot")
library(vioplot)
library(corrplot)

# Load the dataset
advData <- read.csv("Advertising.csv", header = TRUE)

# Basic exploration
dim(advData)
names(advData)
str(advData)
head(advData)
summary(advData)

# Exploring Sales
boxplot(advData$Sales, horizontal=TRUE, 
        col = "lightblue", main = "Boxplot of Sales")
hist(advData$Sales, prob = TRUE, 
     col = "lightgreen", main = "Histogram of Sales")
lines(density(advData$Sales), col = "red", lwd = 3)
lines(density(advData$Sales, adjust = 2), 
      lty = "dotted", col = "magenta", lwd = 3)
vioplot(advData$Sales, horizontal=TRUE, col = "pink")
title(main = "Violin Plot of Sales")

# Exploring each Feature
par(mfrow=c(2,2))
boxplot(advData$TV, horizontal=TRUE, 
        col = "lightblue", main = "Boxplot")
hist(advData$TV, prob = TRUE, col = "lightgreen", 
     main = "Histogram", xlab = "", ylab = "")
lines(density(advData$TV), col = "red", lwd = 3)
vioplot(advData$TV, horizontal=TRUE, col = "pink")
title(main = "Violin Plot")
plot(advData$TV, advData$Sales, pch = 19, col = "red",
     main = "TV vs Sales", xlab = "", ylab = "")
par(mfrow=c(1,1))

# Sales vs TV Advertisement
plot(advData$TV, advData$Sales,
     pch = 19, col = "red",
     xlab = "TV Advertisement", ylab = "Sales")

# Sales vs Radio Advertisement
plot(advData$Radio, advData$Sales, 
     pch = 19, col = "red",
     xlab = "Radio Advertisement", ylab = "Sales")

# Sales vs Newspaper Advertisement
plot(advData$Newspaper, advData$Sales, 
     pch = 19, col = "red",
     xlab = "Newspaper Advertisement", ylab = "Sales")

# Quick Pairwise Plots
pairs(advData[,2:5], pch = 19, col = "red")

# Quick Correlation
corrplot.mixed(cor(advData[,2:5]))

# Simple linear regression
plot(advData$TV, advData$Sales,
     pch = 19, col = "blue",
     xlab = "TV Advertisement", ylab = "Sales")

lmFit <- lm(Sales ~ TV, data = advData)
lmFit$coefficients
abline(lmFit$coefficients, col="black", lwd=3)

# Create a random Train Data
trainIndex <- sample(1:nrow(advData),100,replace = FALSE)
advTrain <- advData[trainIndex,]
plot(advData$TV, advData$Sales, 
     col = "pink", pch = 21, lwd = 2,
     xlab = "TV", ylab = "Sales")
points(advTrain$TV, advTrain$Sales, 
       type="p", pch=21, col="red", bg="red")

# Simple linear regression on Train Data
lmTrain <- lm(Sales ~ TV, data = advTrain)
abline(lmTrain$coefficients, col="red", lwd="2")

# Simple linear regression on Full Data
lmData <- lm(Sales ~ TV, data = advData)
abline(lmData$coefficients, col="black", lwd="2")

# Compare the coefficients
lmTrain$coefficients
lmData$coefficients

# Simple linear regression with many random Train Datasets
plot(advData$TV, advData$Sales, 
     type="p", pch = 21, col="pink", lwd = 2)
lmData <- lm(Sales ~ TV, data = advData)
lmCoef <- as.array(lmData$coefficients)
for (i in 1:100) {
  trainIndex <- sample(1:nrow(advData),100,replace = FALSE)
  advTrain <- advData[trainIndex,]
  lmTrain <- lm(Sales ~ TV, data = advTrain)
  trainCoef <- as.array(lmTrain$coefficients)
  lmCoef <- rbind(lmCoef, trainCoef)
  abline(lmTrain$coefficients, col="red", lwd="0.1")
}

# Simple linear regression on Full Data
abline(lmData$coefficients, col="black", lwd="3")

# Check the distribution of the coefficients
beta.0 <- lmCoef[2:nrow(lmCoef),1]
beta.1 <- lmCoef[2:nrow(lmCoef),2]
summary(beta.0)
summary(beta.1)
vioplot(beta.0, horizontal=TRUE, col = "pink")
title(main = "Distribution of Intercept")
vioplot(beta.1, horizontal=TRUE, col = "pink")
title(main = "Distribution of Slope")

# Explore the complete lm() output
summary(lmFit)
plot(lmFit)


# =======================================================
# Linear Regression : Part 2 : Basic Concepts
# =======================================================

# Load the dataset
advData <- read.csv("Advertising.csv", header = TRUE)

# Sales vs TV Advertisement
plot(advData$Newspaper, advData$Sales,
     pch = 19, col = "red",
     xlab = "TV Advertisement", ylab = "Sales")

# Simple linear regression
lmFit <- lm(Sales ~ Newspaper, data = advData)
lmFit$coefficients
abline(lmFit$coefficients, col="black", lwd=3)

# Explore the lm() output
lmFit$coefficients                          # coefficients (beta)
plot(advData$TV, lmFit$fitted.values)       # fitted values (yhat)
plot(advData$TV, lmFit$residuals)           # residuals (y - yhat)

summary(lmFit)
plot(lmFit)


# Sales vs TV and Radio Advertisement
# install.packages("scatterplot3d")
library(scatterplot3d)
scatterplot3d(advData$TV, advData$Radio, advData$Sales, angle = 0,
              pch = 19, highlight.3d = TRUE, box = FALSE,
              xlab = "TV", ylab = "Radio", zlab = "Sales")

# Multiple linear regression
lmFit <- lm(Sales ~ TV + Radio, data = advData)
lmFit$coefficients
plot3d <- scatterplot3d(advData$TV, advData$Radio, advData$Sales, angle = 0,
                        pch = 19, highlight.3d = TRUE, box = FALSE,
                        xlab = "TV", ylab = "Radio", zlab = "Sales")
plot3d$plane3d(lmFit$coefficients)

summary(lmFit)
plot(lmFit)


# Multiple linear regression with ALL features
lmFit <- lm(Sales ~ TV+Newspaper, data = advData)
summary(lmFit)

# Checking for cross-correlations
# install.packages("corrplot")
library(corrplot)
corrplot.mixed(cor(advData[,2:5]))

# Checking for multicollinearity
# install.packages("car")
library(car)
vif(lmFit)



# =======================================================
# Linear Regression : Part 3 : Confidence and Prediction
# =======================================================

# Load the dataset
advData <- read.csv("Advertising.csv", header = TRUE)

# Simple linear regression
lmFit <- lm(Sales ~ TV, data = advData)
summary(lmFit)

# Predict using the model with desired "level" of precision
newData <- advData[c(50,100,150),]
predict(lmFit, newdata = newData, interval = "confidence", level = 0.95)
predict(lmFit, newdata = newData, interval = "prediction", level = 0.95)
# and notice that prediction interval is larger than confidence interval

# Confidence interval estimates the expectation E(y|x)
# Prediction interval estimates a single output y|x
# Both depend on (in different manners) the following
coef(summary(lmFit))    # the standard errors of coefficients
confint(lmFit)          # the confidence intervals of coefficients
var(lmFit$residuals)    # variance of residuals


# Picture worth a thousand words
plot(advData$TV, advData$Sales,
     pch = 19, col = "lightblue",
     xlab = "TV Advertisement", ylab = "Sales")
abline(lmFit$coefficients, col = "red", lwd = 2)
confint <- predict(lmFit, newdata = advData, interval = "confidence", level = 0.95)
points(advData$TV, confint[,2], col="red", pch = 46)
points(advData$TV, confint[,3], col="red", pch = 46)
predint <- predict(lmFit, newdata = advData, interval = "prediction", level = 0.95)
points(advData$TV, predint[,2], col="green", pch = 46)
points(advData$TV, predint[,3], col="green", pch = 46)


# Multiple linear regression with two features
lmFit <- lm(Sales ~ TV + Radio, data = advData)
summary(lmFit)

# Predict using the model
newData <- advData[c(50,100,150),]
predict(lmFit, newdata = newData, interval = "confidence", level = 0.95)
predict(lmFit, newdata = newData, interval = "prediction", level = 0.95)

coef(summary(lmFit))    # the standard errors of coefficients
var(lmFit$residuals)    # variance of residuals


# Multiple linear regression with ALL features
lmFit <- lm(Sales ~ TV + Radio + Newspaper, data = advData)
summary(lmFit)

# Predict using the model
newData <- advData[c(50,100,150),]
predict(lmFit, newdata = newData, interval = "confidence", level = 0.95)
predict(lmFit, newdata = newData, interval = "prediction", level = 0.95)

coef(summary(lmFit))    # the standard errors of coefficients
var(lmFit$residuals)    # variance of residuals
