# =======================================================
# Model Selection : Part 1 : Manual Tuning
# =======================================================

# Load and explore the dataset
advData <- read.csv("Advertising.csv", header = TRUE)
summary(advData)

# Check the features
# install.packages("corrplot")
library(corrplot)
corrplot.mixed(cor(advData[,-c(1)]))
corrplot(cor(advData[,-c(1,5)]), order ="hclust")

# First Model (FULL)
lmFit1 <- lm(Sales ~ TV + Radio + Newspaper, data = advData)
summary(lmFit1)
plot(lmFit1)

# Check for multicollinearity
# install.packages("car")
library(car)
vif(lmFit1)

# Second Model
# Discard the features with low (> 0.1) significance
lmFit2 <- update(lmFit1, ~ . - (Newspaper))
summary(lmFit2)
vif(lmFit2)

# Check plots for potential non-linearity
plot(lmFit2)
# It seems that there is non-linear interaction

# Plot the remaining features against response
plot(advData$TV, advData$Sales)
plot(advData$Radio, advData$Sales)
# install.packages("scatterplot3d")
library(scatterplot3d)
scatterplot3d(advData$TV, advData$Radio, advData$Sales, angle = 45,
              pch = 19, highlight.3d = TRUE, box = FALSE,
              xlab = "TV", ylab = "Radio", zlab = "Sales")
# Seems that the interaction TV-Radio is non-linear

# Include non-linear interaction terms
lmFit3 <- update(lmFit2, ~ . + I(TV*Radio))  # try I(TV^2) and I(Radio^2) too
summary(lmFit3)
# Seems much better than lmFit2

# Remove outliers and high-leverage points
cd <- cooks.distance(lmFit3)
advData.clean <- advData[abs(cd) < 4/nrow(advData), ]
nrow(advData.clean)

# Fit the best model to the clean data
formula(lmFit3)
lmFit4 <- lm(formula(lmFit3), data = advData.clean)
summary(lmFit4)
plot(lmFit4)

# Predict using the Final Model
newData <- advData[c(50,100,150),]
predict(lmFit4, newdata = newData, interval = "confidence", level = 0.95)
predict(lmFit4, newdata = newData, interval = "prediction", level = 0.95)



# =======================================================
# Model Selection : Part 2 : Cross-Validation
# =======================================================

# Load the dataset
advData <- read.csv("Advertising.csv", header = TRUE)

# Eight models for linear regression
lmFit0 <- lm(Sales ~ 1, data = advData)
summary(lmFit0)

lmFit1 <- lm(Sales ~ TV, data = advData)
summary(lmFit1)

lmFit2 <- lm(Sales ~ Radio, data = advData)
summary(lmFit2)

lmFit3 <- lm(Sales ~ Newspaper, data = advData)
summary(lmFit3)

lmFit4 <- lm(Sales ~ TV + Radio, data = advData)
summary(lmFit4)

lmFit5 <- lm(Sales ~ TV + Newspaper, data = advData)
summary(lmFit5)

lmFit6 <- lm(Sales ~ Radio + Newspaper, data = advData)
summary(lmFit6)

lmFit7 <- lm(Sales ~ TV + Radio + Newspaper, data = advData)
summary(lmFit7)

# Compute Training and Validation MSE over different Models
nModels <- 7
nTrials <- 10
trnMSE <- matrix(data = NA, nrow = nTrials, ncol = nModels)
valMSE <- matrix(data = NA, nrow = nTrials, ncol = nModels)

for (t in 1:nTrials) {
  set.seed(t)
  train <- sample(nrow(advData), 0.7*nrow(advData), replace = FALSE)
  
  lmFit1 <- lm(Sales ~ TV, data = advData, subset = train)
  lmFit2 <- lm(Sales ~ Radio, data = advData, subset = train)
  lmFit3 <- lm(Sales ~ Newspaper, data = advData, subset = train)
  lmFit4 <- lm(Sales ~ TV + Radio, data = advData, subset = train)
  lmFit5 <- lm(Sales ~ TV + Newspaper, data = advData, subset = train)
  lmFit6 <- lm(Sales ~ Radio + Newspaper, data = advData, subset = train)
  lmFit7 <- lm(Sales ~ TV + Radio + Newspaper, data = advData, subset = train)
  
  error1 <- (advData$Sales - predict(lmFit1,advData))
  trnMSE[t,1] <- mean(error1[train]^2)
  valMSE[t,1] <- mean(error1[-train]^2)
  
  error2 <- (advData$Sales - predict(lmFit2,advData))
  trnMSE[t,2] <- mean(error2[train]^2)
  valMSE[t,2] <- mean(error2[-train]^2)

  error3 <- (advData$Sales - predict(lmFit3,advData))
  trnMSE[t,3] <- mean(error3[train]^2)
  valMSE[t,3] <- mean(error3[-train]^2)

  error4 <- (advData$Sales - predict(lmFit4,advData))
  trnMSE[t,4] <- mean(error4[train]^2)
  valMSE[t,4] <- mean(error4[-train]^2)

  error5 <- (advData$Sales - predict(lmFit5,advData))
  trnMSE[t,5] <- mean(error5[train]^2)
  valMSE[t,5] <- mean(error5[-train]^2)

  error6 <- (advData$Sales - predict(lmFit6,advData))
  trnMSE[t,6] <- mean(error6[train]^2)
  valMSE[t,6] <- mean(error6[-train]^2)
  
  error7 <- (advData$Sales - predict(lmFit7,advData))
  trnMSE[t,7] <- mean(error7[train]^2)
  valMSE[t,7] <- mean(error7[-train]^2)
}

# Plot the average Training and Validation MSE for different Models
plot(1:nModels, colMeans(trnMSE), 
     lwd = 2, col = "green", type = "line", 
     ylim = c(min(trnMSE),max(valMSE)), 
     xlab = "Model Complexity", ylab = "Mean Squared Error")
lines(1:nModels, colMeans(valMSE), lwd = 2, col = "red", type = "line")

# Plot the actual Training and Validation MSE for different Models
for (t in 1:nTrials) {
  lines(1:nModels, trnMSE[t,], lwd = 0.5, col = "limegreen", type = "line")
  lines(1:nModels, valMSE[t,], lwd = 0.5, col = "pink", type = "line")
}



# =======================================================
# Model Selection : Part 3 : Subset Selection
# =======================================================

# install.packages("leaps")
# install.packages("glmnet")
library(leaps)
library(glmnet)

# Load the dataset
advData <- read.csv("Advertising.csv", header = TRUE)

# Check the features
# install.packages("corrplot")
library(corrplot)
corrplot.mixed(cor(advData[,-c(1)]))
corrplot(cor(advData[,-c(1,5)]), order ="hclust")

# -------------------------------------------------------
# Complete Subset Selection for choosing model

regFitComplete <- regsubsets(Sales ~ TV + Radio + Newspaper,
                             data = advData, nvmax = 10)
regFitSummary <- summary(regFitComplete)
plot(regFitSummary$adjr2, pch = 19, type = "b", 
     xlab = "Number of Variables ", ylab = "Adjusted R-Squared")
plot(regFitComplete, scale = "adjr2")

# -------------------------------------------------------
# Forward Subset Selection for choosing model

regFitForward <- regsubsets(Sales ~ TV + Radio + Newspaper, 
                            data = advData, nvmax = 10, method = "forward")
regFitSummary <- summary(regFitForward)
plot(regFitSummary$adjr2, pch = 19, type = "b", 
     xlab = "Number of Variables ", ylab = "Adjusted R-Squared")
plot(regFitForward, scale = "adjr2")

# -------------------------------------------------------
# Backward Subset Selection for choosing model

regFitBackward <- regsubsets(Sales ~ TV + Radio + Newspaper, 
                             data = advData, nvmax = 10, method = "backward")
regFitSummary <- summary(regFitBackward)
plot(regFitSummary$adjr2, pch = 19, type = "b", 
     xlab = "Number of Variables ", ylab = "Adjusted R-Squared")
plot(regFitBackward, scale = "adjr2")

# -------------------------------------------------------
# Experiment with extended set of Predictors

regFitComplete <- regsubsets(Sales ~ TV + Radio + Newspaper
                                     + I(TV^2) + I(Radio^2) + I(TV*Radio),
                             data = advData, nvmax = 10)
regFitSummary <- summary(regFitComplete)
plot(regFitSummary$adjr2, pch = 19, type = "b", 
     xlab = "Number of Variables ", ylab = "Adjusted R-Squared")
plot(regFitComplete, scale = "adjr2")



# =======================================================
# Model Selection : Part 4 : Shrinkage or Regularization
# =======================================================

# install.packages("glmnet")
library(glmnet)

# -------------------------------------------------------
# Shrinkage on a cooked-up data (just as an example)

dsize <- 50
xvals <- sort(runif(dsize, 0, 1))
yvals <- sin(4*xvals) + rnorm(dsize, 0, 0.2)
plot(xvals, yvals, pch = 19, xlab = "X", ylab = "Y")

# Build Polynomial Regression Models of varying degree
degPoly <- 20
polyFit <- lm(yvals ~ poly(xvals, degree = degPoly, raw = TRUE))
as.data.frame(polyFit$coefficients)

# Build a Shrunken Polynomial Regression Model
# L2 regularization for shrinking -- Ridge
degPoly <- 10
x <- model.matrix(yvals ~ poly(xvals, degree = degPoly, raw = TRUE))
y <- as.matrix(yvals)
ridgeFit <- glmnet(x, y, alpha = 0)
plot(ridgeFit)

ridgeFitCV <- cv.glmnet(x, y, alpha = 0)
plot(ridgeFitCV)
ridgeFitCV$lambda.min
coef(ridgeFitCV, s = ridgeFitCV$lambda.min)
ridgeFitCV$lambda.1se
coef(ridgeFitCV, s = ridgeFitCV$lambda.1se)

# Build a Shrunken Polynomial Regression Model
# L1 regularization for shrinking -- Lasso
degPoly <- 10
x <- model.matrix(yvals ~ poly(xvals, degree = degPoly, raw = TRUE))
y <- as.matrix(yvals)
lassoFit <- glmnet(x, y, alpha = 1)
plot(lassoFit)

lassoFitCV <- cv.glmnet(x, y, alpha = 1)
plot(lassoFitCV)
lassoFitCV$lambda.min
coef(lassoFitCV, s = lassoFitCV$lambda.min)
lassoFitCV$lambda.1se
coef(lassoFitCV, s = lassoFitCV$lambda.1se)

# -------------------------------------------------------
# Shrinkage on the advData dataset

# L2 regularization for shrinking -- Ridge
x <- model.matrix(Sales ~ TV + Radio + Newspaper, advData)
y <- advData$Sales
ridgeFit <- glmnet(x, y, alpha = 0)
plot(ridgeFit)

ridgeFitCV <- cv.glmnet(x, y, alpha = 0)
plot(ridgeFitCV)
ridgeFitCV$lambda.min
coef(ridgeFitCV, s = ridgeFitCV$lambda.min)
ridgeFitCV$lambda.1se
coef(ridgeFitCV, s = ridgeFitCV$lambda.1se)

# L1 regularization for shrinking -- Lasso
x <- model.matrix(Sales ~ TV + Radio + Newspaper, advData)
y <- advData$Sales
lassoFit <- glmnet(x, y, alpha = 1)
plot(lassoFit)

lassoFitCV <- cv.glmnet(x, y, alpha = 1)
plot(lassoFitCV)
lassoFitCV$lambda.min
coef(lassoFitCV, s = lassoFitCV$lambda.min)
lassoFitCV$lambda.1se
coef(lassoFitCV, s = lassoFitCV$lambda.1se)
