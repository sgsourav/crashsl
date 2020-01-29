# =======================================================
# Linear Regression Experiment : Boston Housing Data
# =======================================================

# -------------------------------------------------------
# Install and Load Packages

# install.packages("vioplot")
# install.packages("corrplot")
# install.packages("scatterplot3d")
# install.packages("car")
# install.packages("leaps")
# install.packages("glmnet")
library(vioplot)
library(corrplot)
library(scatterplot3d)
library(car)
library(leaps)
library(glmnet)


# -------------------------------------------------------
# Load and explore the dataset

# install.packages("MASS")
library(MASS)
attach(Boston)
str(Boston)
help(Boston)

# Convert to a suitable dataframe
boston <- data.frame(Boston)
boston$chas <- as.factor(boston$chas) # qualitative predictor
boston$rad <- as.factor(boston$rad)   # qualitative predictor
str(boston)
summary(boston)


# -------------------------------------------------------
# Check and fix the distribution of the response variable

hist(boston$medv, breaks = 20, prob=TRUE, col="grey") 
lines(density(boston$medv), col="blue", lwd=2)
lines(density(boston$medv, adjust=2), lty="dotted", col="red", lwd=2)

# If it looks skewed, try to check a transformation
medv.trans <- log(boston$medv)
hist(medv.trans, breaks = 20, prob=TRUE, col="grey") 
lines(density(medv.trans), col="blue", lwd=2)
lines(density(medv.trans, adjust=2), lty="dotted", col="red", lwd=2)

# Check for skewness numerically to be sure
skewness(boston$medv)
skewness(medv.trans)


# -------------------------------------------------------
# Quickly check the features for potential dependence

corrplot.mixed(cor(boston[,-c(4,9)]))
corrplot(abs(cor(boston[,-c(4,9,14)])), order ="hclust")


# -------------------------------------------------------
# Build your First Model (FULL)

# Use log(medv) in the model as it is more symmetric
lmFit1 <- lm(log(medv) ~ ., data = boston)

summary(lmFit1)    # summary of linear model
vif(lmFit1)        # check multicollinearity
plot(lmFit1)       # check the various plots


# -------------------------------------------------------
# Build your Second Model

# Discard the features with high collinearity?
# Discard the features with low significance?
# You should be able to justify your choice.


# -------------------------------------------------------
# Compare each Model you build

# Check R^2, adj R^2, F (for each model) when you build.
# You may also use "leaps" library for subset selection.

# Check plot() for non-linear interactions.
# Include non-linear powers and interactions, if required.
# Carefully check if you are OverFitting the Training Data.


# -------------------------------------------------------
# Tune the Best Model you found

# Remove outliers and high-leverage points.
# Fit the best model to the final clean data.

# Re-iterate the whole process, if required.
