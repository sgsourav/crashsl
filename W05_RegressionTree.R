# =======================================================
# Tree Model : Part 4 : Decision Tree (Regression)
# =======================================================

# Load the dataset and explore
advData <- read.csv("Advertising.csv", header = TRUE)
str(advData)
summary(advData)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(advData), 0.7*nrow(advData), replace = FALSE)
advTrain <- advData[train,]
advValid <- advData[-train,]
summary(advTrain)
summary(advValid)


# Build a Regression Tree
# install.packages("tree")
library("tree")
treeFit <- tree(Sales ~ TV + Radio + Newspaper, data = advTrain)

# Visualize the tree
plot(treeFit)
text(treeFit, pretty = FALSE)

# Check the output
summary(treeFit)
treeFit


# Predict using the tree model
predTrain <- predict(treeFit, advTrain, type = "vector")  # on train set
predValid <- predict(treeFit, advValid, type = "vector")  # on validation set

# Prediction accuracy using RSS
sum((predTrain - advTrain$Sales)^2)     # on train set
sum((predValid - advValid$Sales)^2)     # on validation set



# =======================================================
# Tree Model : Part 5 : Pruning a Regression Tree
# =======================================================

# Load the dataset and explore
advData <- read.csv("Advertising.csv", header = TRUE)
str(advData)
summary(advData)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(advData), 0.7*nrow(advData), replace = FALSE)
advTrain <- advData[train,]
advValid <- advData[-train,]
summary(advTrain)
summary(advValid)


# Build a "default" Regression Tree
treeFit <- tree(Sales ~ TV + Radio + Newspaper, data = advTrain)
plot(treeFit)

predTrain <- predict(treeFit, advTrain, type = "vector")  # prediction on train set
sum((predTrain - advTrain$Sales)^2)                       # RSS (or deviance)
predValid <- predict(treeFit, advValid, type = "vector")  # prediction on validation set
sum((predValid - advValid$Sales)^2)                       # RSS (or deviance)


# Build a "large" Regression Tree
ltreeFit <- tree(Sales ~ TV + Radio + Newspaper, data = advTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(advTrain),
                                        mincut = 1,
                                        minsize = 2,
                                        mindev = 0))
plot(ltreeFit)

predTrain <- predict(ltreeFit, advTrain, type = "vector")  # prediction on train set
sum((predTrain - advTrain$Sales)^2)                        # RSS (or deviance)
predValid <- predict(ltreeFit, advValid, type = "vector")  # prediction on validation set
sum((predValid - advValid$Sales)^2)                        # RSS (or deviance)


# Build a "small" Regression Tree
streeFit <- tree(Sales ~ TV + Radio + Newspaper, data = advTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(advTrain),
                                        mincut = 0.1 * nrow(advTrain),
                                        minsize = 0.2 * nrow(advTrain),
                                        mindev = 0.1))
plot(streeFit)

predTrain <- predict(streeFit, advTrain, type = "vector")  # prediction on train set
sum((predTrain - advTrain$Sales)^2)                        # RSS (or deviance)
predValid <- predict(streeFit, advValid, type = "vector")  # prediction on validation set
sum((predValid - advValid$Sales)^2)                        # RSS (or deviance)


# Build a "pruned" Regression Tree over train set
ltreeFit <- tree(Sales ~ TV + Radio + Newspaper, data = advTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(advTrain),
                                        mincut = 1,
                                        minsize = 2,
                                        mindev = 0))
plot(ltreeFit)

cvTree <- cv.tree(ltreeFit, FUN = prune.tree, K = 10)     # K-fold Cross-Validation
cbind(cvTree$size, cvTree$dev, cvTree$k)                  # check cvTree output
plot(cvTree$size, cvTree$dev, type="b")                   # plot deviance vs size
plot(cvTree$k, cvTree$dev, type="b")                      # plot deviance vs alpha

bestSize <- 10  # choose this parameter carefully, based on the cvTree output
ptreeFit <- prune.tree(ltreeFit, best = bestSize)         # prune tree to best size
plot(ptreeFit)
text(ptreeFit, pretty = FALSE)

predTrain <- predict(ptreeFit, advTrain, type = "vector")  # prediction on train set
sum((predTrain - advTrain$Sales)^2)                        # RSS (or deviance)
predValid <- predict(ptreeFit, advValid, type = "vector")  # prediction on validation set
sum((predValid - advValid$Sales)^2)                        # RSS (or deviance)



# =======================================================
# Tree Model : Part 6 : Bagging and Random Forest
# =======================================================

# install.packages("randomForest")
library(randomForest)

# Load the dataset and explore
advData <- read.csv("Advertising.csv", header = TRUE)
str(advData)
summary(advData)

# -------------------------------------------------------
# Split into Train and Validation sets

# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(advData), 0.7*nrow(advData), replace = FALSE)
advTrain <- advData[train,]
advValid <- advData[-train,]
summary(advTrain)
summary(advValid)


# -------------------------------------------------------
# Build a Bagging Model on train set

# Each node splits with all features
bagFit <- randomForest(Sales ~ TV + Radio + Newspaper,   # formula
                       data = advTrain,                  # data set
                       ntree = 500,                      # number of trees
                       mtry = 3)                         # variables for split
bagFit

predTrain <- predict(bagFit, advTrain, type="response")  # prediction on train set
sum((predTrain - advTrain$Sales)^2)                      # RSS (or deviance)
predValid <- predict(bagFit, advValid, type="response")  # prediction on validation set
sum((predValid - advValid$Sales)^2)                      # RSS (or deviance)

varImpPlot(bagFit)        # importance of the variables in the model (visual)


# -------------------------------------------------------
# Build a Random Forest Model on train set

# Each node splits with subset of features
rfFit <- randomForest(Sales ~ TV + Radio + Newspaper,    # formula
                      data = advTrain,                   # data set
                      ntree = 500,                       # number of trees
                      mtry = 2)                          # variables for split
rfFit

predTrain <- predict(rfFit, advTrain, type="response")   # prediction on train set
sum((predTrain - advTrain$Sales)^2)                      # RSS (or deviance)
predValid <- predict(rfFit, advValid, type="response")   # prediction on validation set
sum((predValid - advValid$Sales)^2)                      # RSS (or deviance)

varImpPlot(rfFit)         # importance of the variables in the model (visual)


# -------------------------------------------------------
# Boruta : Variable Importance at a Glance

# install.packages("Boruta")
library("Boruta")
borutaFit <- Boruta(Sales ~ TV + Radio + Newspaper,            # formula
                    data = advTrain)                           # data set

getSelectedAttributes(borutaFit)              # extract important variables
attStats(borutaFit)                           # full finalBoruta statistics
borutaFit                                     # importance of variables (values)
plot(borutaFit)                               # importance of variables (visual)
