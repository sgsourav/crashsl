# =======================================================
# Tree Model : Part 1 : Decision Tree (Classification)
# =======================================================

# Load the dataset and explore
carData <- read.csv("CarEvaluation.csv", header = TRUE)
str(carData)
summary(carData)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(carData), 0.7*nrow(carData), replace = FALSE)
carTrain <- carData[train,]
carValid <- carData[-train,]
summary(carTrain)
summary(carValid)

# Visualize the train dataset
plot(carTrain$BuyingPrice, carTrain$Condition)
plot(carTrain$Maintenance, carTrain$Condition)
plot(carTrain$NumDoors, carTrain$Condition)
plot(carTrain$NumPersons, carTrain$Condition)
plot(carTrain$BootSpace, carTrain$Condition)
plot(carTrain$Safety, carTrain$Condition)

# Build a Classification Tree
# install.packages("tree")
library("tree")
treeFit <- tree(Condition ~ ., data = carTrain)

# Visualize the tree
plot(treeFit)
text(treeFit, pretty = FALSE)

# Predict using the tree model
predTrain <- predict(treeFit, carTrain, type = "class")  # on train set
predValid <- predict(treeFit, carValid, type = "class")  # on validation set

# Confusion matrix for predictions
table(carTrain$Condition, predTrain)      # on train set
table(carValid$Condition, predValid)      # on validation set

# Classification accuracy
mean(predTrain == carTrain$Condition)     # on train set
mean(predValid == carValid$Condition)     # on validation set

# Check the output
summary(treeFit)
treeFit



# =======================================================
# Tree Model : Part 2 : Pruning a Decision Tree
# =======================================================

# install.packages("tree")
library("tree")

# Load the dataset and explore
carData <- read.csv("CarEvaluation.csv", header = TRUE)
str(carData)
summary(carData)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(carData), 0.7*nrow(carData), replace = FALSE)
carTrain <- carData[train,]
carValid <- carData[-train,]
summary(carTrain)
summary(carValid)


# Build a "default" Classification Tree
treeFit <- tree(Condition ~ ., data = carTrain)
plot(treeFit)

predTrain <- predict(treeFit, carTrain, type = "class")  # prediction on train set
mean(predTrain == carTrain$Condition)                    # classification accuracy
predValid <- predict(treeFit, carValid, type = "class")  # prediction on validation set
mean(predValid == carValid$Condition)                    # classification accuracy


# Build a "large" Classification Tree
ltreeFit <- tree(Condition ~ ., data = carTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(carTrain),  # number of sample points
                                        mincut = 1,             # minimum points in each child
                                        minsize = 2,            # minimum points in each parent
                                        mindev = 0))            # minimum information gain to split
plot(ltreeFit)

predTrain <- predict(ltreeFit, carTrain, type = "class")  # prediction on train set
mean(predTrain == carTrain$Condition)                     # classification accuracy
predValid <- predict(ltreeFit, carValid, type = "class")  # prediction on validation set
mean(predValid == carValid$Condition)                     # classification accuracy


# Build a "small" Classification Tree
streeFit <- tree(Condition ~ ., data = carTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(carTrain),
                                        mincut = 0.1 * nrow(carTrain),
                                        minsize = 0.2 * nrow(carTrain),
                                        mindev = 0.1))
plot(streeFit)

predTrain <- predict(streeFit, carTrain, type = "class")  # prediction on train set
mean(predTrain == carTrain$Condition)                     # classification accuracy
predValid <- predict(streeFit, carValid, type = "class")  # prediction on validation set
mean(predValid == carValid$Condition)                     # classification accuracy


# Build a "pruned" Classification Tree over train set
ltreeFit <- tree(Condition ~ ., data = carTrain, 
                 split = "deviance",
                 method = "recursive.partition",
                 control = tree.control(nobs = nrow(carTrain),
                                        mincut = 1,
                                        minsize = 2,
                                        mindev = 0))
plot(ltreeFit)

cvTree <- cv.tree(ltreeFit, FUN = prune.misclass, K = 10) # K-fold Cross-Validation
cbind(cvTree$size, cvTree$dev, cvTree$k)                  # check cvTree output
plot(cvTree$size, cvTree$dev, type="b")                   # plot deviance vs size
plot(cvTree$k, cvTree$dev, type="b")                      # plot deviance vs alpha

bestSize <- 5  # choose this parameter carefully, based on the cvTree output
ptreeFit <- prune.misclass(ltreeFit, best = bestSize)     # prune tree to best size
plot(ptreeFit)
text(ptreeFit, pretty = FALSE)

predTrain <- predict(ptreeFit, carTrain, type = "class")  # prediction on train set
mean(predTrain == carTrain$Condition)                     # classification accuracy
predValid <- predict(ptreeFit, carValid, type = "class")  # prediction on validation set
mean(predValid == carValid$Condition)                     # classification accuracy



# =======================================================
# Tree Model : Part 3 : Bagging and Random Forest
# =======================================================

# install.packages("randomForest")
library(randomForest)

# Load the dataset and explore
carData <- read.csv("CarEvaluation.csv", header = TRUE)
str(carData)
summary(carData)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(carData), 0.7*nrow(carData), replace = FALSE)
carTrain <- carData[train,]
carValid <- carData[-train,]
summary(carTrain)
summary(carValid)


# -------------------------------------------------------
# Build a Bagging Model on train set

# Each node splits with all features
bagFit <- randomForest(Condition ~ .,                    # formula
                       data = carTrain,                  # data set
                       ntree = 5000,                      # number of trees
                       mtry = 6,                         # variables for split
                       importance = TRUE)                # importance recorded
bagFit

predTrain <- predict(bagFit, carTrain, type = "class")   # prediction on train set
mean(predTrain == carTrain$Condition)                    # classification accuracy
predValid <- predict(bagFit, carValid, type = "class")   # prediction on validation set
mean(predValid == carValid$Condition)                    # classification accuracy

importance(bagFit)        # importance of the variables in the model (values)
varImpPlot(bagFit)        # importance of the variables in the model (visual)


# -------------------------------------------------------
# Build a Random Forest Model on train set

# Each node splits with subset of features
rfFit <- randomForest(Condition ~ .,                     # formula
                      data = carTrain,                   # data set
                      ntree = 500,                       # number of trees
                      mtry = 3,                          # variables for split
                      importance = TRUE)                 # importance recorded                 
rfFit

predTrain <- predict(rfFit, carTrain, type = "class")    # prediction on train set
mean(predTrain == carTrain$Condition)                    # classification accuracy
predValid <- predict(rfFit, carValid, type = "class")    # prediction on validation set
mean(predValid == carValid$Condition)                    # classification accuracy

importance(rfFit)         # importance of the variables in the model (values)
varImpPlot(rfFit)         # importance of the variables in the model (visual)


# -------------------------------------------------------
# Boruta : Variable Importance at a Glance

# install.packages("Boruta")
library("Boruta")
borutaFit <- Boruta(Condition ~ .,            # formula
                    data = carTrain)          # data set

getSelectedAttributes(borutaFit)              # extract important variables
attStats(borutaFit)                           # full finalBoruta statistics
borutaFit                                     # importance of variables (values)
plot(borutaFit)                               # importance of variables (visual)
