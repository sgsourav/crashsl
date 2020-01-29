# =======================================================
# SVM : Part 1 : Support Vector Classifier
# =======================================================

# install.packages("e1071")
library(e1071)

# Pretty plotting routine for SVM models
prettyPlot <- function(svmModel, dataSet, gridSize) {
  # create the grid of points to plot
  x1.Pixels <- seq(from = as.numeric(min(dataSet[,1])), 
                   to = as.numeric(max(dataSet[,1])), len = gridSize)
  x2.Pixels <- seq(from = as.numeric(min(dataSet[,2])), 
                   to = as.numeric(max(dataSet[,2])), len = gridSize)
  pixelGrid <- expand.grid(X1 = x1.Pixels, X2 = x2.Pixels)
  names(pixelGrid) <- names(dataSet)[1:2]
  
  # predict the classes of the pixels
  pixelData <- data.frame(pixelGrid)
  classGrid <- predict(svmModel, newdata = pixelData)
  
  # plot pixelGrid with colored class
  plot(pixelGrid, pch = 20, cex = 0.2, 
       col = as.numeric(classGrid) + 1)
  
  # plot the data points on top of this
  points(dataSet[,1:2], pch = 19, cex = 1, col = as.numeric(dataSet[,3]) + 1)
  
  # mark the actual support vectors now
  points(dataSet[svmModel$index, 1:2], pch = 9, cex = 1.5)
}


# -------------------------------------------------------
# Load and explore the dataset

attach(iris)
help(iris)
str(iris)

# Truncate the data to a "2-dimensional 2-class" problem
irisData <- as.data.frame(iris[1:100,c(1,3,5)])
irisData$Species <- droplevels(irisData$Species)
str(irisData)
plot(irisData$Sepal.Length, irisData$Petal.Length,
     col = as.numeric(irisData$Species) + 1, pch = 19,
     xlab = "Sepal.Length", ylab = "Petal.Length")

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(irisData), 0.7*nrow(irisData), replace = FALSE)
irisTrain <- irisData[train,]
irisValid <- irisData[-train,]

plot(irisTrain$Sepal.Length, irisTrain$Petal.Length,
     col = as.numeric(irisTrain$Species) + 1, pch = 19,
     xlab = "Sepal.Length", ylab = "Petal.Length")
points(irisValid$Sepal.Length, irisValid$Petal.Length,
       col = as.numeric(irisValid$Species) + 1, pch = 21)



# =======================================================
# SVM : Part 1A : Maximal Margin Classifier
# =======================================================

# Fit a Maximal Margin Classifier on the train set
# Set high cost to minimize number of support vectors
svmFit <- svm(Species ~ .,           # formula for fit
              data = irisTrain,      # dataset for fit
              kernel = "linear",     # choose a kernel
              cost = 1e6,            # relaxation cost
              scale = FALSE)         # feature-scaling

summary(svmFit)                      # summary of the fitted model
# plot(svmFit, irisTrain)            # default R plot for SVM
prettyPlot(svmFit, irisTrain, 100)   # our prettyPlot for SVM

svmFit$index                         # index of the support vectors
svmFit$SV                            # the actual support vectors

# Predict the classes for the validation set
predValid <- predict(svmFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix



# =======================================================
# SVM : Part 1B : Support Vector Classifier
# =======================================================

# Fit a Support Vector Classifier on the train set
# Set variable cost to vary number of support vectors
svmFit <- svm(Species ~ .,           # formula for fit
              data = irisTrain,      # dataset for fit
              kernel = "linear",     # choose a kernel
              cost = 0,            # relaxation cost
              scale = FALSE)         # feature-scaling

summary(svmFit)                      # summary of the fitted model
# plot(svmFit, irisTrain)            # default R plot for SVM
prettyPlot(svmFit, irisTrain, 100)   # our prettyPlot for SVM

svmFit$index                         # index of the support vectors
svmFit$SV                            # the actual support vectors

# Predict the classes for the validation set
predValid <- predict(svmFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Fit an optimally tuned Support Vector Classifier
# by performing cross-validation with a cost range

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "linear",
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

summary(tuneModel)                   # summary of tuning the cost
bestFit <- tuneModel$best.model      # extract the best tuned model
summary(bestFit)                     # best model after tuning cost
# plot(bestFit, irisTrain)           # default R plot for SVM
prettyPlot(bestFit, irisTrain, 100)  # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(bestFit, newdata = irisValid)       # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix



# =======================================================
# SVM : Part 2 : Support Vector Machine
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Truncate the data to a "2-dimensional 2-class" problem
irisData <- as.data.frame(iris[1:100,c(1,3,5)])
irisData$Species <- droplevels(irisData$Species)
str(irisData)
plot(irisData$Sepal.Length, irisData$Petal.Length,
     col = as.numeric(irisData$Species) + 1, pch = 19,
     xlab = "Sepal.Length", ylab = "Petal.Length")

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(irisData), 0.7*nrow(irisData), replace = FALSE)
irisTrain <- irisData[train,]
irisValid <- irisData[-train,]

plot(irisTrain$Sepal.Length, irisTrain$Petal.Length,
     col = as.numeric(irisTrain$Species) + 1, pch = 19,
     xlab = "Sepal.Length", ylab = "Petal.Length")
points(irisValid$Sepal.Length, irisValid$Petal.Length,
       col = as.numeric(irisValid$Species) + 1, pch = 21)


# Fit a Support Vector machine to the train set using various kernels
# where the choice of the kernel depends on the nature of the dataset

# -------------------------------------------------------
# Polynomial Kernel

polFit <- svm(Species ~ .,           # formula for fit
              data = irisTrain,      # dataset for fit
              kernel = "polynomial", # ( gamma * (u.v) + coef0 ) ^ degree
              gamma = 1,             # gamma is required
              coef0 = 1,             # coef0 is required (default = 0)
              degree = 5,            # degree is required
              cost = 10,             # relaxation cost
              scale = FALSE)         # feature-scaling

summary(polFit)                      # summary of the fitted model
# plot(polFit, irisTrain)            # default R plot for SVM
prettyPlot(polFit, irisTrain, 100)   # our prettyPlot for SVM

polFit$index                         # index of the support vectors
polFit$SV                            # the actual support vectors

# Predict the classes for the validation set
predValid <- predict(polFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Radial (Gaussian) Kernel

radFit <- svm(Species ~ .,           # formula for fit
              data = irisTrain,      # dataset for fit
              kernel = "radial",     # exp( - gamma * (u - v) . (u - v) )
              gamma = 1,             # gamma is required
              cost = 10,             # relaxation cost
              scale = FALSE)         # feature-scaling

summary(radFit)                      # summary of the fitted model
# plot(radFit, irisTrain)            # default R plot for SVM
prettyPlot(radFit, irisTrain, 100)   # our prettyPlot for SVM

radFit$index                         # index of the support vectors
radFit$SV                            # the actual support vectors

# Predict the classes for the validation set
predValid <- predict(radFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Sigmoid Kernel

sigFit <- svm(Species ~ .,           # formula for fit
              data = irisTrain,      # dataset for fit
              kernel = "sigmoid",    # tanh( gamma * (u.v) + coef0 )
              gamma = 0.01,          # gamma is required
              coef0 = 1,             # coef0 is required (default = 0)
              cost = 10,             # relaxation cost
              scale = FALSE)         # feature-scaling

summary(sigFit)                      # summary of the fitted model
# plot(sigFit, irisTrain)            # default R plot for SVM
prettyPlot(sigFit, irisTrain, 100)   # our prettyPlot for SVM

sigFit$index                         # index of the support vectors
sigFit$SV                            # the actual support vectors

# Predict the classes for the validation set
predValid <- predict(radFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix



# =======================================================
# SVM : Part 2B : Tuning the various SVM Kernels
# =======================================================

# Fit an optimally tuned Support Vector Machine on the train set
# by performing cross-validation with a range of parameter values


# -------------------------------------------------------
# Polynomial Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "polynomial", # ( gamma * (u.v) + coef0 ) ^ degree
                  ranges = list(gamma = c(0.01, 0.1, 1, 10, 100),
                                coef0 = c(0.01, 0.1, 1, 10, 100),
                                degree = c(1, 2, 3, 4, 5, 6 ,7),
                                cost = c(0.01, 0.1, 1, 10, 100)), 
                  scale = FALSE)

summary(tuneModel)                   # summary of tuning parameters
bestFit <- tuneModel$best.model      # extract the best tuned model
summary(bestFit)                     # best parameter-tuned model
# plot(bestFit, irisTrain)           # default R plot for SVM
prettyPlot(bestFit, irisTrain, 100)  # our prettyPlot for SVM


# -------------------------------------------------------
# Radial (Gaussian) Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "radial", # exp( - gamma * (u - v) . (u - v) )
                  ranges = list(gamma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

summary(tuneModel)                   # summary of tuning parameters
bestFit <- tuneModel$best.model      # extract the best tuned model
summary(bestFit)                     # best parameter-tuned model
# plot(bestFit, irisTrain)           # default R plot for SVM
prettyPlot(bestFit, irisTrain, 100)  # our prettyPlot for SVM


# -------------------------------------------------------
# Sigmoid Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "sigmoid", # tanh( gamma * (u.v) + coef0 )
                  ranges = list(gamma = c(0.01, 0.1, 1, 10, 100),
                                coef0 = c(0.01, 0.1, 1, 10, 100),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

summary(tuneModel)                   # summary of tuning parameters
bestFit <- tuneModel$best.model      # extract the best tuned model
summary(bestFit)                     # best parameter-tuned model
# plot(bestFit, irisTrain)           # default R plot for SVM
prettyPlot(bestFit, irisTrain, 100)  # our prettyPlot for SVM
