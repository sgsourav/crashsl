# =======================================================
# SVM : Part 3 : Case Studies for Models and Tuning
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



# =======================================================
# SVM : Part 3A : Multi-Class Classification (tuned)
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Truncate the data to a "2-dimensional multi-class" problem
irisData <- as.data.frame(iris[,c(1,3,5)])
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


# Fit an optimally tuned Support Vector Machine on the train set
# by performing cross-validation with a range of parameter values

# -------------------------------------------------------
# Linear Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "linear", # (u.v)
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)),
                  scale = FALSE)

linFit <- tuneModel$best.model       # extract the best tuned model
summary(linFit)                      # best parameter-tuned model
prettyPlot(linFit, irisTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(linFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Polynomial Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "polynomial", # ( gamma * (u.v) + coef0 ) ^ degree
                  ranges = list(gamma = c(0.01, 0.1, 0.5),
                                coef0 = c(0.01, 0.1, 1),
                                degree = c(1, 2, 3),
                                cost = c(0.001, 0.01, 0.1, 1, 10)), 
                  scale = FALSE)

polFit <- tuneModel$best.model       # extract the best tuned model
summary(polFit)                      # best parameter-tuned model
prettyPlot(polFit, irisTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(polFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Radial (Gaussian) Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "radial", # exp( - gamma * (u - v) . (u - v) )
                  ranges = list(gamma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

radFit <- tuneModel$best.model       # extract the best tuned model
summary(radFit)                      # best parameter-tuned model
prettyPlot(radFit, irisTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(radFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


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

sigFit <- tuneModel$best.model       # extract the best tuned model
summary(sigFit)                      # best parameter-tuned model
prettyPlot(sigFit, irisTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(sigFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix



# =======================================================
# SVM : Part 3B : Mixture Model of Gaussians (tuned)
# =======================================================

# Create an artificial dataset
x <- matrix(rnorm(200*2), ncol=2)             # 200 two-dimensional data points
x[1:100,] <- x[1:100,] + 2                    # rearrange the 2-dim data points
x[101:150,] <- x[101:150,] - 2                # to make the problem interesting
y <- c(rep(1,150), rep(2,50))                 # set random class labels on data

artiData <- data.frame(x=x,y=as.factor(y))    # the created artificial data set
names(artiData) <- c("Variable.X1", 
                     "Variable.X2", 
                     "Class.Label")
plot(artiData[,1:2], pch = 19, col = as.numeric(artiData$Class.Label) + 1)

# Split into Train and Validation sets
# Training Set : Validation Set = 1 : 1 (random)
train <- sample(nrow(artiData), 0.5*nrow(artiData), replace = FALSE)
artiTrain <- artiData[train,]
artiValid <- artiData[-train,]
plot(artiTrain[,1:2], col = as.numeric(artiTrain$Class.Label) + 1, pch = 19)
points(artiValid[,1:2], col = as.numeric(artiValid$Class.Label) + 1, pch = 21)


# Fit an optimally tuned Support Vector Machine on the train set
# by performing cross-validation with a range of parameter values

# -------------------------------------------------------
# Linear Kernel

tuneModel <- tune(svm, 
                  Class.Label ~ ., 
                  data = artiTrain, 
                  kernel = "linear", # (u.v)
                  ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),
                  scale = FALSE)

linFit <- tuneModel$best.model       # extract the best tuned model
summary(linFit)                      # best parameter-tuned model
prettyPlot(linFit, artiTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(linFit, newdata = artiValid)          # prediction
table(predict = predValid, truth = artiValid$Class.Label)  # confusion matrix


# -------------------------------------------------------
# Polynomial Kernel

tuneModel <- tune(svm, 
                  Class.Label ~ ., 
                  data = artiTrain, 
                  kernel = "polynomial", # ( gamma * (u.v) + coef0 ) ^ degree
                  ranges = list(gamma = c(0.01, 0.1, 0.5),
                                coef0 = c(0.01, 0.1, 1),
                                degree = c(1, 2, 3),
                                cost = c(0.001, 0.01, 0.1, 1, 10)), 
                  scale = FALSE)

polFit <- tuneModel$best.model       # extract the best tuned model
summary(polFit)                      # best parameter-tuned model
prettyPlot(polFit, artiTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(polFit, newdata = artiValid)          # prediction
table(predict = predValid, truth = artiValid$Class.Label)  # confusion matrix


# -------------------------------------------------------
# Radial (Gaussian) Kernel

tuneModel <- tune(svm, 
                  Class.Label ~ ., 
                  data = artiTrain, 
                  kernel = "radial", # exp( - gamma * (u - v) . (u - v) )
                  ranges = list(gamma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

radFit <- tuneModel$best.model       # extract the best tuned model
summary(radFit)                      # best parameter-tuned model
prettyPlot(radFit, artiTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(radFit, newdata = artiValid)          # prediction
table(predict = predValid, truth = artiValid$Class.Label)  # confusion matrix


# -------------------------------------------------------
# Sigmoid Kernel

tuneModel <- tune(svm, 
                  Class.Label ~ ., 
                  data = artiTrain, 
                  kernel = "sigmoid", # tanh( gamma * (u.v) + coef0 )
                  ranges = list(gamma = c(0.01, 0.1, 1, 10, 100),
                                coef0 = c(0.01, 0.1, 1, 10, 100),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

sigFit <- tuneModel$best.model       # extract the best tuned model
summary(sigFit)                      # best parameter-tuned model
prettyPlot(sigFit, artiTrain, 100)   # our prettyPlot for SVM

# Predict the classes for the validation set
predValid <- predict(sigFit, newdata = artiValid)          # prediction
table(predict = predValid, truth = artiValid$Class.Label)  # confusion matrix



# =======================================================
# SVM : Part 3C : Multi-Feature Multi-Class (tuned)
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Keep the data as a "multi-dimensional multi-class" problem
irisData <- as.data.frame(iris)

# Split into Train and Validation sets
# Training Set : Validation Set = 70 : 30 (random)
train <- sample(nrow(irisData), 0.7*nrow(irisData), replace = FALSE)
irisTrain <- irisData[train,]
irisValid <- irisData[-train,]


# Fit an optimally tuned Support Vector Machine on the train set
# by performing cross-validation with a range of parameter values

# -------------------------------------------------------
# Linear Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "linear", # (u.v)
                  ranges = list(cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)),
                  scale = FALSE)

linFit <- tuneModel$best.model                           # best tuned model
summary(linFit)                                          # best model parameters
predValid <- predict(linFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Polynomial Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "polynomial", # ( gamma * (u.v) + coef0 ) ^ degree
                  ranges = list(gamma = c(0.01, 0.1, 0.5),
                                coef0 = c(0.01, 0.1, 1),
                                degree = c(1, 2, 3),
                                cost = c(0.001, 0.01, 0.1, 1, 10)), 
                  scale = FALSE)

polFit <- tuneModel$best.model                           # best tuned model
summary(polFit)                                          # best model parameters
predValid <- predict(polFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


# -------------------------------------------------------
# Radial (Gaussian) Kernel

tuneModel <- tune(svm, 
                  Species ~ ., 
                  data = irisTrain, 
                  kernel = "radial", # exp( - gamma * (u - v) . (u - v) )
                  ranges = list(gamma = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
                                cost = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)), 
                  scale = FALSE)

radFit <- tuneModel$best.model                           # best tuned model
summary(radFit)                                          # best model parameters
predValid <- predict(radFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix


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

sigFit <- tuneModel$best.model                           # best tuned model
summary(sigFit)                                          # best model parameters
predValid <- predict(sigFit, newdata = irisValid)        # prediction
table(predict = predValid, truth = irisValid$Species)    # confusion matrix
