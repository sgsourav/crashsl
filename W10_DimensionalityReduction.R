# =======================================================
# Dimensionality Reduction : Part 1 : Visual Example
# =======================================================

# source("https://bioconductor.org/biocLite.R")
# biocLite()
biocLite("EBImage")

# Read and display the original image
origImage <- EBImage::readImage("joker.jpg","JPG")
dim(origImage)
EBImage::display(origImage)

# Compute the SVD/PCA of original image
origImage.svd <- svd(origImage)
recImage <- origImage.svd$u %*% diag(origImage.svd$d) %*% t(origImage.svd$v)
EBImage::display(recImage)

# Compress down to Principal Components
pc1Image <- origImage.svd$u[,1:2] %*% diag(origImage.svd$d[1:2]) %*% t(origImage.svd$v[,1:2])
EBImage::display(pc1Image)
pc10Image <- origImage.svd$u[,1:10] %*% diag(origImage.svd$d[1:10]) %*% t(origImage.svd$v[,1:10])
EBImage::display(pc10Image)
pc20Image <- origImage.svd$u[,1:20] %*% diag(origImage.svd$d[1:20]) %*% t(origImage.svd$v[,1:20])
EBImage::display(pc20Image)
pc100Image <- origImage.svd$u[,1:90] %*% diag(origImage.svd$d[1:90]) %*% t(origImage.svd$v[,1:90])
EBImage::display(pc100Image)

for (i in seq(10, 120, 10)) { 
  pciImage <- origImage.svd$u[,1:i] %*% diag(origImage.svd$d[1:i]) %*% t(origImage.svd$v[,1:i])
  EBImage::display(pciImage)
}

# Check the variance explained
plot(origImage.svd$d, type = 'h')
plot(origImage.svd$d[1:120], type = 'h')

# Choose an optimal threshold
k = 60
t(origImage.svd$d) %*% origImage.svd$d
t(origImage.svd$d[1:k]) %*% origImage.svd$d[1:k]
t(origImage.svd$d[1:k]) %*% origImage.svd$d[1:k] / t(origImage.svd$d) %*% origImage.svd$d

pckImage <- origImage.svd$u[,1:k] %*% diag(origImage.svd$d[1:k]) %*% t(origImage.svd$v[,1:k])
EBImage::display(pckImage)



# =======================================================
# Dimensionality Reduction : Part 2 : PCA
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Convert and visualize the dataset
irisData <- as.data.frame(iris)
pairs(irisData[,1:4], pch = 19, col = as.numeric(irisData$Species) + 1)

# -------------------------------------------------------
# Perform Principal Component Analysis

irisPCA <- prcomp(irisData[,1:4], center = TRUE, scale. = TRUE)
irisPCA
summary(irisPCA)

# Projection on Comp.1 and Comp.2
# install.packages("ggfortify")
library(ggfortify)
autoplot(irisPCA, col = as.numeric(irisData$Species) + 1,
         loadings = TRUE, loadings.label = TRUE)



# =======================================================
# Dimensionality Reduction : Part 3 : t-SNE
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Convert and visualize the dataset
irisData <- as.data.frame(iris)
pairs(irisData[,1:4], pch = 19, col = as.numeric(irisData$Species) + 1)

# -------------------------------------------------------
# Perform t-SNE analysis

# install.packages("Rtsne")
library(Rtsne)
irisUnique <- unique(irisData)
irisTSNE <- Rtsne(irisUnique[,1:4])

# Projection on Comp.1 and Comp.2
plot(irisTSNE$Y, pch = 19, col = as.numeric(irisUnique$Species) + 1,
     xlab = "t-SNE Comp.1", ylab = "t-SNE Comp.2")
