# =======================================================
# Clustering Methods : K-Means and Hierarchical
# =======================================================

# Load and explore the dataset
attach(iris)
help(iris)
str(iris)

# Subset the features to get 2-D dataset
irisData <- as.data.frame(iris[,c(1,3)])
plot(irisData, pch = 19)



# =======================================================
# Clustering Methods : Part 1 : K-Means Clustering
# =======================================================

# Set an interesting color palette for visualization
# install.packages("RColorBrewer")
library(RColorBrewer)
# display.brewer.all()
myPal <- brewer.pal(n = 9, name = "Set1")

# K-Means Clustering
# Set the value of K
K <- 6
kMeansFit <- kmeans(irisData, centers = K)
kMeansFit
plot(irisData, pch = 19, col = palette(myPal)[as.numeric(kMeansFit$cluster)])

# K-Means Clustering with multiple random starts
# Set the value of K
K <- 4
kMeansFit <- kmeans(irisData, centers = K, nstart = 20)
kMeansFit
plot(irisData, pch = 19, col = palette(myPal)[as.numeric(kMeansFit$cluster)])



# =======================================================
# Clustering Methods : Part 2 : Hierarchical Clustering
# =======================================================

# Hierarchical Clustering : Complete Linkage
hiercFit <- hclust(dist(irisData), method="complete")
hiercFit
plot(hiercFit, main="Complete Linkage", xlab="", ylab="", sub="", cex =.5)

K <- 3
plot(hiercFit, main="Complete Linkage", xlab="", ylab="", sub="", cex =.5)
rect.hclust(hiercFit, k = K)
plot(irisData, pch = 19, col = palette(myPal)[as.numeric(cutree(hiercFit, k = K))])

# Hierarchical clustering : Average Linkage
hiercFit <- hclust(dist(irisData), method="average")
hiercFit
plot(hiercFit, main="Average Linkage", xlab="", ylab="", sub="", cex =.5)

K <- 3
plot(hiercFit, main="Average Linkage", xlab="", ylab="", sub="", cex =.5)
rect.hclust(hiercFit, k = K)
plot(irisData, pch = 19, col = palette(myPal)[as.numeric(cutree(hiercFit, k = K))])

# Hierarchical clustering : Single Linkage
hiercFit <- hclust(dist(irisData), method="single")
hiercFit
plot(hiercFit, main="Single Linkage", xlab="", ylab="", sub="", cex =.5)

K <- 3
plot(hiercFit, main="Single Linkage", xlab="", ylab="", sub="", cex =.5)
rect.hclust(hiercFit, k = K)
plot(irisData, pch = 19, col = palette(myPal)[as.numeric(cutree(hiercFit, k = K))])



# =======================================================
# Clustering Methods : Part 3 : Graph Clustering
# =======================================================

# install.packages("igraph")
library(igraph)

# Example Graph : Zachary's Karate Club
graphZKC <- make_graph("Zachary")
plot(graphZKC, 
     layout = layout.kamada.kawai,
     vertex.size = 7,
     vertex.color = "darkgray",
     edge.width = 1,  
     edge.color = "darkgray",
     vertex.label = NA)

# Fast Greedy Clustering on the Graph
fgcFit <- cluster_fast_greedy(graphZKC)
fgcFit
membership(fgcFit)
plot(graphZKC,
     layout = layout.kamada.kawai,
     vertex.size = 7,
     vertex.color = palette(myPal)[as.numeric(membership(fgcFit))],
     edge.width = 1,  
     edge.color = "darkgray",
     vertex.label = NA)
