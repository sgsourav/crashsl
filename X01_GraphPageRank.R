# =======================================================
# PageRank computation on a Graph
# =======================================================

#install.packages("igraph")
library(igraph)

# Adjacency matrix for the example graph
adjMat <- rbind(
  c(0,1,1,0,0,0,0),
  c(1,0,1,1,0,0,0),
  c(1,1,0,0,0,0,0),
  c(0,1,0,0,1,1,1),
  c(0,0,0,1,0,1,0),
  c(0,0,0,1,1,0,1),
  c(0,0,0,1,0,1,0))

# Creating a graph object from the adjacency matrix
g  <- graph.adjacency(adjMat, mode = "undirected")
plot(g, layout = layout.fruchterman.reingold,
     vertex.size = 25,
     vertex.color="red",
     vertex.frame.color= "white",
     vertex.label.color = "white",
     vertex.label.family = "sans",
     edge.width=2,  
     edge.color="black")


# -------------------------------------------------------
# Compute PageRank on the graph

# Compute the Transition Matrix for Random Walk
transMat <- t(adjMat / rowSums(adjMat))
transMat

# Compute the principal eigenvector of transMat
transEig <- eigen(transMat)
transEig
pageRank <- transEig$vec[,1]
pageRank

# Normalized PageRank
npageRank <- pageRank / sum(pageRank)
npageRank

# Represent PageRank as size of vertices
V(g)$size <- 25 + 100 * round(npageRank[V(g)], 2)
plot(g, layout = layout.fruchterman.reingold,
     vertex.size = V(g)$size,
     vertex.color="red",
     vertex.frame.color= "white",
     vertex.label.color = "white",
     vertex.label.family = "sans",
     edge.width=2,  
     edge.color="black")


# -------------------------------------------------------
# Shortcut : Use the stock pagerank function in R
npageRank <- page.rank(g)
npageRank
V(g)$size <- 25 + 100 * round(npageRank$vector[V(g)], 2)
plot(g, layout = layout.fruchterman.reingold,
     vertex.size = V(g)$size,
     vertex.color= "red",
     vertex.frame.color= "white",
     vertex.label.color = "white",
     vertex.label.family = "sans",
     edge.width=2,  
     edge.color="black")


# -------------------------------------------------------
# Example graph: Zachary's Karate Club

g <- make_graph("Zachary")
plot(g, layout = layout.fruchterman.reingold,
     vertex.size = 10,
     vertex.label = NA,
     vertex.color="red",
     vertex.frame.color= "white",
     vertex.label.color = "white",
     vertex.label.family = "sans",
     edge.width=2,  
     edge.color="black")

npageRank <- page.rank(g)
npageRank

V(g)$size <- 10 + 100 * round(npageRank$vector[V(g)], 2)
plot(g, layout = layout.fruchterman.reingold,
     vertex.size = V(g)$size,
     vertex.label = NA,
     vertex.color="red",
     vertex.frame.color= "white",
     vertex.label.color = "white",
     vertex.label.family = "sans",
     edge.width=2,  
     edge.color="black")
