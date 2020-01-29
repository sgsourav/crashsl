# =======================================================
# Data Scraping in R
# =======================================================

# install.packages("rvest")
library(rvest)


# -------------------------------------------------------
# Example 1 : Scrape Table from Election Results India

# Access ElectionResults Website
upResultsURL <- "http://eciresults.nic.in/PartyWiseResultS24.htm?st=S24"
upResultsPage <- read_html(upResultsURL)

# Fetch the table with results
upResults <- upResultsPage %>%
  html_nodes("#div1 > table:nth-child(2)") %>%
  html_table() %>% .[[1]]

# Sanitize the table for clarity
names(upResults) <- upResults[3, ]
upResults <- upResults[-c(1, 2, 3, 13, 14, 15), ]

# Print the sanitized data
upResults


# -------------------------------------------------------
# Example 2 : Scrape Table from Wikipedia

# Download the webpage as an HTML file
download.file("https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(PPP)",
              destfile = "wikiPage.html")

# Access the locally stored HTML file
wikiPage <- read_html("wikiPage.html")

# Extract and print the desired table
wikiPage %>% html_nodes(".wikitable") %>% .[[1]] %>% html_table()
