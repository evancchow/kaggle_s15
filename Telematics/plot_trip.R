# Quick little script to plot a trip.

# imports
lapply(c("ggplot2", "data.table"), require, character.only=T)

filepath <- './2.csv'
trip.data <- as.data.frame(fread(filepath))
# Different color in journey: line gets darker the later in the trip
trip.data$path <- 1:nrow(trip.data)
p <- ggplot(trip.data, aes(x=x, y=y, color=path)) + geom_path()
show(p)