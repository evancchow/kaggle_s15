# Script to compare time series with dynamic time warping.

lapply(c("data.table", "ggplot2", "dtw", "gridExtra"), 
       require, character.only=T)

# make the time series for trip 2
ts.accel <- as.data.frame(fread("2_csv_accelerationSeries.csv"))
ts.ctr <- as.data.frame(fread("2_csv_centroidDistSeries.csv"))
second.ts.accel.x <- ts(ts.accel$V1, start=0)
second.ts.accel.y <- ts(ts.accel$V2, start=0)
second.ts.ctr.x <- ts(ts.ctr$V1, start=0)
second.ts.ctr.y <- ts(ts.ctr$V2, start=0)

# make the time series for trip 3
ts.accel <- as.data.frame(fread("3_csv_accelerationSeries.csv"))
ts.ctr <- as.data.frame(fread("3_csv_centroidDistSeries.csv"))
third.ts.accel.x <- ts(ts.accel$V1, start=0)
third.ts.accel.y <- ts(ts.accel$V2, start=0)
third.ts.ctr.x <- ts(ts.ctr$V1, start=0)
third.ts.ctr.y <- ts(ts.ctr$V2, start=0)

# make the time series for trip 4
ts.accel <- as.data.frame(fread("4_csv_accelerationSeries.csv"))
ts.ctr <- as.data.frame(fread("4_csv_centroidDistSeries.csv"))
fourth.ts.accel.x <- ts(ts.accel$V1, start=0)
fourth.ts.accel.y <- ts(ts.accel$V2, start=0)
fourth.ts.ctr.x <- ts(ts.ctr$V1, start=0)
fourth.ts.ctr.y <- ts(ts.ctr$V2, start=0)

# Run dynamic time warping. Gives a numeric measure of how "far" (or
# different) two time series are from each other.
# alignment <- dtw(second.ts.accel.x)
grid.arrange(qplot(1:length(second.ts.accel.x), second.ts.accel.x) +
               ggtitle("Second trip"),
             qplot(1:length(third.ts.accel.x), third.ts.accel.x) +
               ggtitle("Third trip"),
             qplot(1:length(fourth.ts.accel.x), fourth.ts.accel.x) +
               ggtitle("Fourth trip"))

# Get distances from dynamic time warping between different combinations
# of time series.
# Normalized does NOT mean between 0 and 1 necessarily. See first paragraph
# of page 8 of the handout on the dtw package:
# http://www.jstatsoft.org/v31/i07/paper
second.third.accel.align <- dtw(second.ts.accel.x,
                                third.ts.accel.x)
print(sprintf("Distance between second, third trips: %.8f",
              second.third.accel.align$normalizedDistance))

second.fourth.accel.align <- dtw(second.ts.accel.x,
                                 fourth.ts.accel.x)
print(sprintf("Distance between second, fourth trips: %.8f",
              second.fourth.accel.align$normalizedDistance))

third.fourth.accel.align <- dtw(third.ts.accel.x,
                                fourth.ts.accel.x)
print(sprintf("Distance beetween third, fourth trips: %.8f",
              third.fourth.accel.align$normalizedDistance))












