setwd('D:/OneDrive - University of Glasgow/UofG/Traffic flow prediction_Part2/New_flow')

#5.4.4 Train an SVR model
# Read csv file 
flow_data <- read.csv("GD1351_R_new.csv")
View(flow_data)


#select data from 2019-11-21
start_row <- which(flow_data$timeStamp == '2019-11-21 00:00:00') 
end_row <- which(flow_data$timeStamp == '2019-12-28 23:45:00')
new_flow <- flow_data[start_row:end_row,]
row.names(new_flow) <- NULL  #reset the index
View(new_flow)

# embed the data
m <- 3
data <- embed(x = new_flow[,"newFlow"], dimension = m+1)
colnames(data) <- c("y_t-3","y_t-2","y_t-1","y_t")

#In non-temporal data the training and testing sets are usually divided randomly. 
#However, with temporal data it is important to consider the ordering of time.
#In this case, we will use the first 80% of the months to train the model, 
#and the remaining 20% of months for testing.
# divide data into training and testing sets

n <- nrow(data)
trainDays <- ceiling(n*0.8/96)
split <- 96*trainDays

yTrain <- data[1:split,1]
XTrain <- data[1:split,-1]
yTest <- data[(split+1):nrow(data),1]
XTest <- data[(split+1):nrow(data),-1]

View(XTrain)

#use k-fold cross validation, with k set to 5.
library(sp)
#install.packages('caret')
#install.packages('rlang')
library(ggplot2)
library(caret)
ctrl <- trainControl(method = "cv", number=5) 

#Create a grid of parameters to test and train the model:
SVRGridCoarse <- expand.grid(.sigma=c(0.001, 0.01, 0.1), .C=c(10,100,1000))
SVRFitCoarse <- train(XTrain, yTrain, method="svmRadial", tuneGrid=SVRGridCoarse
                      , trControl=ctrl, type="eps-svr")
SVRFitCoarse
# root mean squared error (RMSE), R squared and Mean Absolute Error (MAE).

plot(SVRFitCoarse)

# refine the grid to see if we can gain further improvements in performance:
SVRGridFine <- expand.grid(.sigma=c(0.05, 0.1, 0.15), .C=c(50,100,150))
system.time(SVRFitFine <- train(XTrain, yTrain, method="svmRadial", tuneGrid=SVRGridFine, 
                                trControl=ctrl, type="eps-svr"))

plot(SVRFitFine)


#Examine the properties of the fitted model:
names(SVRFitFine)
#The model object for the best model can be accessed by
SVRFitFine$finalModel

SVRFitFine

#use the model for one-step-ahead prediction and plot the results
yPred <- predict(SVRFitFine, XTest)
plot(yTest, type="l", xaxt="n", xlab="Date", ylab="UKTemp")
lines(yPred, col="red", lwd=2)
points(yPred, col="blue", pch=21, bg="blue")
#axis(1, at=seq(90, (180*5)+90, 180), labels=unique(dates[(split+1):nrow(data),1]))
axis(1, at=seq(6, (12*19)+6, 12), labels=format(dates[(trainYears+1):length(dates)], "%Y"))
legend(90, 3, legend=c("Observed"), lty=1, bty="n")
legend(150, 3, legend=c("Predicted"), pch=21, col="blue", bg="blue", bty="n")


#check whether any temporal autocorrelation remains in the residuals of the model:
SVRResidual <- yTest-yPred
plot(SVRResidual)

acf(SVRResidual)


