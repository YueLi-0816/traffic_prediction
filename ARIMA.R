setwd('D:/OneDrive - University of Glasgow/UofG/Traffic flow prediction_Part2/New_flow')



#------------------------------------------
#Exploratory data analysis

library(rgdal)

#Read csv file 
flow_data <- read.csv("GD1351_R_new.csv")
View(flow_data)
sapply(flow_data, class)

#select data from 2019-11-21
start_row <- which(flow_data$timeStamp == '2019-11-21 00:00:00') 
new_flow <- flow_data[7105:nrow(flow_data),]
row.names(new_flow) <- NULL  #reset the index
#rownames(new_flow) <- flow_data[7105:nrow(flow_data),'timeStamp']

View(new_flow)
sapply(new_flow, class)  #check the data type of each column
class(new_flow)

#Daily Traffic flow of GD1351_R 
plot(new_flow[1:96,"newFlow"], main="2019-11-21", ylab="Traffic flows", xlab="Time (in 15-min interval)", type="l")
#Weekly Traffic flow 
plot(new_flow[1:(96*7),"newFlow"],  main="from 2019-11-21 to 2019-11-28", ylab="Traffic flows", xlab="Time (in 15-min interval)", type="l")
# Monthly Traffic flow
plot(new_flow[1:(96*30),"newFlow"], main="from 2019-11-21 to 2019-12-21", ylab="Traffic flows", xlab="Time (in 15-min interval)", type="l")

lag.plot(new_flow[1:(96*7),"newFlow"], lags=96, do.lines=FALSE)
lag.plot(new_flow[1:(96*30),"newFlow"],lags=6, do.lines = FALSE)


#----------------------------------------------------
#Autocorrelation and partial autocorrelation analysis

#the ACF is used to determine the MA order of an ARIMA model and the PACF is used to determine the AR order. 
#The number of significant autocorrelations in each plot informs the order of p and q. 

acf(new_flow[1:(96*30),"newFlow"], lag.max=(96*30), xlab="Lag", ylab="ACF", main="Autocorrelation plot of traffic flow in 15-min interval")
acf(new_flow[1:(96*30),"newFlow"], lag.max=(96*2), xlab="Lag", ylab="ACF", main="Autocorrelation plot of traffic flow in 15-min interval")

# Differenced ACF plot of the traffic flow in 15-min interval 
plot(new_flow[1:(96*30),"newFlow"], ylab="Traffic flow", xlab="Time (in 15-min interval)", type="l")
TF.s.diff <- diff(new_flow[,"newFlow"], lag=96, differences=1)
acf(TF.s.diff, lag.max=(96*30), xlab="Lag", ylab="ACF", main="Differenced autocorrelation plot")

#partial autocorrelation plot of the monthly average temperatures in East Anglia.
pacf(new_flow[,"newFlow"], lag.max=(96*3),xlab="Lag",ylab="PACF",main="Partial Autocorrelation plot of Traffic flow in 15-min interval")
#partial autocorrelation plot of daily differenced data
pacf(TF.s.diff, lag.max=(96*3), xlab="Lag", ylab="ACF",main="Partial Autocorrelation plot of 15-min interval traffic flow")


#----------------------------------------------
# Parameter estimation and fitting

#The training set (first year, from from 2019-11-21 to 2020-12-21) is used for fitting the model (one hour for running)
fit.ar <- arima(new_flow[1:(96*30),"newFlow"],order=c(2,0,2),seasonal=list(order=c(2,1,1),period=96))
fit.ar

auto.arima(new_flow[1:(96*30),"newFlow"])
#When comparing two models on the same data, 
#a higher log likelihood is better, while a smaller AIC is better. 
#These numbers are not comparable between datasets.

#further test our model using the normalized root mean squared error (NRMSE). 
setwd('D:/Postgraduate/Programme/term2/CEGE0042/week3/Data')
source("Data/starima_package.R")
NRMSE_fit <- NRMSE(res=fit.ar$residuals, obs=new_flow[1:(96*30),"newFlow"])


#----------------------------------------------
#Diagnostic Checking

#test the residuals for normality.
#If the residuals are not normally distributed and uncorrelated 
#then there is more information in the errors that has not been accounted for in the model. 
tsdiag(fit.ar)


#----------------------------------------------
#Prediction
pre.ar<-predict(fit.ar, n.ahead=96)
matplot(1:96,cbind(new_flow[(96*30+1):(96*31),"newFlow"],pre.ar$pred),type="l",main="", xlab="15-min interval", ylab="Traffic flows")

# make one-step-ahead predictions using the newest available data at each time point
pre.Ar <- Arima(new_flow[(96*30+1):(96*37),"newFlow"], model=fit.ar)
#pre.Ar <- Arima(new_flow[(96*30+1):(nrow(new_flow)),"newFlow"], model=fit.Ar)
matplot(cbind(pre.Ar$fitted, pre.Ar$x), type="l")
