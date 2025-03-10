library(readxl)
library(tseries)
library(forecast)
library(hts)
library(zoo)
library(urca)
library(bsts)
library(lmtest)


# load data 
setwd("~/Downloads")
dailyaqi <- read_excel("London_AQI_Daily.xlsx")
dailyaqi$Date <- as.Date(dailyaqi$Date, format="%Y-%m-%d")
head(dailyaqi)
summary(dailyaqi)


plot(dailyaqi$USAQI, type = 'l')
acf(dailyaqi$USAQI)
pacf(dailyaqi$USAQI)
ggseasonplot(aqi_train_ts)

kpss.test(dailyaqi$USAQI)  # non-stationary 


train_data <- subset(dailyaqi, Date < as.Date("2024-12-01"))
test_data  <- subset(dailyaqi, Date >= as.Date("2024-12-01"))

aqi_train_ts <- ts(train_data$USAQI, frequency = 365, start = c(2013, 1))
aqi_test_ts  <- ts(test_data$USAQI, frequency = 365, start = c(2024, 12))


adf.test(aqi_train_ts) # stationary

# may have a deterministic trend since kpss test and adf test contradict eachother 
# use AddLocalLinearTrend to model any residual trends



####==== run BSTS model ====####

ss <- list()
ss <- AddLocalLinearTrend(ss, y = aqi_train_ts) 
ss <- AddSeasonal(ss, y = train_data$USAQI, nseasons = 365, season.duration = 1) # yearly seasonality

# fit BSTS model on training set
model <- bsts(aqi_train_ts, state.specification = ss, niter = 200)

residuals_bsts <- residuals(model)
summary(model)

# plot residuals
residuals_bsts <- as.numeric(residuals(model))  # flatten to a vector
plot(residuals_bsts)
acf(residuals_bsts[1:5000])  # demonstrates autocorrelation, needs differencing

# test for Heteroscedasticity
bptest(lm(residuals_bsts ~ seq_along(residuals_bsts)))


# difference the data
aqi_train_ts_diff <- diff(aqi_train_ts, differences = 1)
acf(aqi_train_ts_diff) # no auto-correlation 



ss_diff <- list()
ss_diff <- AddLocalLinearTrend(ss_diff, y = aqi_train_ts_diff)
ss_diff <- AddSeasonal(ss_diff, y = aqi_train_ts_diff, nseasons = 7, season.duration = 1)   # weekly
ss_diff <- AddSeasonal(ss_diff, y = aqi_train_ts_diff, nseasons = 30, season.duration = 1)  # monthly 
ss_diff <- AddSeasonal(ss_diff, y = aqi_train_ts_diff, nseasons = 365, season.duration = 1)  # yearly

# re-run on difference data
model_diff <- bsts(aqi_train_ts_diff, state.specification = ss_diff, niter = 100)

# plot residuals 
residuals_bsts_diff <- residuals(model_diff)
residuals_bsts_diff <- rowMeans(model_diff$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_diff <- as.numeric(residuals(model_diff))
plot(residuals_bsts_diff, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_diff[1:5000])


# forecast the next 31 days (December 2024)
pred <- predict(model_diff, horizon = 31)

# Accuracy metrics 
actual_values <- test_data$USAQI
forecast_values <- cumsum(c(tail(train_data$USAQI, 1), pred$mean))[-1] # Revert differencing by adding the last known actual AQI value
# forecast_values <- pred$mean

rmse <- sqrt(mean((actual_values - forecast_values)^2))
mae  <- mean(abs(actual_values - forecast_values))
mape <- mean(abs((actual_values - forecast_values) / actual_values)) * 100

cat("RMSE:", rmse, "\n")
cat("MAE: ", mae, "\n")
cat("MAPE:", mape, "%\n")





