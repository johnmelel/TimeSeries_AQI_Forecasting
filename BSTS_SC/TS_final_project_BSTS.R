library(readxl)
library(tseries)
library(forecast)
library(hts)
library(zoo)
library(urca)
library(bsts)
library(lmtest)
library(bsts)
library(Boom)
library(parallel)
library(ldsr)

# load data 
dailyaqi <- read_excel("London_AQI_Daily.xlsx")
dailyaqi$Date <- as.Date(dailyaqi$Date, format="%Y-%m-%d")
head(dailyaqi)
summary(dailyaqi)

# examine trends and autocorrelation
plot(dailyaqi$USAQI, type = 'l')
acf(dailyaqi$USAQI)   # gradual decay  
pacf(dailyaqi$USAQI)  # peak at lag 1 suggests AR(1) process 

# check for stationarity 
adf.test(dailyaqi$USAQI)   # stationary 
kpss.test(dailyaqi$USAQI)  # non-stationary 

# convert to timeseries 
aqi_ts <- ts(dailyaqi$USAQI, frequency = 365, start = c(2013, 1))


#### apply first-order differencing ####
diff_aqi <- diff(aqi_ts, differences = 1)

# test again for stationarity
adf.test(diff_aqi)  # stationary
kpss.test(diff_aqi) # stationary
acf(diff_aqi)
pacf(diff_aqi)
" 
The time series still removes the slow decay in the ACF plot, but the PACF plot suggests that an MA term is likely needed 
due to the gradual decline seen. 

"
# still may need an AR() component in model 

#####

#### apply box-cox transformation ####
lambda <- BoxCox.lambda(aqi_ts)
boxcox_aqi <- BoxCox(aqi_ts, lambda)

# test again for stationarity
adf.test(boxcox_aqi)  # stationary
kpss.test(boxcox_aqi) # non-stationary
acf(boxcox_aqi)
pacf(boxcox_aqi)
# did not remove non-stationarity
# suggests box-cox transform alone is not enough to achieve stationarity

#####

#### apply differenced box-cox transformation ####
lambda <- BoxCox.lambda(aqi_ts)
boxcox_aqi <- BoxCox(aqi_ts, lambda)

diff_boxcox_aqi <- diff(boxcox_aqi, differences = 1)

# test again for stationarity
adf.test(diff_boxcox_aqi)  # stationary
kpss.test(diff_boxcox_aqi) # stationary
acf(diff_boxcox_aqi)
pacf(diff_boxcox_aqi)
# differencing the box-cox removes non-stationarity
# ACF plot also shows that the slow decay was removed, however there is still small peaks after lag 0

#####

#### apply log-transformation ####
log_aqi <- log(aqi_ts)

# test again for stationarity
adf.test(log_aqi)  # stationary
kpss.test(log_aqi)  # non-stationary
acf(log_aqi)
pacf(log_aqi)
# did not remove non-stationarity
# suggests box-cox transform alone is not enough to achieve stationarity

#####

#### apply seasonal differencing ####
seasonal_aqi <- diff(aqi_ts, lag = 365)

# test again for stationarity
adf.test(seasonal_aqi)  # stationary
kpss.test(seasonal_aqi) # stationary
acf(seasonal_aqi)
pacf(seasonal_aqi)
# The time series still shows slow decay in the ACF plot, but the PACF plot suggests that an AR term might be necessary. 


##### 


"
Based on the ACF plots and tests for stationarity, first-order differencing, differenced box-cox transformation, and
seasonal differencing transforms the data into a stationary time series. However, there seems to still have some residual autocorrelation
in these transformations, suggesting that including a AR component may improve the model. Therefore, we will run 3 BSTS models
on each of these transformations, with and without an AR component. We can then compare the accuracy of each model and determine
the best model to use. 

"

# check AR component for each transformations 
best_model <- auto.arima(diff_aqi)
summary(best_model)
checkresiduals(best_model)

best_model_seasonal <- auto.arima(seasonal_aqi, seasonal = TRUE)
summary(best_model_seasonal)
checkresiduals(best_model_seasonal)

best_model_boxcox <- auto.arima(diff_boxcox_aqi)
summary(best_model_boxcox)
checkresiduals(best_model_boxcox)

"
Based on the auto.arima for each of the three transformations, an AR(5) for both the differenced and seasonally differences transformations
was the best model. For the differenced box-cox transformation, an AR(2) was the best model. Therefore, these AR components will be used 
in the respective transformations.

"


# split the data into train and test 
train_data <- subset(dailyaqi, Date < as.Date("2025-1-01"))
test_data  <- subset(dailyaqi, Date >= as.Date("2025-1-01"))

# create time series 
aqi_train_ts <- ts(train_data$USAQI, frequency = 365, start = c(2013, 1))
aqi_test_ts  <- ts(test_data$USAQI, frequency = 365, start = c(2025, 1))

#### Apply 3 transformation chosen from above ####

# first-order differencing 
diff_train <- diff(aqi_train_ts, differences = 1)
diff_test <- diff(aqi_test_ts, differences = 1)

# differenced box-cox 
boxcox_train <- BoxCox(aqi_train_ts, lambda = (BoxCox.lambda(aqi_train_ts)))
boxcox_test <- BoxCox(aqi_test_ts, lambda = (BoxCox.lambda(aqi_test_ts)))

diff_boxcox_train <- diff(boxcox_train, differences = 1)
diff_boxcox_test <- diff(boxcox_test, differences = 1)

# seasonal differencing 
seasonal_train <- diff(aqi_train_ts, lag = 365)
# seasonal_test <- diff(aqi_test_ts, lag = 365)

#####


#######   START BSTS MODELING   #######   

# check cores for parallel processing 
detectCores()

#####  Differenced BSTS model  #####

ss_diff <- list()
ss_diff <- AddLocalLinearTrend(ss_diff, y = diff_train)
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 7, season.duration = 1)   # weekly
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 30, season.duration = 1)  # monthly 
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 365, season.duration = 1)  # yearly

# run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_diff <- bsts(diff_train, state.specification = ss_diff, niter = 1000, ping = 100)

checkresiduals(model_diff)

# plot residuals 
residuals_bsts_diff <- residuals(model_diff)
residuals_bsts_diff <- rowMeans(model_diff$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_diff <- as.numeric(residuals(model_diff))
plot(residuals_bsts_diff, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_diff[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_diff <- SuggestBurn(0.1, model_diff)
pred_diff <- predict(model_diff, horizon = 59, burn = burn_diff, quantiles = c(.025, .975))


# Accuracy metrics 

actual_values <- test_data$USAQI

# Reverse differencing
forecast_values_diff <- cumsum(c(tail(train_data$USAQI, 1), pred_diff$mean))[-1] 

rmse <- sqrt(mean((actual_values - forecast_values_diff)^2))
mae  <- mean(abs(actual_values - forecast_values_diff))
mape <- mean(abs((actual_values - forecast_values_diff) / actual_values)) * 100

cat("RMSE for 1st order Differencing:", rmse, "\n")
cat("MAE for 1st order Differencing: ", mae, "\n")
cat("MAPE for 1st order Differencing:", mape, "%\n")

#####

#####  Differenced BSTS model with AR(5) component  #####

ss_diff_ar <- list()
ss_diff_ar <- AddLocalLinearTrend(ss_diff_ar, y = diff_train)
ss_diff_ar <- AddAr(lags = 5, y = diff_train, state.specification = ss_diff_ar)
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 7, season.duration = 1)   # weekly
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 30, season.duration = 1)  # monthly 
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 365, season.duration = 1)  # yearly

# run model
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_diff_ar <- bsts(diff_train, state.specification = ss_diff_ar, niter = 1000, ping = 100)

checkresiduals(model_diff_ar)

# plot residuals 
residuals_bsts_diff_ar <- residuals(model_diff_ar)
residuals_bsts_diff_ar <- rowMeans(model_diff_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_diff_ar <- as.numeric(residuals(model_diff_ar))
plot(residuals_bsts_diff_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_diff_ar[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_diff_ar <- SuggestBurn(0.1, model_diff_ar)
pred_diff_ar <- predict(model_diff_ar, burn = burn_diff_ar, horizon = 59)


# Accuracy metrics 

actual_values <- test_data$USAQI

# Revert differencing by adding the last known actual AQI value
forecast_values_diff_ar <- cumsum(c(tail(train_data$USAQI, 1), pred_diff_ar$mean))[-1]

rmse <- sqrt(mean((actual_values - forecast_values_diff_ar)^2))
mae  <- mean(abs(actual_values - forecast_values_diff_ar))
mape <- mean(abs((actual_values - forecast_values_diff_ar) / actual_values)) * 100

cat("RMSE for 1st order Differencing w/ AR(5):", rmse, "\n")
cat("MAE for 1st order Differencing w/ AR(5): ", mae, "\n")
cat("MAPE for 1st order Differencing w/ AR(5):", mape, "%\n")

#####

#####  Differenced Box-Cox BSTS model  #####

ss_bc <- list()
ss_bc <- AddLocalLinearTrend(ss_bc, y = diff_boxcox_train)
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 7, season.duration = 1)   # weekly
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 30, season.duration = 1)  # monthly 
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 365, season.duration = 1)  # yearly

# run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_bc <- bsts(diff_boxcox_train, state.specification = ss_bc, niter = 1000, ping = 100)

checkresiduals(model_bc)

# plot residuals 
residuals_bsts_bc <- residuals(model_bc)
residuals_bsts_bc <- rowMeans(model_bc$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_bc <- as.numeric(residuals(model_bc))
plot(residuals_bsts_bc, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_bc[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_bc <- SuggestBurn(0.1, model_bc)

pred_bc <- predict(model_bc, burn = burn_bc, horizon = 59)


# Accuracy metrics 
actual_values <- test_data$USAQI

# reverse differencing and box-cox 
lambda_value <- BoxCox.lambda(aqi_train_ts)

forecast_values_bc <- cumsum(c(tail(train_data$USAQI, 1), pred_bc$mean))[-1]
# forecast_values_real <- inv_boxcox(forecast_values_bc, lambda = lambda_value)

# have to shift values to ensure positive values since some values in (forecast_values_bc * lambda_value + 1) are negative, causing the exponentiation to fail. 
forecast_values_real1 <- ((forecast_values_bc - min(forecast_values_bc) + 1) * lambda_value + 1)^(1/lambda_value)

rmse <- sqrt(mean((actual_values - forecast_values_real1)^2))
mae  <- mean(abs(actual_values - forecast_values_real1))
mape <- mean(abs((actual_values - forecast_values_real1) / actual_values)) * 100

cat("RMSE for BC transformation:", rmse, "\n")
cat("MAE for BC transformation: ", mae, "\n")
cat("MAPE for BC transformation:", mape, "%\n")

#####

#####  Differenced Box-Cox BSTS model with AR(2) component #####

ss_bc_ar <- list()
ss_bc_ar <- AddLocalLinearTrend(ss_bc_ar, y = diff_boxcox_train)
ss_bc_ar <- AddAr(lags = 2, y = diff_boxcox_train, state.specification = ss_bc_ar)
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 7, season.duration = 1)   # weekly
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 30, season.duration = 1)  # monthly 
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 365, season.duration = 1)  # yearly

# run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_bc_ar <- bsts(diff_boxcox_train, state.specification = ss_bc_ar, niter = 1000, ping = 100)

checkresiduals(model_bc_ar)

# plot residuals 
residuals_bsts_bc_ar <- residuals(model_bc_ar)
residuals_bsts_bc_ar <- rowMeans(model_bc_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_bc_ar <- as.numeric(residuals(model_bc_ar))
plot(residuals_bsts_bc_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_bc_ar[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_bc_ar <- SuggestBurn(0.1, model_bc_ar)
pred_bc_ar <- predict(model_bc_ar, burn = burn_bc_ar, horizon = 59)


# Accuracy metrics 
actual_values <- test_data$USAQI

# reverse differencing and box-cox 
lambda_value <- BoxCox.lambda(aqi_train_ts)
forecast_values_bc_ar <- cumsum(c(tail(train_data$USAQI, 1), pred_bc_ar$mean))[-1]
# have to shift values to ensure positive values since some values in (forecast_values_bc * lambda_value + 1) are negative, causing the exponentiation to fail. 
forecast_values_realar <- ((forecast_values_bc_ar - min(forecast_values_bc_ar) + 1) * lambda_value + 1)^(1/lambda_value)

# forecast_values_real_ar <- inv_boxcox(forecast_values_bc_ar, lambda = (BoxCox.lambda(aqi_train_ts)))

rmse <- sqrt(mean((actual_values - forecast_values_realar)^2))
mae  <- mean(abs(actual_values - forecast_values_realar))
mape <- mean(abs((actual_values - forecast_values_realar) / actual_values)) * 100

cat("RMSE for BC AR(2):", rmse, "\n")
cat("MAE for BC AR(2): ", mae, "\n")
cat("MAPE for BC AR(2):", mape, "%\n")

#####

#####  Seasonal Differenced BSTS model  #####

ss_season <- list()
ss_season <- AddLocalLinearTrend(ss_season, y = seasonal_train)
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 7, season.duration = 1)   # weekly
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 30, season.duration = 1)  # monthly 
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 365, season.duration = 1)  # yearly

# run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_season <- bsts(seasonal_train, state.specification = ss_season, niter = 1000, ping = 100)

checkresiduals(model_season)

# plot residuals 
residuals_bsts_season <- residuals(model_season)
residuals_bsts_season <- rowMeans(model_season$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_season <- as.numeric(residuals(model_season))
plot(residuals_bsts_season, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_season[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_season <- SuggestBurn(0.1, model_season)
pred_season <- predict(model_season, burn = burn_season, horizon = 59)


# Accuracy metrics 

actual_values <- test_data$USAQI

# Reverse seasonal differencing
forecast_values_season <- (aqi_train_ts[(length(aqi_train_ts) - 365 + 1):(length(aqi_train_ts) - 365 + 59)]) + pred_season$mean

rmse <- sqrt(mean((actual_values - forecast_values_season)^2))
mae  <- mean(abs(actual_values - forecast_values_season))
mape <- mean(abs((actual_values - forecast_values_season) / actual_values)) * 100

cat("RMSE for Seasonal Differencing:", rmse, "\n")
cat("MAE for Seasonal Differencing: ", mae, "\n")
cat("MAPE for Seasonal Differencing:", mape, "%\n")

#####

#####  Seasonal Differenced BSTS model with AR(5) component #####

ss_season_ar <- list()
ss_season_ar <- AddLocalLinearTrend(ss_season_ar, y = seasonal_train)
ss_season_ar <- AddAr(order = 5, y = seasonal_train, state.specification = ss_season_ar)
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 7, season.duration = 1)   # weekly
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 30, season.duration = 1)  # monthly 
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 365, season.duration = 1)  # yearly

# run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_season_ar <- bsts(seasonal_train, state.specification = ss_season_ar, niter = 1000, ping = 100)

checkresiduals(model_season_ar)

# plot residuals 
residuals_bsts_season_ar <- residuals(model_season_ar)
residuals_bsts_season_ar <- rowMeans(model_season_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_season_ar <- as.numeric(residuals(model_season_ar))
plot(residuals_bsts_season_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_season_ar[1:5000])


# forecast the next 59 days (Jan 2025 - Feb 2025)
burn_season_ar <- SuggestBurn(0.1, model_season_ar)
pred_season_ar <- predict(model_season_ar, burn = burn_season_ar, horizon = 59)


# Accuracy metrics 

actual_values <- test_data$USAQI

# Reverse seasonal differencing
forecast_values_season_ar <- (aqi_train_ts[(length(aqi_train_ts) - 365 + 1):(length(aqi_train_ts) - 365 + 59)]) + pred_season_ar$mean


rmse <- sqrt(mean((actual_values - forecast_values_season_ar)^2))
mae  <- mean(abs(actual_values - forecast_values_season_ar))
mape <- mean(abs((actual_values - forecast_values_season_ar) / actual_values)) * 100

cat("RMSE for Seasonal Differencing w/ AR(5):", rmse, "\n")
cat("MAE for Seasonal Differencing w/ AR(5): ", mae, "\n")
cat("MAPE for Seasonal Differencing w/ AR(5):", mape, "%\n")

#####


