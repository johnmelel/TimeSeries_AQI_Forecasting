# Time Series Final Project: Forecasting London's AQI

# Group: Anusha Bhat, Sara Chaker, Hritik Jhaveri, John Melel
# March 13, 2025

# link to github for all files 
https://github.com/johnmelel/TimeSeries_AQI_Forecasting


########## Libraries ##########

library(readxl)
library(tseries)
library(forecast)
library(hts)
library(zoo)
library(tidyverse)
library(tseries)
library(lmtest)
library(FinTS)
library(fracdiff)
library(urca)
library(bsts)
library(lmtest)
library(bsts)
library(Boom)
library(parallel)
library(ldsr)


########## Data Cleaning: Everyone ##########

# Note: We did the data cleaning mainly in Excel. We did linear interpolation
# of the hourly data to fill in for missing values. Then we aggregated the data
# into daily values.

# Load the data
aqi = read_excel("London_AQI_Daily.xlsx")
head(aqi)

# Check the structure of the data
summary(aqi$USAQI)

# Check outliers
hist(aqi$USAQI,
     xlab = "USAQI",
     main = "Histogram of USAQI",
     breaks = sqrt(length(aqi$USAQI)) # set number of bins
)

qqnorm(aqi$USAQI, main = "QQ Plot of USAQI")
qqline(aqi$USAQI, col = "blue")

# There are a few large outliers so data transformation may be necessary

# Correct date structure
aqi$Date = as.Date(aqi$Date, format = "%Y-%m-%d")
aqi$USAQI = as.integer(aqi$USAQI)

# Convert to time series
aqi_ts = ts(aqi$USAQI, frequency = 365, start = c(2013, 1, 2))

########## EDA: Everyone ##########

# Plot time series
plot(y = aqi$USAQI, x = aqi$Date, main = "Time Series of London AQI", type = "l",
     ylab = "AQI", xlab = "Date")

# ACF plot
acf(aqi_ts, main = "ACF of AQI")

# PACF plot
pacf(aqi_ts, main = "PACF of AQI")
# Lags have autocorrelation based on ACF but pacf shows limited autocorrelation
# when removing effect of short term lags

# Stationarity test --> not stationary
adf.test(aqi_ts)
kpss.test(aqi_ts)

# Apply differencings
new_aqi_ts = diff(aqi_ts)
acf(new_aqi, main = "ACF After Differencing")
pacf(new_aqi, main = "PACF After Differencing")
adf.test(new_aqi_ts) # stationary
kpss.test(new_aqi_ts) # stationary
# There's still autocorrelation after differencing, may need an AR component

# Seasonal decomposition on differenced time series
decomp = stl(new_aqi, s.window = "periodic")
plot(decomp, main = "Seasonal Decomposition")
# There seems to be a decreasing but stable trend over time with seasonality.
# The overall trend is very small in value, however.

# Apply Box-Cox transformation
lambda <- BoxCox.lambda(aqi_ts)
boxcox_aqi <- BoxCox(aqi_ts, lambda)

# Test again for stationarity
adf.test(boxcox_aqi)  # stationary
kpss.test(boxcox_aqi) # non-stationary
acf(boxcox_aqi)
pacf(boxcox_aqi)
# Did not remove non-stationarity
# Suggests box-cox transform alone is not enough to achieve stationarity


# Difference the Box-Cox transformation
diff_boxcox_aqi <- diff(boxcox_aqi, differences = 1)

# Ttest again for stationarity
adf.test(diff_boxcox_aqi)  # stationary
kpss.test(diff_boxcox_aqi) # stationary
acf(diff_boxcox_aqi)
pacf(diff_boxcox_aqi)
# Differencing the box-cox removes non-stationarity
# ACF plot also shows that the slow decay was removed, however there are still
# small peaks after lag 0

# Apply seasonal differencing
seasonal_aqi <- diff(aqi_ts, lag = 365)

# Test again for stationarity
adf.test(seasonal_aqi)  # stationary
kpss.test(seasonal_aqi) # stationary
acf(seasonal_aqi)
pacf(seasonal_aqi)
# The time series still shows slow decay in the ACF plot, but the PACF plot
# suggests that an AR component might be necessary.


########## Train/Test Split & Transformations ##########

# Split the data into train and test
train_data <- subset(dailyaqi, Date < as.Date("2025-1-01"))
test_data  <- subset(dailyaqi, Date >= as.Date("2025-1-01"))

# Create time series 
aqi_train_ts <- ts(train_data$USAQI, frequency = 365, start = c(2013, 1))
aqi_test_ts  <- ts(test_data$USAQI, frequency = 365, start = c(2025, 1))

# First-order differencing
diff_train <- diff(aqi_train_ts, differences = 1)

# Differenced box-cox
boxcox_train <- BoxCox(aqi_train_ts, lambda = (BoxCox.lambda(aqi_train_ts)))
diff_boxcox_train <- diff(boxcox_train, differences = 1)

# Seasonal differencing
seasonal_train <- diff(aqi_train_ts, lag = 365)

# We will construct our ARIMA/SARIMA/ARFIMA, ETS, Holt Winters, and BSTS models
# on the differenced, Box-Cox with difference, and seasonally differenced data
# since these time series have stationarity and help with the outliers. For the
# exponential smoothing and BSTS models, we will incorporate AR components to
# help minimize the autocorrelations.




########## ARIMA/SARIMA/ARFIMA Models: Hritik ##########
# Ensure ts_data is a daily time series object
ts_data <- ts(aqi$USAQI, start = c(2013, 1, 2), frequency = 365)

# Apply transformations to the entire dataset
transformed_data <- list(
  "1st_diff" = diff(ts_data, differences = 1),
  "diff_boxcox" = diff(BoxCox(ts_data, lambda = BoxCox.lambda(ts_data)),
                       differences = 1),
  "seasonal_diff" = diff(ts_data, lag = 365)
)

# Split data into train and test sets
train <- window(ts_data, end = c(2024, 367))
test <- window(ts_data, start = c(2025, 3))


# Make list of transformations
for (name in names(transformed_data)) {
  series <- transformed_data[[name]]

  par(mfrow = c(1, 2))  # Set layout for two plots side by side

  # ACF Plot
  acf(series, main = paste(name, "- ACF"))

  # PACF Plot
  pacf(series, main = paste(name, "- PACF"))
}

# Model Fitting 

# Initialize results list
results <- list()

# Fit models to each transformed series
for (name in names(transformed_data)) {
  train_series <- transformed_data[[name]]

  # ARIMA Model
  arima_model <- auto.arima(train_series)
  results[[paste0(name, "_ARIMA")]] <- arima_model

  # SARIMA Model
  sarima_model <- auto.arima(train_series, seasonal = TRUE)
  results[[paste0(name, "_SARIMA")]] <- sarima_model

  # ARFIMA Model
  arfima_model <- arfima(train_series)
  results[[paste0(name, "_ARFIMA")]] <- arfima_model
}


# ACF and PACF plots
for (name in names(transformed_data)) {
  series <- transformed_data[[name]]

  par(mfrow = c(1, 2))  

  # ACF Plot
  acf(series, main = paste(name, "- ACF"))

  # PACF Plot
  pacf(series, main = paste(name, "- PACF"))
}

# Residuals diagnostics
for (name in names(transformed_data)) {
  for (model_type in c("ARIMA", "SARIMA", "ARFIMA")) {
    model_name <- paste0(name, "_", model_type)
    print(paste("Diagnostics for", model_name))
    checkresiduals(results[[model_name]])
    
    # Residuals vs. Fitted Plot
    plot(results[[model_name]][["fitted"]], results[[model_name]][["residuals"]],
         main = paste(model_name, "- Residuals vs. Fitted"),
         xlab = "Fitted Values", ylab = "Residuals",
         col = "blue", pch = 19)
    abline(h = 0, col = "red", lwd = 2)
  }
}

# The residuals are all stationary. Some of the models have slight
# heteroskedasticity and the errors have autocorrelation in the earlier lags

for (name in names(transformed_data)) {
  for (model_type in c("ARIMA", "SARIMA", "ARFIMA")) {
    model_name <- paste0(name, "_", model_type)
    print(paste("Diagnostics for", model_name))
    print(adf.test(residuals(results[[model_name]])))
    print(kpss.test(residuals(results[[model_name]])))
  }
}

# Forecasting 
forecast_horizon <- length(test)
forecasts <- list()

for (name in names(transformed_data)) {
  for (model_type in c("ARIMA", "SARIMA", "ARFIMA")) {
    model_name <- paste0(name, "_", model_type)
    forecasts[[model_name]] <- forecast(results[[model_name]],
                                        h = forecast_horizon)
  }
}


for (model_name in names(forecasts)) {
  plot(forecasts[[model_name]], main = paste(model_name, "Forecast"))
  lines(test, col = "red")
  legend("topright", legend = c("Forecast", "Actual"), col = c("black", "red"),
         lty = 1)
}

# Testing 
# Function to reverse 1st differencing

reverse_diff <- function(predicted, original) {
  cumsum(c(original[length(original)], predicted))[-1]
}

# Function to reverse Box-Cox differencing
reverse_diff_boxcox <- function(predicted, original, lambda) {
  inv_boxcox <- function(x, lambda) {
    if (lambda == 0) {
      exp(x)
    } else {
      (lambda * (x - min(x) + 1) + 1)^(1 / lambda)
    }
  }
  cumsum(c(original[length(original)], predicted))[-1] %>% inv_boxcox(lambda)
}

# Function to reverse seasonal differencing
reverse_seasonal_diff <- function(predicted, original) {
  return(original[(length(original) - 365 + 1):(length(original) - 365 + 59)] + predicted)
}


# Undo differencing for each forecast
for (name in names(forecasts)) {
  if (grepl("1st_diff", name)) {
    forecasts[[name]]$mean <- reverse_diff(forecasts[[name]]$mean, ts_data)
  } else if (grepl("diff_boxcox", name)) {
    lambda <- BoxCox.lambda(ts_data)
    forecasts[[name]]$mean <- reverse_diff_boxcox(forecasts[[name]]$mean, ts_data, lambda)
  } else if (grepl("seasonal_diff", name)) {
    forecasts[[name]]$mean <- reverse_seasonal_diff(forecasts[[name]]$mean, ts_data)
  }
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}


rmse_results <- list()
for (model_name in names(forecasts)) {
  model_rmse <- rmse(test, forecasts[[model_name]]$mean)
  print(model_rmse)
  rmse_results[[model_name]] <- model_rmse
}

print(rmse_results)

rmse_table <- data.frame(
  Transformation = c("1st Differencing", "Box-Cox Differencing", "Seasonal Differencing"),
  ARIMA_RMSE = c(23.56, 42.53, 27.22),
  SARIMA_RMSE = c(23.56, 42.53, 27.22),
  ARFIMA_RMSE = c(18.68, 42.64, 26.89)
)

print(rmse_table)

# 1st differencing ARFIMA is the best. ARFIMA best overall.


########## Exponential Smoothing Models: Anusha ##########

# ETS Models


# Differenced data

# Model
diff_ets_mod = ets(diff_train)
summary(diff_ets_mod)

checkresiduals(diff_ets_mod)
diff_for = forecast(diff_ets_mod, h = 59)
plot(diff_for, main = "Forecast for 2025 (Differenced)", ylab = "AQI", xlab = "Date")
plot(fitted(diff_ets_mod), residuals(diff_ets_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing 
forecast_values_diff = cumsum(c(tail(train_data$USAQI, 1), diff_for$mean))[-1]
diff_resids = aqi_test_ts - forecast_values_diff
diff_rmse = sqrt(mean(diff_resids^2))
cat("Test RMSE:", diff_rmse)

# Add AR Component (AR(5)) since the residuals are autocorrelated
diff_ar = auto.arima(diff_train)
summary(diff_ar)
checkresiduals(diff_ar)
Box.test(residuals(diff_ets_mod) + residuals(diff_ar), lag = 10)

# Testing
diff_for_ar = forecast(diff_ar, h = 59)
diff_preds = forecast_values_diff + diff_for_ar$mean

diff_cets_rmse = sqrt(mean((aqi_test_ts - diff_preds)^2))
cat("Test RMSE:", diff_cets_rmse)

# Residuals are still autocorrelated and the test RMSE went up.


# Box-Cox with differencing

# Model
dbc_ets_mod = ets(diff_boxcox_train)
summary(dbc_ets_mod)

checkresiduals(dbc_ets_mod)
dbc_for = forecast(dbc_ets_mod, h = 59)
plot(diff_for, main = "Forecast for 2025 (Differenced BC)", ylab = "AQI", xlab = "Date")
plot(fitted(dbc_ets_mod), residuals(dbc_ets_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing
# reverse differencing and box-cox
lambda_value = BoxCox.lambda(aqi_train_ts)z

forecast_values_bc = cumsum(c(tail(train_data$USAQI, 1), dbc_for$mean))[-1]
# have to shift values to ensure positive values since some values in
# (forecast_values_bc * lambda_value + 1) are negative, causing the
# exponentiation to fail.
forecast_values_bc = ((forecast_values_bc - min(forecast_values_bc) + 1) * lambda_value + 1)^(1/lambda_value)
dbc_resids = aqi_test_ts - forecast_values_bc
dbc_rmse = sqrt(mean(dbc_resids^2))
cat("Test RMSE:", dbc_rmse)

# Add AR Component (AR(2)) since the residuals are autocorrelated
dbc_ar = auto.arima(boxcox_train)
summary(dbc_ar)
checkresiduals(dbc_ar)
Box.test(rdbc_for_ar = forecast(dbc_ar, h = 59)

# Testing
dbc_preds = forecast_values_bc + dbc_for_ar$mean
dbc_cets_rmse = sqrt(mean((aqi_test_ts - dbc_preds)^2))
cat("Test RMSE:", dbc_cets_rmse)esiduals(dbc_ets_mod) + residuals(dbc_ar), lag = 10)

# Residuals are still autocorrelated but they decreased a little with using the AR.


# Seasonally Differenced

# Model
ds_ets_mod = ets(seasonal_train)
summary(ds_ets_mod)

checkresiduals(ds_ets_mod)
ds_for = forecast(ds_ets_mod, h = 59)
plot(ds_for, main = "Forecast for 2025 (Differenced Seasonal)", ylab = "AQI",
     xlab = "Date", xlim = c(2024, 2025))
plot(fitted(ds_ets_mod), residuals(ds_ets_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing
original = aqi_train_ts
forecast_values_season = (original[(length(original) - 365 + 1):(length(original) - 365 + 59)]) + ds_for$mean
ds_resids = aqi_test_ts - forecast_values_season
ds_rmse = sqrt(mean(ds_resids^2))
cat("Test RMSE:", ds_rmse)

# Add AR Component (AR(5)) since the residuals are autocorrelated
ds_ar = auto.arima(seasonal_train)
summary(ds_ar)
checkresiduals(ds_ar)
Box.test(residuals(ds_ets_mod) + residuals(ds_ar), lag = 10)

# Testing 
ds_for_ar = forecast(ds_ar, h = 59)
ds_preds = forecast_values_season + ds_for_ar$mean
ds_cets_rmse = sqrt(mean((aqi_test_ts - ds_preds)^2))
cat("Test RMSE:", ds_cets_rmse)



# Holt-Winters Models


# Differenced data

# Model
diff_hw_mod = HoltWinters(diff_train, seasonal = "additive")
summary(diff_hw_mod)

checkresiduals(diff_hw_mod)
diff_hw_for = forecast(diff_hw_mod, h = 59)
plot(diff_hw_for, main = "Forecast for 2025 HW (Differenced)")
plot(diff_hw_mod$fitted[,1], residuals(diff_hw_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing 
forecast_values_hw_diff =  cumsum(c(tail(train_data$USAQI, 1),
                                    diff_hw_for$mean))[-1]
diff_hw_res = aqi_test_ts - forecast_values_hw_diff
diff_hw_rmse = sqrt(mean(diff_hw_res^2))
cat("Test RMSE:", diff_hw_rmse)

# Add AR Component (AR(5)) since the residuals are autocorrelated
Box.test(residuals(diff_hw_mod) + residuals(diff_ar), lag = 10)

# Testing 
diff_preds2 = forecast_values_hw_diff + diff_for_ar$mean
diff_chw_rmse = sqrt(mean((aqi_test_ts - diff_preds2)^2))
cat("Test RMSE:", diff_chw_rmse)
# Residuals are still autocorrelated and the test RMSE went up.


# Box-Cox with differencing

# Model
dbc_hw_mod = HoltWinters(diff_boxcox_train, seasonal = "additive")
summary(dbc_hw_mod)

checkresiduals(dbc_hw_mod)
dbc_hw_for = forecast(dbc_hw_mod, h = 59)
plot(diff_hw_for, main = "Forecast for 2025 HW (Differenced BC)")
plot(dbc_hw_mod$fitted[,1], residuals(dbc_hw_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing
# reverse differencing and box-cox
lambda_value = BoxCox.lambda(aqi_train_ts)
forecast_values_hw_bc = cumsum(c(tail(train_data$USAQI, 1), dbc_hw_for$mean))[-1]
forecast_values_hw_bc = ((forecast_values_hw_bc - min(forecast_values_hw_bc) + 1) * lambda_value + 1)^(1/lambda_value)

dbc_hw_res = aqi_test_ts - forecast_values_hw_bc
dbc_hw_rmse = sqrt(mean(dbc_hw_res^2))
cat("Test RMSE:", dbc_hw_rmse)

# Add AR Component (AR(2)) since the residuals are autocorrelated
Box.test(residuals(dbc_hw_mod) + residuals(dbc_ar), lag = 10)

# Testing
dbc_preds2 = forecast_values_hw_diff + diff_for_ar$mean

dbc_chw_rmse = sqrt(mean((aqi_test_ts - dbc_preds2)^2))
cat("Test RMSE:", dbc_chw_rmse)
# Residuals are still autocorrelated but the test had much lower RMSE.


# Seasonally Differenced

# Model
ds_hw_mod = HoltWinters(seasonal_train, seasonal = "additive")
summary(ds_hw_mod)

checkresiduals(ds_hw_mod)
dds_hw_for = forecast(ds_hw_mod, h = 59)
plot(diff_hw_for, main = "Forecast for 2025 HW (Differenced Seasonal)")
plot(ds_hw_mod$fitted[,1], residuals(ds_hw_mod),
     main="Residuals vs Fitted Values",
     xlab="Fitted Values",
     ylab="Residuals",
     pch = 16)
abline(h=0, col="red")

# Testing
original = aqi_train_ts
forecast_values_hw_season = (original[(length(original) - 365 + 1):(length(original) - 365 + 59)]) + ds_hw_for$mean
ds_resids = aqi_test_ts - forecast_values_hw_season
ds_hw_rmse = sqrt(mean(ds_resids^2))
cat("Test RMSE:", ds_hw_rmse)

# Add AR Component (AR(5)) since the residuals are autocorrelated
Box.test(residuals(ds_hw_mod) + residuals(ds_ar), lag = 10)

# Testing 
ds_preds2 = forecast_values_hw_season + diff_for_ar$mean
ds_chw_rmse = sqrt(mean((aqi_test_ts - ds_preds2)^2))
cat("Test RMSE:", ds_chw_rmse)
# Residuals are still autocorrelated but the test had similar RMSE.

# Model Comparison

# ETS
ets_rmse_table <- data.frame(
  "Differenced" = diff_rmse,
  "Differenced + ARIMA" = diff_cets_rmse,
  "Differenced Boxcox" = dbc_rmse,
  "Differenced Boxcox + ARIMA" = dbc_cets_rmse,
  "Seasonal" = ds_rmse,
  "Seasonal + ARIMA" = ds_cets_rmse
)
rownames(ets_rmse_table) <- c("Test Accuracy")

print(ets_rmse_table)

# HW
hw_rmse_table <- data.frame(
  "Differenced" = diff_hw_rmse,
  "Differenced + ARIMA" = diff_chw_rmse,
  "Differenced Boxcox" = dbc_hw_rmse,
  "Differenced Boxcox + ARIMA" = dbc_chw_rmse,
  "Seasonal" = ds_hw_rmse,
  "Seasonal + ARIMA" = ds_chw_rmse
)
rownames(hw_rmse_table) <- c("Test Accuracy")

print(hw_rmse_table)

# Holt-winters made seasonal and differenced perform worse but boxcox
# differenced performed better. For both, differenced was the best performing,
# though it had lower RMSE for the simple ets model.


########## BSTS Models: Sarah ##########

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

# Check cores for parallel processing
detectCores()

# Differenced

ss_diff <- list()
ss_diff <- AddLocalLinearTrend(ss_diff, y = diff_train)
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 7, season.duration = 1)   # weekly
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 30, season.duration = 1)  # monthly
ss_diff <- AddSeasonal(ss_diff, y = diff_train, nseasons = 365, season.duration = 1)  # yearly

# Run model with parallel processing to fix crashing
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_diff <- bsts(diff_train, state.specification = ss_diff, niter = 1000, ping = 100)
checkresiduals(model_diff)

# Plot residuals
residuals_bsts_diff <- residuals(model_diff)
residuals_bsts_diff <- rowMeans(model_diff$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_diff <- as.numeric(residuals(model_diff))
plot(residuals_bsts_diff, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_diff[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
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

# Differenced + AR(2) component

ss_diff_ar <- list()
ss_diff_ar <- AddLocalLinearTrend(ss_diff_ar, y = diff_train)
ss_diff_ar <- AddAr(lags = 5, y = diff_train, state.specification = ss_diff_ar)
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 7, season.duration = 1)   # weekly
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 30, season.duration = 1)  # monthly 
ss_diff_ar <- AddSeasonal(ss_diff_ar, y = diff_train, nseasons = 365, season.duration = 1)  # yearly

# Run model
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_diff_ar <- bsts(diff_train, state.specification = ss_diff_ar, niter = 1000, ping = 100)
checkresiduals(model_diff_ar)

# Plot residuals
residuals_bsts_diff_ar <- residuals(model_diff_ar)
residuals_bsts_diff_ar <- rowMeans(model_diff_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_diff_ar <- as.numeric(residuals(model_diff_ar))
plot(residuals_bsts_diff_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_diff_ar[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
burn_diff_ar <- SuggestBurn(0.1, model_diff_ar)
pred_diff_ar <- predict(model_diff_ar, burn = burn_diff_ar, horizon = 59)


# Accuracy metrics
# Revert differencing by adding the last known actual AQI value
forecast_values_diff_ar <- cumsum(c(tail(train_data$USAQI, 1), pred_diff_ar$mean))[-1]

rmse <- sqrt(mean((actual_values - forecast_values_diff_ar)^2))
mae  <- mean(abs(actual_values - forecast_values_diff_ar))
mape <- mean(abs((actual_values - forecast_values_diff_ar) / actual_values)) * 100

cat("RMSE for 1st order Differencing w/ AR(5):", rmse, "\n")
cat("MAE for 1st order Differencing w/ AR(5): ", mae, "\n")
cat("MAPE for 1st order Differencing w/ AR(5):", mape, "%\n")


# Differenced Box-Cos

ss_bc <- list()
ss_bc <- AddLocalLinearTrend(ss_bc, y = diff_boxcox_train)
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 7, season.duration = 1)   # weekly
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 30, season.duration = 1)  # monthly
ss_bc <- AddSeasonal(ss_bc, y = diff_boxcox_train, nseasons = 365, season.duration = 1)  # yearly

# Run model with parallel processing to fix crashing
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_bc <- bsts(diff_boxcox_train, state.specification = ss_bc, niter = 1000, ping = 100)
checkresiduals(model_bc)

# Plot residuals
residuals_bsts_bc <- residuals(model_bc)
residuals_bsts_bc <- rowMeans(model_bc$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_bc <- as.numeric(residuals(model_bc))
plot(residuals_bsts_bc, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_bc[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
burn_bc <- SuggestBurn(0.1, model_bc)
pred_bc <- predict(model_bc, burn = burn_bc, horizon = 59)


# Accuracy metrics

# Reverse differencing and box-cox
lambda_value <- BoxCox.lambda(aqi_train_ts)

forecast_values_bc <- cumsum(c(tail(train_data$USAQI, 1), pred_bc$mean))[-1]
forecast_values_real1 <- ((forecast_values_bc - min(forecast_values_bc) + 1) * lambda_value + 1)^(1/lambda_value)

rmse <- sqrt(mean((actual_values - forecast_values_real1)^2))
mae  <- mean(abs(actual_values - forecast_values_real1))
mape <- mean(abs((actual_values - forecast_values_real1) / actual_values)) * 100

cat("RMSE for BC transformation:", rmse, "\n")
cat("MAE for BC transformation: ", mae, "\n")
cat("MAPE for BC transformation:", mape, "%\n")


#  Differenced Box-Cox BSTS model with AR(2) component #####

ss_bc_ar <- list()
ss_bc_ar <- AddLocalLinearTrend(ss_bc_ar, y = diff_boxcox_train)
ss_bc_ar <- AddAr(lags = 2, y = diff_boxcox_train, state.specification = ss_bc_ar)
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 7, season.duration = 1)   # weekly
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 30, season.duration = 1)  # monthly 
ss_bc_ar <- AddSeasonal(ss_bc_ar, y = diff_boxcox_train, nseasons = 365, season.duration = 1)  # yearly

# Run model with parallel processing to fix crashing
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_bc_ar <- bsts(diff_boxcox_train, state.specification = ss_bc_ar, niter = 1000, ping = 100)

checkresiduals(model_bc_ar)

# Plot residuals
residuals_bsts_bc_ar <- residuals(model_bc_ar)
residuals_bsts_bc_ar <- rowMeans(model_bc_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_bc_ar <- as.numeric(residuals(model_bc_ar))
plot(residuals_bsts_bc_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_bc_ar[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
burn_bc_ar <- SuggestBurn(0.1, model_bc_ar)
pred_bc_ar <- predict(model_bc_ar, burn = burn_bc_ar, horizon = 59)


# Accuracy metrics

# Reverse differencing and box-cox
lambda_value <- BoxCox.lambda(aqi_train_ts)
forecast_values_bc_ar <- cumsum(c(tail(train_data$USAQI, 1), pred_bc_ar$mean))[-1]
forecast_values_realar <- ((forecast_values_bc_ar - min(forecast_values_bc_ar) + 1) * lambda_value + 1)^(1/lambda_value)

rmse <- sqrt(mean((actual_values - forecast_values_realar)^2))
mae  <- mean(abs(actual_values - forecast_values_realar))
mape <- mean(abs((actual_values - forecast_values_realar) / actual_values)) * 100

cat("RMSE for BC AR(2):", rmse, "\n")
cat("MAE for BC AR(2): ", mae, "\n")
cat("MAPE for BC AR(2):", mape, "%\n")


#  Seasonal Differencing

ss_season <- list()
ss_season <- AddLocalLinearTrend(ss_season, y = seasonal_train)
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 7, season.duration = 1)   # weekly
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 30, season.duration = 1)  # monthly 
ss_season <- AddSeasonal(ss_season, y = seasonal_train, nseasons = 365, season.duration = 1)  # yearly

# Run model with parallel processing to fix crashing
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_season <- bsts(seasonal_train, state.specification = ss_season, niter = 1000, ping = 100)

checkresiduals(model_season)

# Plot residuals 
residuals_bsts_season <- residuals(model_season)
residuals_bsts_season <- rowMeans(model_season$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_season <- as.numeric(residuals(model_season))
plot(residuals_bsts_season, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_season[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
burn_season <- SuggestBurn(0.1, model_season)
pred_season <- predict(model_season, burn = burn_season, horizon = 59)


# Accuracy metrics

# Reverse seasonal differencing
forecast_values_season <- (aqi_train_ts[(length(aqi_train_ts) - 365 + 1):(length(aqi_train_ts) - 365 + 59)]) + pred_season$mean

rmse <- sqrt(mean((actual_values - forecast_values_season)^2))
mae  <- mean(abs(actual_values - forecast_values_season))
mape <- mean(abs((actual_values - forecast_values_season) / actual_values)) * 100

cat("RMSE for Seasonal Differencing:", rmse, "\n")
cat("MAE for Seasonal Differencing: ", mae, "\n")
cat("MAPE for Seasonal Differencing:", mape, "%\n")


#  Seasonal Differencing

ss_season_ar <- list()
ss_season_ar <- AddLocalLinearTrend(ss_season_ar, y = seasonal_train)
ss_season_ar <- AddAr(order = 5, y = seasonal_train, state.specification = ss_season_ar)
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 7, season.duration = 1)   # weekly
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 30, season.duration = 1)  # monthly 
ss_season_ar <- AddSeasonal(ss_season_ar, y = seasonal_train, nseasons = 365, season.duration = 1)  # yearly

# Run model with parallel processing to fix crashing 
options(mc.cores = parallel::detectCores() - 4) # use 10 cores
model_season_ar <- bsts(seasonal_train, state.specification = ss_season_ar, niter = 1000, ping = 100)

checkresiduals(model_season_ar)

# Plot residuals 
residuals_bsts_season_ar <- residuals(model_season_ar)
residuals_bsts_season_ar <- rowMeans(model_season_ar$one.step.prediction.errors, na.rm = TRUE)
residuals_bsts_season_ar <- as.numeric(residuals(model_season_ar))
plot(residuals_bsts_season_ar, main = "BSTS Residuals", ylab = "Residuals", xlab = "Time")
acf(residuals_bsts_season_ar[1:5000])


# Forecast the next 59 days (Jan 2025 - Feb 2025)
burn_season_ar <- SuggestBurn(0.1, model_season_ar)
pred_season_ar <- predict(model_season_ar, burn = burn_season_ar, horizon = 59)


# Accuracy metrics


# Reverse seasonal differencing
forecast_values_season_ar <- (aqi_train_ts[(length(aqi_train_ts) - 365 + 1):(length(aqi_train_ts) - 365 + 59)]) + pred_season_ar$mean


rmse <- sqrt(mean((actual_values - forecast_values_season_ar)^2))
mae  <- mean(abs(actual_values - forecast_values_season_ar))
mape <- mean(abs((actual_values - forecast_values_season_ar) / actual_values)) * 100

cat("RMSE for Seasonal Differencing w/ AR(5):", rmse, "\n")
cat("MAE for Seasonal Differencing w/ AR(5): ", mae, "\n")
cat("MAPE for Seasonal Differencing w/ AR(5):", mape, "%\n")

"
Among the six models evaluated, the BSTS with seasonal differencing with AR(5)
component was the best-performing model. This specific data transformation met
the assumption of the BSTS model, ensuring that the residuals were uncorrelated.
It also achieved the lowest error metrics, although the metrics were still
relatively high, indicating that the model does not accurately capture the
underlying patterns in the data.
"
#####

# note: John's LSTM model is in the LSTM jupyter notebooks. The LSTM model
# performed the best out of all the models we constructed.
