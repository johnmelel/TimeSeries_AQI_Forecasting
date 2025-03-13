# Time Series Final Project: Forecasting London's AQI

# Group: Anusha Bhat, Sarah Chaker, Hritik Jhaveri, John Melel
# March 13, 2025



########## Libraries ##########
library(readxl)
library(tseries)
library(forecast)
library(hts)
library(zoo)


########## Data Cleaning ##########

# Load the data 
aqi = read_excel("London_AQI_Daily.xlsx")
head(aqi)

# Check the structure of the data
summary(data$USAQI)

# Check outliers 
hist(data$USAQI,
     xlab = "USAQI",
     main = "Histogram of USAQI",
     breaks = sqrt(length(data$USAQI)) # set number of bins
)

qqnorm(data$USAQI, main = "QQ Plot of USAQI")
qqline(data$USAQI, col = "blue")

# There are a few large outliers so data transformation may be necessary

# Correct date structure 
aqi$Date = as.Date(aqi$Date, format = "%Y-%m-%d")
aqi$USAQI = as.integer(aqi$USAQI)

# Convert to time series 
aqi_ts = ts(aqi$USAQI, frequency = 365, start = c(2013, 1, 2))

########## EDA ##########

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
new_aqi = diff(aqi_ts)
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

























