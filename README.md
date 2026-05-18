# test-sARIMAx-Prediction-with-Dash
My Time series model: prediction of air passengers with the data on the internet

## My codes of Dash dropdown for predicting No. of the air passengers us sARIMAx algorithm

### Author
* written by K.H. Hwang  

### Concepts of predicting and analyzing time series
* The codes, only for conceptual frameworks of ARIMA, sARIMAx or SARIMAX modeling with dash incorporation, were written around August, 2025  
  referenced with Dash HTML Components homepage, sARIMAx library and SARIMAX guides in www.geeksforgeeks.org  
  plot Graph with sARIMAx time series data using Plotly
* My codes of Dash dropdown aims at predicting No. of the air passengers using sARIMAx algorithm
* The codes, only for conceptual frameworks of ARIMA, sARIMAx or sARIMAx modeling with dash incorporation, were written around the date of 20250817
* referenced with Dash HTML Components homepage, sARIMAx library and sARIMAx guides in www.geeksforgeeks.org
* also referenced with Birat Poudel, Three techniques to improve sARIMAx model for time series forecasting, www.Medium.com
* plot Graph with sARIMAx time series data using Plotly
* This algorithm needs to be tunned with the parameters for sARIMAx modeling

### Contributions
* The example codes are focused on more analytic results with measuring errors of prediction
* The codes are lighter in comparison to other heavy models using sARIMAx.
* The codes are a classical example to be updated with modern AI techniques.

---
### prediction and summary of sARIMAx modeling
* Tab-1: Showing a predicted trend

* Tab-2: Text results from model summary

---

### ✈️ Preparing Data to Predict Air Passenger Numbers (Beginner-Friendly)

To forecast future air passenger numbers using a time series model like SARIMAX, you need to set up your data properly. Here's how to do it step by step:

---

#### 🗓️ 1. Understand the Time Range

* **Last known data point**: December 1, 1960 (`1960-12-01`)
* **Prediction target**: Starting from January 2, 1970 (`1970-01-02`)
* **Prediction span**: From December 1960 to December 1972

---

#### 📄 2. Create the Input Table (DataFrame)

* Start with an empty table (called `df_rowsnull`) with N rows.
* Fill it with `None` values to represent missing future data.
* Add a column called `"date"` that contains the timeline (monthly dates).
* Set `"date"` as the index so the model knows the order of time.

---

#### 🔍 3. Prepare Training Data

* Split the full dataset into a smaller portion called `df_train`.
* Use data from **January 1949 to December 1962** to train the model.
* Run tests to check if the data is stable (stationary):
    - **ADF Test** (Augmented Dickey-Fuller): Checks for randomness or "white noise."
    - **KPSS Test**: Another method to check stability over time.

---

#### ⚙️ 4. Set Model Parameters

SARIMAX uses several settings to understand patterns in the data:

| Parameter                  | Meaning                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ |
| `order=(p,d,q)`            | Basic model settings: AR (p), differencing (d), MA (q)                         |
| `seasonal_order=(P,D,Q,s)` | Seasonal settings: AR (P), differencing (D), MA (Q), period (s)                |
| `trend`                    | Adds a time-based trend: `'c'` for constant, `'t'` for linear, `'ct'` for both |

**Examples**:

* `order=(1,1,1)` means: use 1 lag for AR, 1 differencing, and 1 lag for MA.
* `seasonal_order=(1,1,1,12)` means: monthly seasonality with similar settings.
* `trend='ct'` adds both a constant and a time-based slope.

---

#### 📚 5. Learn More

* Please refer various materials about SARIMAX or ARIMA modeling on internets
