# Weather Forecasting using RNN, GRU, and LSTM

This repository contains a Jupyter Notebook designed for time-series forecasting of weather data. The project compares three different Recurrent Neural Network (RNN) architectures—**Simple RNN, Gated Recurrent Unit (GRU), and Long Short-Term Memory (LSTM)**—across both univariate and multivariate settings.


The system uses proper chronological splitting to prevent data leakage and evaluates out-of-sample performance on a strict 2025–2026 forecast window.

---

## Dataset

- Source: Google Drive weather CSV
- Parsed using `pandas`
- Date column: `time`
- Targets:
  - `temperature_2m_mean (°C)`
  - `precipitation_sum (mm)`

### Evaluation Period
- **Train:** Data before `2025-01-04`
- **Test:** `2025-01-04` to `2026-01-04`

This ensures a realistic forward-looking forecast.

---

## Preprocessing

###  Cleaning
- Date parsing using `pd.to_datetime`
- Dropped rows with missing values
- Ensured numeric conversion for target variables

###  Feature Engineering

**Univariate Models**
- Temperature → temperature history only
- Precipitation → precipitation history only

**Multivariate Models**
- All numeric columns (excluding date)

---

###  Sliding Window Construction

- Window size: **14 days**
- Forecast horizon: **1 day ahead**


###  Scaling

Custom NumPy-based standardization:


* X_scaled = (X - X_mean)/standard_deviation


- Fitted only on training data
- Validation and test sets transformed using training statistics
- Target values inverse-transformed for final metric computation



###  Train/Validation Split

- 80% training
- 20% validation
- Strict chronological split


---
---

## 2. Model Architecture

A unified sequential regressor class was implemented:

```
SeqRegressor(cell_type, input_size, hidden_size=64)
```

### Recurrent Backbone
- `nn.RNN`
- `nn.GRU`
- `nn.LSTM`

### Prediction Head
```
Linear(64 → 64)
ReLU
Linear(64 → 1)
```

Only the final time-step hidden state is used for regression.

---

## 3. Training Configuration

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Batch size: 32
- Max epochs: 200
- Early stopping patience: 20
- GPU support enabled (if available)

---

## 4. Results

The models were evaluated using Mean Absolute Error (**MAE**) and Root Mean Squared Error (**RMSE**). 

### Temperature Prediction
| Setting | Model | MAE | RMSE |
| :--- | :--- | :--- | :--- |
| **Multivariate** | **GRU (Best)** | **0.3975** | **0.5221** |
| Multivariate | LSTM | 0.4033 | 0.5293 |
| Univariate | GRU | 0.4045 | 0.5268 |

### Precipitation Prediction
| Setting | Model | MAE | RMSE |
| :--- | :--- | :--- | :--- |
| **Multivariate** | **LSTM (Best)** | **6.7244** | **14.8696** |
| Multivariate | GRU | 6.6622 | 14.9878 |
| Univariate | LSTM | 7.2181 | 15.5280 |

---

## 5. Plots

The notebook generates the following visualizations for the top-performing models:

1.  **Training vs. Validation Loss:** A plot showing the MSE loss over 200 epochs (with early stopping). These curves demonstrate the convergence of the models.
2.  **True vs. Predicted Values:**  **Temperature:** Shows a high degree of correlation, with the model accurately tracking seasonal and daily fluctuations.
    * **Precipitation:** Visualizes the model's ability to identify rainfall events, though with higher variance compared to temperature.

---

## 6. Summary

* **Model Selection:** Gated architectures (**GRU and LSTM**) significantly outperform the Simple RNN in capturing long-term dependencies in weather data.
* **Multivariate Advantage:** Using multiple weather features (solar radiation, pressure, etc.) improves prediction accuracy compared to using the target variable alone.
* **Predictability:** Temperature is a highly stable time series that yields very low error rates. Conversely, precipitation is much more volatile and sparse, representing a significantly harder forecasting challenge.
* **Optimization:** Early stopping effectively prevents overfitting and proper chronological splitting ensures realistic generalization.

For each target, the best-performing model is automatically selected and visualized.

---

## Technical Highlights

- Strict chronological splitting (no leakage)
- Custom sliding window implementation
- Manual NumPy standardization
- Multi-model benchmarking framework
- Early stopping
- GPU-enabled training
- Proper inverse-scaling before metric computation
- Clean PyTorch architecture design

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
