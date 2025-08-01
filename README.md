# Delhi Weather Forecast Using LSTM Encoder-Decoder

![Weather Forecast Banner](https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&q=80)

## 🚀 Project Overview

This project implements a **deep learning weather forecasting system** for Delhi using an **LSTM encoder-decoder model**. Leveraging historical weather data from 2000 to 2024, the model predicts key weather parameters for the upcoming 7 days, such as:

- Minimum Temperature (tmin)
- Maximum Temperature (tmax)
- Average Temperature (tavg)
- Precipitation (prcp)
- Wind Speed (wspd)

The solution offers both command-line scripts for training, evaluation, and prediction, as well as a user-friendly **interactive Streamlit app** for visualization and easy forecasting.

---

## 📊 Features

- **Data preprocessing and time series windowing:** Efficiently creates input-output sequences for sequence-to-sequence forecasting.
- **Encoder-decoder LSTM architecture:** Captures temporal dependencies in weather patterns.
- **Model evaluation:** Provides overall and per-feature MAE and RMSE metrics.
- **Inverse scaling:** Transforms model outputs back to original units for meaningful interpretation.
- **Streamlit web app:** Interactive forecast visualization with:
  - Automatic prediction for next 7 days from current date
  - Overlay comparison with same month of previous year
  - Downloadable CSV export of forecast data

---

## 🛠️ Tech Stack

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Streamlit

---

## ⚙️ Setup & Usage
1. Clone the repo
```bash
git clone https://github.com/yourusername/delhi-weather-lstm.git
cd delhi-weather-lstm
```
2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```bash
Copy
Edit
pip install -r requirements.txt
```
4. Prepare data & window sequences
```bash
Copy
Edit
python run_windowing.py
```
5. Train the model
```bash
Copy
Edit
python run_training.py
```
6. Evaluate model performance
```bash
Copy
Edit
python evaluate.py
```
7. Run predictions
```bash
Copy
Edit
python predict.py
```
8. Launch Streamlit app
```bash
Copy
Edit
streamlit run app.py
Open http://localhost:8501 in your browser to use the app.
```

## 🔍 How It Works
The run_windowing.py script loads and preprocesses the cleaned weather data to generate input-output sequences.

run_training.py trains an LSTM encoder-decoder model on the prepared sequences.

The trained model weights and scaler are saved for later use.

evaluate.py assesses model accuracy using MAE and RMSE metrics, both overall and feature-wise.

predict.py loads the model and scaler, generates forecasts starting from the current date, and outputs a DataFrame.

app.py provides an interactive Streamlit dashboard to visualize forecasts and compare with last year's same month.

## 📈 Model Performance
Overall MAE: e.g., 2.5 degrees / units

Overall RMSE: e.g., 5.4 degrees / units

Feature-specific metrics are displayed in the evaluation report.

## ✨ Future Improvements
Incorporate more advanced architectures like Transformers for improved accuracy.

Add real-time weather API integration for live updates.

Deploy the Streamlit app on cloud platforms for easy access.

Expand to multi-city forecasting using transfer learning.

## 📞 Contact
Developed by Rahul Pandey
Email: rahulpandey@example.com
GitHub: https://github.com/yourusername


Feel free to star ⭐ the repo if you find it helpful!

Happy Forecasting! ☀️🌧️🌬️

