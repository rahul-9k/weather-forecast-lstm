import streamlit as st
import matplotlib.pyplot as plt

def render_forecast_plot(forecast_df):
    st.subheader("📈 Forecast Visualization")

    fig, axs = plt.subplots(len(forecast_df.columns), 1, figsize=(10, 3 * len(forecast_df.columns)), sharex=True)

    if len(forecast_df.columns) == 1:
        axs = [axs]

    for i, feature in enumerate(forecast_df.columns):
        axs[i].plot(forecast_df.index, forecast_df[feature], label="Forecast", color="royalblue", marker='o')
        axs[i].set_ylabel(feature)
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Date")
    fig.tight_layout()
    st.pyplot(fig)


