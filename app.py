import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Title
st.title("🧱 Fly Ash Concrete Strength Estimator")
st.markdown("""
*Civil Engineering Final Year Project*  
By Vamshi  
Predict the compressive strength of concrete with Fly Ash replacement.
""")

# Training data (from your experiment)
fly_ash_percent = np.array([5, 10, 15, 20, 25]).reshape(-1, 1)
strength_data = np.array([18, 16, 15, 13, 12])

# Polynomial regression model (degree 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(fly_ash_percent)
model = LinearRegression()
model.fit(X_poly, strength_data)

# User inputs
fly_input = st.slider("🔘 Fly Ash Replacement (%)", 0, 30, step=1, value=10)
wcr_input = st.number_input("💧 Water-Cement Ratio", min_value=0.10, max_value=0.60, step=0.01, value=0.20)

# Prediction
input_data = poly.transform([[fly_input]])
predicted_strength = model.predict(input_data)[0]

st.success(f"✅ Predicted Compressive Strength: *{predicted_strength:.2f} MPa*")

# Graph (Optional visual)
st.subheader("📈 Strength vs Fly Ash Replacement")
fly_range = np.linspace(0, 30, 100).reshape(-1, 1)
predicted_values = model.predict(poly.transform(fly_range))

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
sns.scatterplot(x=fly_ash_percent.flatten(), y=strength_data, color='red', label='Actual Data')
sns.lineplot(x=fly_range.flatten(), y=predicted_values, color='blue', label='Prediction Curve')
plt.xlabel("Fly Ash %")
plt.ylabel("Compressive Strength (MPa)")
plt.title("Fly Ash % vs Strength")
plt.grid(True)
plt.legend()
st.pyplot(fig)
