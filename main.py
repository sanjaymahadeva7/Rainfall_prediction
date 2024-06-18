import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model_rf = joblib.load('random_forest_model.pkl')

# Define the input features
features = ['lat', 'lon', 'omega_x', 'omega_y', 'omega', 'pr_wtr', 'rhum_x', 'rhum_y', 'rhum', 'slp', 
            'tmp_x', 'tmp_y', 'tmp', 'uwnd_x', 'uwnd_y', 'uwnd', 'vwnd_x', 'vwnd_y', 'vwnd']

# Create the Streamlit app
st.title("Rain Prediction")


# Add a button to show feature details
if st.button("Show Feature Details"):
    st.subheader("Feature Details and Their Relationship with the Output")
    
    # Display feature information
    feature_info = {
        'lat': 'Latitude of the location. Important for understanding the regional climate patterns which affect rainfall.',
        'lon': 'Longitude of the location. Helps in identifying the geographical climate variations.',
        'omega_x': 'Zonal component of the vertical velocity. Indicates the upward or downward movement of air which can affect precipitation.',
        'omega_y': 'Meridional component of the vertical velocity. Similar to omega_x, but in the north-south direction.',
        'omega': 'Vertical velocity. Critical for understanding the movement of air masses and cloud formation, directly impacting rain.',
        'pr_wtr': 'Precipitable water. Measures the total atmospheric water vapor which is a key ingredient for rain.',
        'rhum_x': 'Zonal component of relative humidity. Indicates the moisture content in the air, which affects rain formation.',
        'rhum_y': 'Meridional component of relative humidity. Similar to rhum_x, but in the north-south direction.',
        'rhum': 'Relative humidity. Directly affects the likelihood of precipitation, as higher humidity can lead to cloud and rain formation.',
        'slp': 'Sea level pressure. Influences weather patterns and storm systems which can result in rain.',
        'tmp_x': 'Zonal component of temperature. Temperature variations can impact the formation of clouds and rain.',
        'tmp_y': 'Meridional component of temperature. Similar to tmp_x, but in the north-south direction.',
        'tmp': 'Temperature. Affects the evaporation and condensation processes that are crucial for rain formation.',
        'uwnd_x': 'Zonal component of the wind speed. Wind patterns affect moisture transport and cloud formation.',
        'uwnd_y': 'Meridional component of the wind speed. Similar to uwnd_x, but in the north-south direction.',
        'uwnd': 'Wind speed. Influences weather systems and the movement of air masses that bring rain.',
        'vwnd_x': 'Zonal component of the wind speed. Similar to uwnd_x, providing more detail on wind direction.',
        'vwnd_y': 'Meridional component of the wind speed. Similar to uwnd_y, providing more detail on wind direction.',
        'vwnd': 'Wind speed. Important for understanding the dynamics of the atmosphere which influence rain.'
    }
    
    for feature, description in feature_info.items():
        st.write(f"**{feature}**: {description}")

# Add a button to explain how the model predicts rain
if st.button("Explain Rain Prediction"):
    st.subheader("How the Model Predicts Rain Using All Features")

    explanation = """
    The Random Forest model predicts rain by analyzing various atmospheric and geographical features. Here's a general explanation of how it works:

    1. **Geographical Features**: Latitude and longitude provide the model with the location's position relative to weather systems and prevailing wind patterns. This information helps in understanding regional climate patterns that influence rain formation.

    2. **Atmospheric Variables**: Vertical velocities (omega_x, omega_y, omega) indicate the movement of air masses, which is crucial for cloud formation and precipitation. Precipitable water (pr_wtr) measures atmospheric moisture, essential for determining the potential for rain.

    3. **Temperature and Humidity**: Relative humidity (rhum_x, rhum_y, rhum) and temperature (tmp_x, tmp_y, tmp) influence the air's ability to hold moisture. Higher humidity and warmer temperatures increase the likelihood of cloud formation and rain.

    4. **Pressure and Wind Patterns**: Sea level pressure (slp) affects atmospheric stability and the likelihood of storm systems, influencing rain patterns. Wind speed and direction (uwnd_x, uwnd_y, uwnd, vwnd_x, vwnd_y, vwnd) impact moisture transport and cloud dynamics, which in turn affect rain intensity and distribution.

    5. **Modeling Process**: The Random Forest algorithm uses multiple decision trees to analyze these features collectively. Each tree evaluates different combinations of features to predict rain, and the final prediction is an aggregate of the predictions from all trees.

    By integrating these variables, the model provides a comprehensive assessment of the atmospheric conditions that contribute to rain prediction.
    """
    
    st.write(explanation)
    

# Create input fields for each feature
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Predict the rain amount using the model
if st.button("Predict"):
    # Ensure input data matches model input format
    input_array = input_df.values
    prediction = model_rf.predict(input_array)
    st.write(f"Predicted rain amount: {prediction[0]}")

# Display the input data for verification
st.subheader("Input Data")
st.write(input_df)

