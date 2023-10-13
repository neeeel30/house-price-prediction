# import joblib
# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler

# # Load the trained model and scaler
# loaded_model = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-5\my_model.pkl')
# # scaler = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-5\scaler.pkl')

# # Set a custom background image
# st.markdown(
#     """
#     <style>
#     .main {
#         background-image: url(https://png.pngtree.com/thumb_back/fh260/background/20230720/pngtree-sleek-and-modern-3d-black-luxury-house-image_3693364.jpg);  /* Replace with your image URL */
#         background-size: cover;
#         opacity: 0.7
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#         color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Page title and description
# st.title("House Price Prediction App")
# st.write("Predict house prices using a trained machine learning model.")

# # Sidebar
# st.sidebar.header("User Input")
# features = {}

# # Create input fields for each feature
# feature_names = [
#     'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
#        'sqft_living15', 'sqft_lot15', 'month', 'year'
# ]

# for feature_name in feature_names:
#     feature_value = st.sidebar.number_input(f"{feature_name.capitalize()}")
#     features[feature_name] = feature_value

# # Predict button
# if st.sidebar.button("Predict"):
#     # Prepare the input data for the model
#     input_data = [features[feature_name] for feature_name in feature_names]
#     # scaled_input_data = scaler.transform([input_data])

#     # Make predictions using the loaded model
#     prediction = loaded_model.predict(input_data)

#     # Display the prediction
#     st.subheader("Prediction")
#     st.write(f'Predicted Price: ${prediction[0][0]:}')

# # Display dataset (optional)
# if st.sidebar.checkbox("Show Dataset"):
#     df = pd.read_csv(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-5\kc_house_data.csv')
#     st.subheader("Dataset")
#     st.write(df)

# # Main content
# st.header("Model Training and Evaluation")
# # Add visualizations and model evaluation results here as needed
# # You can include plots, performance metrics, etc.

# # Footer
# st.sidebar.markdown("### About")
# st.sidebar.text("This app is for educational and demonstration purposes. The model may not be suitable for real-world pricing decisions.")


import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
loaded_model = joblib.load(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-5\my_model.pkl')

st.markdown(
    """
    <style>
    .main {
        background-image: url(https://png.pngtree.com/thumb_back/fh260/background/20230720/pngtree-sleek-and-modern-3d-black-luxury-house-image_3693364.jpg);  /* Replace with your image URL */
        background-size: cover;
        opacity: 0.7
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page title and description
st.title("House Price Prediction App")
st.write("Predict house prices using a trained machine learning model.")

# Sidebar for user input
st.sidebar.header("User Input")
features = {}

# Define input fields for each feature
feature_names = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
    'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
    'sqft_living15', 'sqft_lot15', 'month', 'year'
]

for feature_name in feature_names:
    feature_value = st.sidebar.number_input(f"{feature_name.capitalize()}")
    features[feature_name] = feature_value

# Predict button
if st.sidebar.button("Predict"):
    # Prepare the input data for the model
    input_data = [features[feature_name] for feature_name in feature_names]

    # Make predictions using the loaded model
    prediction = loaded_model.predict([input_data])#[0][0]

    # Display the prediction
    st.subheader("Prediction")
    st.write(f'Predicted Price: ${prediction:}')

# Display dataset (optional)
if st.sidebar.checkbox("Show Dataset"):
    df = pd.read_csv(r'C:\Users\NEEL\Desktop\Coding\Anaconda_Data_Science\Projects\Project-5\kc_house_data.csv')
    st.subheader("Dataset")
    st.dataframe(df)

# Main content
st.header("Model Training and Evaluation")
st.write("Add visualizations and model evaluation results here as needed.")

# Footer
st.sidebar.markdown("### About")
st.sidebar.text("This app is for educational and demonstration purposes. The model may not be suitable for real-world pricing decisions.")

# Run the app
if __name__ == '__main__':
    st.sidebar.markdown("This app is for educational and demonstration purposes. The model may not be suitable for real-world pricing decisions.")
