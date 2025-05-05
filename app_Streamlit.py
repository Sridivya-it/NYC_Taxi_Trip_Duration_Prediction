import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(layout="wide")

# Load models
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

models = {
    "Linear Regression": load_model("linear_reg.pkl"),
    "Decision Tree": load_model("decision_tree.pkl"),
    "Random Forest": load_model("random_forest.pkl"),
    "LightGBM": load_model("light_GBM.pkl")
}

# Load data
@st.cache_data
def load_data():
    file_path = "train_updated.csv"  
    df = pd.read_csv(file_path)
    df.insert(0, "index", df.index)  # Add index column
    return df

df = load_data()
df = df.dropna()

# Define feature columns and target variable
feature_columns = [
    'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude', 'trip_distance',
                   'trip_category_encoded', 'pickup_hour', 'pickup_day', 'pickup_month',
                   'pickup_weekday', 'trip_duration_hours', 'speed', 'hourly_avg_duration'
]

target_column = "trip_duration"

df = df[["index"] + feature_columns + [target_column]]  # Ensure correct column order

# Streamlit app
st.title("NYC Taxi Trip Duration Analysis")

# Sidebar
st.sidebar.header("Team")
st.sidebar.write("Sridivya Gorantla")
st.sidebar.write("Bhargavi")
st.sidebar.write("Veera Reddy")

st.sidebar.header("About")
st.sidebar.write("This app predicts NYC taxi trip duration using various machine learning models.")

# User choice: Enter dynamic values or use index-based input
input_option = st.radio("Choose Input Method", ("Use Index", "Enter Values Manually"))

# Input section
if input_option == "Use Index":
    st.subheader("Select Row by Index")
    selected_index = st.number_input("Enter Row Index", min_value=int(df["index"].min()), max_value=int(df["index"].max()), step=1)
    selected_row = df[df["index"] == selected_index]
    st.write(selected_row)

    st.subheader("Filters")
    vendor = st.selectbox("Select Vendor", df["vendor_id"].unique())
    passenger_count = st.slider("Passenger Count", int(df["passenger_count"].min()), int(df["passenger_count"].max()), (1, 4))
    pickup_hour = st.slider("Pickup Hour", 0, 23, (0, 23))

    # Apply filters
    filtered_df = df[(df["vendor_id"] == vendor) &
                 (df["passenger_count"].between(passenger_count[0], passenger_count[1])) &
                 (df["pickup_hour"].between(pickup_hour[0], pickup_hour[1]))]
    input_df = df[df["index"] == selected_index].copy()
    if input_df.empty:
        st.warning("Invalid index selected. Please choose a valid index.")
    else:
        st.write(input_df)

else:
    st.subheader("Enter Feature Values Manually")

    input_values = {}
    for feature in feature_columns:
        if feature == "vendor_id":
            input_values[feature] = st.number_input(f"Enter {feature}", value=int(df[feature].mode()[0]))
        elif df[feature].dtype in ["int64", "float64"]:
            input_values[feature] = st.number_input(f"Enter {feature}", value=float(df[feature].mean()))
        else:
            input_values[feature] = st.text_input(f"Enter {feature}", value=str(df[feature].mode()[0]))

    # Convert input values to DataFrame
    input_df = pd.DataFrame([input_values])

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Model prediction
if st.button("Predict"):
    if input_df.empty:
        st.error("Please enter valid input data.")
    else:
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(input_df[feature_columns])[0]

        st.subheader("Predicted Trip Durations")
        for model_name, pred in predictions.items():
            st.write(f"{model_name}: {pred:.2f} minutes")