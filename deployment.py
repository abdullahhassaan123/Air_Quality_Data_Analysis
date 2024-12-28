import base64
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load data
df = pd.read_csv("data_utf8.csv")    

# Convert 'date' column to datetime and extract month, day, and year
df["month"] = pd.to_datetime(df["date"]).dt.strftime('%b')  # E.g., Jan, Feb
df["day"] = pd.to_datetime(df["date"]).dt.day               # Day as a number (1-31)
df["year"] = pd.to_datetime(df["date"]).dt.year             # Year as a four-digit number


# Save the updated DataFrame back to the same file
df.to_csv("new_data.csv", index=False)

# imputation

#first remove unnecessary columns
df = df.drop(['stn_code', 'sampling_date', 'date', 'location_monitoring_station', 'pm2_5', 'agency'], axis=1)
# axis=1: Indicates that you're dropping columns (not rows)
print(df.head())
print(df.isnull().sum())

# Remove rows where all three columns 'month', 'day', and 'year' are missing
df = df.dropna(subset=['month', 'day', 'year'], how='all')
print(df.isnull().sum())

# Impute 'rspm' based on the mean of 'rspm' for each month
month_means = df.groupby('month')['rspm'].transform('mean')
df['rspm'] = df['rspm'].fillna(month_means)

month_means1 = df.groupby('month')['spm'].transform('mean')
df['spm'] = df['spm'].fillna(month_means1)

month_means2 = df.groupby('month')['so2'].transform('mean')
df['so2'] = df['so2'].fillna(month_means2)

month_means3 = df.groupby('month')['no2'].transform('mean')  
df['no2'] = df['no2'].fillna(month_means3)

# for locations missing:
# Calculate the mode of 'location' for each 'state'
state_mode = df.groupby('state')['location'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

#  Impute missing 'location' values using the mode for the corresponding 'state'
df['location'] = df.apply(
    lambda row: state_mode[row['state']] if pd.isnull(row['location']) else row['location'], axis=1
)
# for type missing:
#  Calculate the mode of 'type' for each 'state'
location_mode = df.groupby('state')['type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

# Impute missing 'type' values using the mode for the corresponding 'state'
df['type'] = df.apply(
    lambda row: location_mode[row['state']] if pd.isnull(row['type']) else row['type'], axis=1   
)

print(df.isnull().sum())



def classify_aqi(row):  # rspm and spm decide the AQI index , can be equated with PM10 or PM2.5  
    rspm = row['rspm']     #Respirable suspended particulate matter also known as PM10
    if (rspm <= 50) :
        return 'Good'
    elif (rspm <= 100) :
        return 'Moderate'
    elif (rspm <= 250 ) :
        return 'Poor'
    elif (rspm >= 250) :  
        return 'Hazardous'
    else:
        return 'Unrecognized'
    
    
# Function to return colored text for AQI category
def get_colored_aqi_category(category):
    color_map = {
        'Good': 'green',
        'Moderate': 'blue',
        'Poor': 'red',
        'Hazardous': 'darkred',
    }
    color = color_map.get(category, 'black')  # Default to black if category is unrecognized
    return f"<span style='color:{color}; font-weight:bold;'>{category}</span>"   



# Streamlit app

st.set_page_config(layout="wide")

# Custom CSS for crystal-like pattern with very light yellowish and golden tones
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(45deg, rgba(255, 223, 100, 0.1) 25%, transparent 25%) -50px 0,
                        linear-gradient(-45deg, rgba(255, 223, 100, 0.1) 25%, transparent 25%) -50px 0,
                        linear-gradient(45deg, transparent 75%, rgba(255, 223, 100, 0.1) 75%) 0 0,
                        linear-gradient(-45deg, transparent 75%, rgba(255, 223, 100, 0.1) 75%) 0 0;
            background-size: 80px 80px;
            background-attachment: fixed;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Indian Season-Based Air Quality Prediction")  
st.header("Concentration Levels")      

# Sidebar inputs

with st.sidebar:
    #adding color
    st.markdown(
        """
        <style>
        .stApp
        [data-testid="stSidebar"]{
            background-color: #00008B; /* Dark Blue background color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.subheader("Select Inputs")
    state = st.selectbox("Select State", options=df['state'].unique())
    location = st.selectbox("Select Location", options=df[df['state'] == state]['location'].unique())  
    type = st.selectbox("Select Area Type", options=df['type'].unique())    
    month = st.selectbox("Select Month", options=df['month'].unique())
    day = st.selectbox("Select Date", options=df['day'].unique())
   

# Calculate the means for non-user given inputs with the given combination of month, location, and type
so2_mean = df[
    (df["month"] == month) &
    (df["location"] == location) &
    (df["type"] == type)
]["so2"].mean()

no2_mean = df[
    (df["month"] == month) &
    (df["location"] == location) &
    (df["type"] == type)
]["no2"].mean()


spm_mean = df[
    (df["month"] == month) &
    (df["location"] == location) &
    (df["type"] == type)
]["spm"].mean()

categorical_columns = ['state', 'location', 'type', 'month', 'day' ]
label_encoders = {col: LabelEncoder() for col in categorical_columns}

for col, encoder in label_encoders.items():
    df[col] = encoder.fit_transform(df[col])
print(df.head())  


# Train-test split for two target variables: SPM and RSPM
X = df[[ 'state', 'location', 'type', 'month', 'day', 'so2', 'no2', 'spm']]
y_rspm = df['rspm']



# Split data
X_train_rspm, X_test_rspm, y_train_rspm, y_test_rspm = train_test_split(X, y_rspm, test_size=0.3, random_state=42)


# Train models
model_rspm = LinearRegression()
model_rspm.fit(X_train_rspm, y_train_rspm)



st.write("After entering your inputs on the left sidebar, please press the button below and wait a while, so we proceed the expected results...")
   
  

# Button for prediction
if st.button("Predict Concentration Levels"):
    # Prepare input data
    input_data = pd.DataFrame({
    
        "state": [state],
        "location": [location],
        "type": [type],  
        "month": [month],
        "day": [day],
        "so2": [so2_mean], 
        "no2": [no2_mean],
        "spm": [spm_mean] 
       
    })

    
    # Apply label encoding
    for col, encoder in label_encoders.items():
        input_data[col] = encoder.transform(input_data[col])
    
    # Predict
    rspm_prediction = model_rspm.predict(input_data)[0]

    
    st.subheader(f"> Predicted RSPM: {rspm_prediction:.2f} μg/m³")

 
    # Determine AQI category using classify_aqi function
    aqi_category = classify_aqi({'rspm': rspm_prediction})
    
    # Display AQI category with color
    colored_category = get_colored_aqi_category(aqi_category)
    st.markdown(f"<h2>Air Quality Category: {colored_category} </h2>",unsafe_allow_html=True)     