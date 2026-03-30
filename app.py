import streamlit as st
import pandas as pd
import pickle
import time
import requests
import plotly.express as px
from streamlit_lottie import st_lottie

# 1. Page Configuration (Must be the very first Streamlit command)
st.set_page_config(page_title="Agri-Smart AI", page_icon="🌾", layout="wide")

# --- CUSTOM CSS FOR ANIMATIONS ---
st.markdown("""
<style>
    @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }
    .stApp { animation: fadeIn 1.5s ease-in; }
    div.stButton > button:first-child { transition: all 0.3s ease-in-out; }
    div.stButton > button:first-child:hover { transform: scale(1.02); box-shadow: 0px 4px 15px rgba(0, 200, 0, 0.4); }
    .title-glow { font-size: 40px; font-weight: bold; background: linear-gradient(90deg, #4CAF50, #2E7D32); -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: pulse 2s infinite alternate; }
    @keyframes pulse { 0% { text-shadow: 0px 0px 5px rgba(76, 175, 80, 0.2); } 100% { text-shadow: 0px 0px 15px rgba(76, 175, 80, 0.8); } }
</style>
""", unsafe_allow_html=True)

# Helper function to load the Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load the saved model and raw dataset
with open('notebooks/model/crop_model.pkl', 'rb') as file:
    model = pickle.load(file)
df = pd.read_csv('data/Crop_recommendation.csv') 

# --- HEADER SECTION ---
st.markdown('<p class="title-glow">🌾 Agri-Smart: Advanced Crop Recommendation AI</p>', unsafe_allow_html=True)

lottie_farming = load_lottieurl("https://lottie.host/7e0a8b9a-7c9c-4610-8b01-38cb44131df3/R5f94u6H0A.json")
left_col, right_col = st.columns([2, 1])
with left_col:
    st.write("Welcome to the predictive agriculture dashboard. This system uses a **Random Forest Ensemble Model** to analyze chemical soil makeup and local weather patterns to recommend the optimal crop for maximizing yield.")
with right_col:
    if lottie_farming:
        st_lottie(lottie_farming, height=120, key="farming")
st.divider()

# --- CREATE TABS ---
tab1, tab2 , tab3= st.tabs(["🔮 AI Prediction Engine", "📊 Data Analytics & Insights","🧪 Model Diagnostics"])

# ==========================================
# TAB 1: THE PREDICTION ENGINE
# ==========================================
with tab1:
    st.sidebar.header("🌍 Live Weather Integration")
    st.sidebar.write("Fetch real-time weather data for your farm's location.")
    
    # 1. City Input & API Call
    city = st.sidebar.text_input("Enter City Name (e.g., Punjab, Delhi)")
    if st.sidebar.button("📡 Fetch Live Weather"):
        api_key = "fe3e681762b99e28e3e5f89093dc474d"  
        
        if city:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.live_temp = data['main']['temp']
                    st.session_state.live_humidity = data['main']['humidity']
                    st.sidebar.success(f"✅ Live Weather for {city.title()}: {st.session_state.live_temp}°C | {st.session_state.live_humidity}%")
                else:
                    st.sidebar.error("❌ City not found. Please check the spelling.")
            except Exception as e:
                st.sidebar.error("❌ Connection error.")
        else:
            st.sidebar.warning("⚠️ Please enter a city name first.")

    st.sidebar.divider()
    
    # 2. Manual/Automatic Sliders
    st.sidebar.header("🧪 Soil & Weather Parameters")
    N = st.sidebar.slider("Nitrogen (N)", 0, 150, 90)
    P = st.sidebar.slider("Phosphorus (P)", 0, 150, 42)
    K = st.sidebar.slider("Potassium (K)", 0, 205, 43)
    ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 202.9)
    
    # These sliders automatically update if weather is fetched!
    default_temp = st.session_state.get('live_temp', 20.8)
    default_humidity = st.session_state.get('live_humidity', 82.0)
    temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, float(default_temp))
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(default_humidity))

    # 3. Main Dashboard Display
    st.subheader("Current Environmental Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nitrogen", f"{N} mg/kg")
    col2.metric("Phosphorus", f"{P} mg/kg")
    col3.metric("Potassium", f"{K} mg/kg")
    col4.metric("pH Level", f"{ph}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Temperature", f"{temp} °C")
    col6.metric("Humidity", f"{humidity} %")
    col7.metric("Rainfall", f"{rainfall} mm")
    st.write("") 

    # 4. The Prediction Engine Button
    if st.button("🧠 Run AI Analysis", type="primary", use_container_width=True):
        progress_text = "Analyzing soil chemistry and weather patterns..."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.5)
        my_bar.empty()
        
        # Format input for the ML model
        input_data = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]], 
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Make predictions
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_names = model.classes_
        
        # Format the probabilities for the chart
        prob_df = pd.DataFrame({'Crop': class_names, 'Probability': probabilities})
        prob_df = prob_df.sort_values(by='Probability', ascending=False).head(3)
        prob_df['Probability (%)'] = prob_df['Probability'] * 100
        
        # Celebration and Output
        st.balloons()
        st.success(f"### 🏆 Primary Recommendation: **{prediction.upper()}**")
        st.markdown("#### Model Confidence Breakdown")
        
        # Plotly Interactive Bar Chart
        fig_bar = px.bar(
            prob_df, x='Probability (%)', y='Crop', color='Crop',
            orientation='h', text_auto='.2f'
        )
        fig_bar.update_layout(showlegend=False, xaxis_title="Probability (%)", yaxis_title="")
        st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# TAB 2: DATA ANALYTICS
# ==========================================
with tab2:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.write("This section provides a transparent view into the dataset used to train our Random Forest model.")
    
    st.subheader("1. Dataset Overview")
    st.dataframe(df, use_container_width=True, height=200)
    
    st.subheader("2. Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("3. Interactive Nutrient Distribution (Nitrogen vs Phosphorus)")
    st.write("Hover over the dots to see exact values. Use the legend on the right to isolate specific crops.")
    
    # Plotly Interactive Scatter Plot
    fig_scatter = px.scatter(
        df, x='N', y='P', color='label',
        labels={'N': 'Nitrogen (N) mg/kg', 'P': 'Phosphorus (P) mg/kg', 'label': 'Crop Type'},
        hover_data=['K', 'temperature', 'ph'],
        height=600
    )
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.8))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ==========================================
# TAB 3: MODEL DIAGNOSTICS
# ==========================================
with tab3:
    st.header("🧪 Model Performance Metrics")
    st.write("This tab proves the mathematical reliability of the Random Forest model using standard Data Science evaluation techniques.")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np

    # 1. Prepare data for a quick evaluation
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Generate Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    st.subheader("1. Confusion Matrix (Heatmap)")
    st.write("This matrix shows how many times the AI correctly predicted each crop vs. how many times it was wrong.")
    
    # Plotly Heatmap for Confusion Matrix
    labels = sorted(y.unique())
    fig_cm = px.imshow(
        cm, 
        text_auto=True, 
        aspect="auto",
        x=labels, 
        y=labels, 
        labels=dict(x="Predicted Crop", y="Actual Crop", color="Count"),
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # 3. Classification Report
    st.subheader("2. Detailed Accuracy Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Show the main metrics (Precision, Recall, F1)
    st.dataframe(report_df.iloc[:-3, :3].style.background_gradient(cmap='Greens'), use_container_width=True)
    
    st.info("""
    **Quick Guide for the Viva:**
    * **Precision:** How many of the predicted 'Rice' crops were actually Rice?
    * **Recall:** Out of all the actual 'Rice' in the data, how many did the AI find?
    * **F1-Score:** The perfect balance between the two. The closer to 1.0, the better!
    """)