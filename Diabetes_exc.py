import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Diabetes Health Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better readability on both light and dark themes
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #64B5F6;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #90CAF9;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #1E3A5F;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 1.5rem;
        color: #E3F2FD;
    }
    .prediction-box {
        background-color: #1B4332;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1.5rem;
        color: #E8F5E9;
    }
    .risk-low { color: #81C784; font-weight: bold; }
    .risk-medium { color: #FFB74D; font-weight: bold; }
    .risk-high { color: #E57373; font-weight: bold; }
    .feature-card {
        background-color: #2D3748;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #E2E8F0;
        border: 1px solid #4A5568;
    }
    /* Improved contrast for all text - works on both light and dark themes */
    .stSlider label, .stRadio label, .stNumberInput label, 
    .stSelectbox label, .stTextInput label {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    /* Specific styling for cholesterol and blood sugar sections */
    .cholesterol-section, .blood-sugar-section {
        background-color: #2D3748;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #4A5568;
        color: #E2E8F0;
    }
    .cholesterol-section h3, .blood-sugar-section h3 {
        color: #90CAF9;
        margin-top: 0;
    }
    /* Ensure all text in the app is visible */
    .stApp {
        color: #E2E8F0;
    }
    /* Style for slider values */
    .stSlider > div > div > div {
        color: #E2E8F0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained model
@st.cache_data
def load_pretrained_model():
    try:
        with open('diabetes.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        if isinstance(loaded_data, tuple):
            if len(loaded_data) == 2:
                ridge_model, scaler = loaded_data
            else:
                ridge_model = loaded_data[0]
                scaler = StandardScaler()
        else:
            ridge_model = loaded_data
            scaler = StandardScaler()
        
        # Get feature names
        feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        return ridge_model, scaler, feature_names
        
    except FileNotFoundError:
        st.error("diabetes.pkl not found. Please make sure the model file is in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def scale_age(age):
    """Convert age from 0-100 to standardized scale"""
    age_mean = 48.5
    age_std = 13.1
    return (age - age_mean) / age_std

def encode_sex(sex_selection):
    """Convert sex selection to binary encoding"""
    return 1 if sex_selection.lower() == 'male' else 0

def scale_sex_for_model(sex_binary):
    """Convert binary sex to standardized scale for model input"""
    return 0.05 if sex_binary == 1 else -0.04

def predict_data(user_data, ridge_model, scaler, feature_names):
    try:
        user_array = np.array([[user_data[feature] for feature in feature_names]])
        user_df = pd.DataFrame(user_array, columns=feature_names)
        user_array_scaled = scaler.transform(user_df)
        prediction = ridge_model.predict(user_array_scaled)
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return [0.0]

def create_radar_chart(values, categories):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Values',
        line_color='blue'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
        showlegend=False,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0')
    )
    return fig

def create_feature_importance_chart(coefficients, feature_names):
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    importance_df['Abs_Importance'] = abs(importance_df['Importance'])
    importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance in Diabetes Prediction',
                 color='Importance', color_continuous_scale='RdBu_r')
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0')
    )
    return fig

def main():
    # Header section
    st.markdown('<h1 class="main-header">🩺 Diabetes Health Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Understand your diabetes progression risk")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>Welcome to your personal diabetes health assessment!</h3>
    <p>This tool helps you understand how various health factors contribute to diabetes progression. 
    <strong>Note:</strong> This is for educational purposes only and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    ridge_model, scaler, feature_names = load_pretrained_model()
    if ridge_model is None:
        return
    
    # Create input section
    st.markdown('<h2 class="sub-header">Your Health Profile</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_input = st.slider("**Age**", min_value=1, max_value=100, value=45, step=1,
                             help="Age is an important factor in diabetes risk")
        
        sex_selection = st.radio("**Biological Sex**", ['Female', 'Male'], 
                                help="Diabetes risk factors can vary by sex")
        
        bmi_normal = st.slider("**Body Mass Index (BMI)**", min_value=15.0, max_value=40.0, value=25.0, step=0.1,
                              help="Healthy range: 18.5-24.9")
        
        # Visual indicator for BMI
        if bmi_normal < 18.5:
            st.warning("Underweight (BMI < 18.5)")
        elif bmi_normal <= 24.9:
            st.success("Healthy weight (BMI 18.5-24.9)")
        elif bmi_normal <= 29.9:
            st.warning("Overweight (BMI 25-29.9)")
        else:
            st.error("Obese (BMI ≥ 30)")
        
        bp_normal = st.slider("**Blood Pressure (mmHg)**", min_value=80, max_value=180, value=120, step=1,
                             help="Normal blood pressure is around 120/80 mmHg")
        
        # Visual indicator for BP
        if bp_normal < 90:
            st.warning("Low blood pressure")
        elif bp_normal <= 120:
            st.success("Normal blood pressure")
        elif bp_normal <= 139:
            st.warning("Elevated blood pressure")
        else:
            st.error("High blood pressure (Hypertension)")
    
    with col2:
        # Cholesterol inputs with improved section styling
        st.markdown('<div class="cholesterol-section"><h3>Cholesterol Levels (mg/dL)</h3></div>', unsafe_allow_html=True)
        
        s1_normal = st.slider("**Total Cholesterol**", min_value=150, max_value=300, value=200, step=1,
                             help="Desirable: <200 mg/dL")
        
        s2_normal = st.slider("**LDL (Bad Cholesterol)**", min_value=70, max_value=200, value=100, step=1,
                             help="Optimal: <100 mg/dL")
        
        s3_normal = st.slider("**HDL (Good Cholesterol)**", min_value=20, max_value=100, value=50, step=1,
                             help="Better: >60 mg/dL")
        
        # Blood sugar inputs with improved section styling
        st.markdown('<div class="blood-sugar-section"><h3>Blood Sugar Levels</h3></div>', unsafe_allow_html=True)
        
        s5_normal = st.slider("**Fasting Blood Sugar (mg/dL)**", min_value=70, max_value=200, value=95, step=1,
                             help="Normal fasting glucose: 70-100 mg/dL")
        
        s6_normal = st.slider("**Post-Meal Blood Sugar (mg/dL)**", min_value=100, max_value=300, value=140, step=1,
                             help="Normal 2 hours after eating: <140 mg/dL")
        
        # Thyroid function
        s4_normal = st.slider("**Thyroid Stimulating Hormone (μIU/mL)**", min_value=0.1, max_value=10.0, value=2.5, step=0.1,
                             help="Normal range: 0.4-4.0 μIU/mL")
    
    # Process inputs
    age = scale_age(age_input)
    sex_binary = encode_sex(sex_selection)
    sex = scale_sex_for_model(sex_binary)
    
    # Convert values to standardized scale for the model
    bmi = (bmi_normal - 26.0) / 4.0
    bp = (bp_normal - 94.0) / 14.0
    s1 = (s1_normal - 190.0) / 36.0
    s2 = (s2_normal - 115.0) / 28.0
    s3 = (s3_normal - 50.0) / 13.0
    s4 = (s4_normal - 4.0) / 2.0
    s5 = (s5_normal - 100.0) / 15.0
    s6 = (s6_normal - 125.0) / 20.0
    
    # Create radar chart data
    radar_categories = ['BMI', 'BP', 'Total Chol', 'LDL', 'HDL', 'Thyroid', 'Fasting BS', 'Post-Meal BS']
    radar_values = [bmi, bp, s1, s2, s3, s4, s5, s6]
    
    # Prediction button
    if st.button("🔍 Analyze My Diabetes Risk", type="primary", use_container_width=True):
        user_data = {
            "age": age, "sex": sex, "bmi": bmi, "bp": bp,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6
        }
        
        try:
            prediction = predict_data(user_data, ridge_model, scaler, feature_names)
            prediction_value = round(prediction[0], 2)
            st.session_state.prediction = prediction_value
            st.session_state.user_data = user_data
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
    
    # Display prediction
    if 'prediction' in st.session_state:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### Your Diabetes Progression Prediction")
        
        if st.session_state.prediction < 100:
            st.markdown(f'<p class="risk-low">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
            st.markdown("**Lower diabetes progression risk**")
        elif st.session_state.prediction < 200:
            st.markdown(f'<p class="risk-medium">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
            st.markdown("**Moderate diabetes progression risk**")
        else:
            st.markdown(f'<p class="risk-high">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
            st.markdown("**Higher diabetes progression risk**")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed results and educational content
        tab1, tab2 = st.tabs(["📊 Your Results", "📚 Health Insights"])
        
        with tab1:
            st.markdown("### Detailed Health Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_radar_chart(radar_values, radar_categories), use_container_width=True)
                st.caption("Your health metrics compared to standardized ranges")
            
            with col2:
                if hasattr(ridge_model, 'coef_'):
                    st.plotly_chart(create_feature_importance_chart(ridge_model.coef_, feature_names), use_container_width=True)
                    st.caption("Which factors most influence diabetes progression")
        
        with tab2:
            st.markdown("### Understanding Diabetes Risk Factors")
            
            factors = [
                {"title": "Body Mass Index (BMI)", "content": "Maintaining a healthy weight is crucial for diabetes prevention.", "ideal": "18.5-24.9"},
                {"title": "Blood Pressure", "content": "High blood pressure often accompanies diabetes.", "ideal": "<120/80 mmHg"},
                {"title": "Cholesterol Levels", "content": "Diabetes affects cholesterol levels, increasing heart disease risk.", "ideal": "Total: <200 mg/dL, LDL: <100 mg/dL, HDL: >60 mg/dL"},
                {"title": "Blood Sugar Levels", "content": "Consistently high blood sugar levels can lead to diabetes diagnosis.", "ideal": "Fasting: 70-100 mg/dL, Post-meal: <140 mg/dL"}
            ]
            
            for factor in factors:
                with st.expander(factor["title"]):
                    st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
                    st.markdown(f"**Ideal range:** {factor['ideal']}")
                    st.markdown(f"**Why it matters:** {factor['content']}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #A0AEC0;">
    <p>This tool is for educational purposes only. It is not a substitute for professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
