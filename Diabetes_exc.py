import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Diabetes Health Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with better contrast on white background
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1565C0;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin-bottom: 1.5rem;
        color: #212121;
    }
    .info-box h3 {
        color: #0D47A1;
    }
    .info-box strong {
        color: #0D47A1;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1.5rem;
        color: #212121;
    }
    .risk-low {
        color: #2E7D32;
        font-weight: bold;
    }
    .risk-medium {
        color: #EF6C00;
        font-weight: bold;
    }
    .risk-high {
        color: #C62828;
        font-weight: bold;
    }
    .feature-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #212121;
        border: 1px solid #E0E0E0;
    }
    .stSlider label, .stRadio label, .stNumberInput label {
        color: #212121 !important;
    }
    .stTooltipIcon {
        color: #616161 !important;
    }
    .dataframe {
        color: #212121 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #E3F2FD;
        border-radius: 4px 4px 0px 0px;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #0D47A1;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0D47A1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained model
@st.cache_data
def load_pretrained_model():
    try:
        # Load your pre-trained diabetes model
        with open('diabetes.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Check if the loaded data is a tuple (model, scaler) or just the model
        if isinstance(loaded_data, tuple):
            # If it's a tuple, extract model and scaler
            if len(loaded_data) == 2:
                ridge_model, scaler = loaded_data
            else:
                # If tuple has different structure, extract just the model (first element)
                ridge_model = loaded_data[0]
                # Create a new scaler
                diabetes = load_diabetes()
                X, y = diabetes.data, diabetes.target
                scaler = StandardScaler()
                scaler.fit(X)
        else:
            # If it's just the model
            ridge_model = loaded_data
            # Create a new scaler
            diabetes = load_diabetes()
            X, y = diabetes.data, diabetes.target
            scaler = StandardScaler()
            scaler.fit(X)
        
        # Get feature names
        diabetes = load_diabetes()
        
        return ridge_model, scaler, diabetes.feature_names
        
    except FileNotFoundError:
        st.error("diabetes.pkl not found. Please make sure the model file is in the same directory as your app.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading pre-trained model: {str(e)}")
        return None, None, None

def scale_age(age):
    """Convert age from 0-100 to standardized scale"""
    try:
        # Diabetes dataset age statistics (approximate)
        age_mean = 48.5  # Mean age in original dataset
        age_std = 13.1   # Standard deviation in original dataset
        
        # Convert to standardized scale
        scaled_age = (age - age_mean) / age_std
        return scaled_age
    except Exception as e:
        st.error(f"Error scaling age: {str(e)}")
        return 0.0

def encode_sex(sex_selection):
    """Convert sex selection to binary encoding (0 for Female, 1 for Male)"""
    try:
        if sex_selection.lower() == 'male':
            return 1
        else:
            return 0
    except Exception as e:
        st.error(f"Error encoding sex: {str(e)}")
        return 0

def scale_sex_for_model(sex_binary):
    """Convert binary sex to standardized scale for model input"""
    try:
        # In diabetes dataset, sex is already scaled around 0
        # Male is typically positive, Female is typically negative
        if sex_binary == 1:  # Male
            return 0.05  # Approximate positive value for males
        else:  # Female
            return -0.04  # Approximate negative value for females
    except Exception as e:
        st.error(f"Error scaling sex for model: {str(e)}")
        return 0.0

def predict_data(user_data, ridge_model, scaler, feature_names):
    try:
        # Convert user data to numpy array in the correct order
        feature_order = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        user_array = np.array([[user_data[feature] for feature in feature_order]])
        
        # Create a DataFrame with proper feature names to avoid the warning
        user_df = pd.DataFrame(user_array, columns=feature_names)
        
        # Scale the input data
        user_array_scaled = scaler.transform(user_df)
        
        # Make prediction
        prediction = ridge_model.predict(user_array_scaled)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
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
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-3, 3]
            )),
        showlegend=False,
        height=400
    )
    
    return fig

def create_feature_importance_chart(coefficients, feature_names):
    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    
    # Sort by absolute value of importance
    importance_df['Abs_Importance'] = abs(importance_df['Importance'])
    importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
    
    # Create the bar chart
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importance in Diabetes Progression Prediction',
                 color='Importance', color_continuous_scale='RdBu_r')
    
    fig.update_layout(height=400)
    return fig

def main():
    try:
        # Header section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="main-header">🩺 Diabetes Health Predictor</h1>', unsafe_allow_html=True)
            st.markdown("### Understand your diabetes progression risk and take control of your health")
        
        # Introduction
        st.markdown("""
        <div class="info-box">
        <h3>Welcome to your personal diabetes health assessment!</h3>
        <p>This tool helps you understand how various health factors contribute to diabetes progression. 
        By entering your information below, you'll receive a personalized prediction along with 
        educational insights to help you make informed health decisions.</p>
        <p><strong>Note:</strong> This tool is for educational purposes only and should not replace 
        professional medical advice. Always consult with healthcare providers for medical decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load your pre-trained ridge model
        ridge_model, scaler, feature_names = load_pretrained_model()
        
        if ridge_model is None:
            st.error("Failed to load the model. Please check your setup.")
            return
        
        # Create input section
        st.markdown('<h2 class="sub-header">Your Health Profile</h2>', unsafe_allow_html=True)
        st.markdown("Please provide your health information below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age input (0-100) with automatic scaling
            age_input = st.slider("**Age**", min_value=1, max_value=100, value=45, step=1,
                                 help="Age is an important factor in diabetes risk")
            
            # Sex selection with automatic encoding
            sex_selection = st.radio("**Biological Sex**", ['Female', 'Male'], 
                                    help="Diabetes risk factors can vary by sex")
            
            # BMI input with visual indicators
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
            
            # BP input with visual indicators
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
            # Cholesterol inputs
            st.markdown("**Cholesterol Levels (mg/dL)**")
            
            s1_normal = st.slider("**Total Cholesterol**", min_value=150, max_value=300, value=200, step=1,
                                 help="Desirable: <200 mg/dL")
            
            s2_normal = st.slider("**LDL (Bad Cholesterol)**", min_value=70, max_value=200, value=100, step=1,
                                 help="Optimal: <100 mg/dL")
            
            s3_normal = st.slider("**HDL (Good Cholesterol)**", min_value=20, max_value=100, value=50, step=1,
                                 help="Better: >60 mg/dL")
            
            # Blood sugar inputs
            st.markdown("**Blood Sugar Levels**")
            
            s5_normal = st.slider("**Fasting Blood Sugar (mg/dL)**", min_value=70, max_value=200, value=95, step=1,
                                 help="Normal fasting glucose: 70-100 mg/dL")
            
            s6_normal = st.slider("**Post-Meal Blood Sugar (mg/dL)**", min_value=100, max_value=300, value=140, step=1,
                                 help="Normal 2 hours after eating: <140 mg/dL")
            
            # Thyroid function
            s4_normal = st.slider("**Thyroid Stimulating Hormone (μIU/mL)**", min_value=0.1, max_value=10.0, value=2.5, step=0.1,
                                 help="Normal range: 0.4-4.0 μIU/mL")
        
        # Process the inputs after collecting them
        age = scale_age(age_input)
        sex_binary = encode_sex(sex_selection)
        sex = scale_sex_for_model(sex_binary)
        
        # Convert normal values to standardized values for the model
        # These conversion factors are approximations based on typical ranges
        bmi = (bmi_normal - 26.0) / 4.0  # Approximate standardization
        bp = (bp_normal - 94.0) / 14.0  # Approximate standardization
        s1 = (s1_normal - 190.0) / 36.0  # Approximate standardization
        s2 = (s2_normal - 115.0) / 28.0  # Approximate standardization
        s3 = (s3_normal - 50.0) / 13.0  # Approximate standardization
        s4 = (s4_normal - 4.0) / 2.0  # Approximate standardization
        s5 = (s5_normal - 100.0) / 15.0  # Approximate standardization
        s6 = (s6_normal - 125.0) / 20.0  # Approximate standardization
        
        # Create a radar chart of user inputs
        radar_categories = ['BMI', 'BP', 'Total Chol', 'LDL', 'HDL', 'Thyroid', 'Fasting BS', 'Post-Meal BS']
        radar_values = [bmi, bp, s1, s2, s3, s4, s5, s6]
        
        # Create two columns for the prediction button and visualization
        pred_col, viz_col = st.columns([1, 2])
        
        with pred_col:
            if st.button("🔍 Analyze My Diabetes Risk", type="primary", use_container_width=True):
                user_data = {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "bp": bp,
                    "s1": s1,
                    "s2": s2,
                    "s3": s3,
                    "s4": s4,
                    "s5": s5,
                    "s6": s6
                }
                
                try:
                    prediction = predict_data(user_data, ridge_model, scaler, feature_names)
                    prediction_value = round(prediction[0], 2)
                    
                    # Store the prediction in session state
                    st.session_state.prediction = prediction_value
                    st.session_state.user_data = user_data
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
        
        with viz_col:
            if 'prediction' in st.session_state:
                # Display prediction with appropriate styling
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Your Diabetes Progression Prediction")
                
                # Determine risk level
                if st.session_state.prediction < 100:
                    st.markdown(f'<p class="risk-low">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
                    st.markdown("**Lower diabetes progression risk**")
                    st.markdown("Your current health profile suggests a lower risk of diabetes progression. Keep up the good work!")
                elif st.session_state.prediction < 200:
                    st.markdown(f'<p class="risk-medium">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
                    st.markdown("**Moderate diabetes progression risk**")
                    st.markdown("There may be some areas of your health profile that could benefit from attention to reduce diabetes risk.")
                else:
                    st.markdown(f'<p class="risk-high">Prediction Score: <span style="font-size: 2rem;">{st.session_state.prediction}</span></p>', unsafe_allow_html=True)
                    st.markdown("**Higher diabetes progression risk**")
                    st.markdown("Your health profile suggests several factors that may contribute to diabetes progression. Consider consulting with a healthcare provider.")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # If we have a prediction, show detailed results and educational content
        if 'prediction' in st.session_state:
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Your Results", "📚 Health Insights", "💡 Prevention Tips", "ℹ️ About This Tool"])
            
            with tab1:
                st.markdown("### Detailed Health Analysis")
                
                # Create two columns for charts
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Radar chart
                    st.plotly_chart(create_radar_chart(radar_values, radar_categories), use_container_width=True)
                    st.caption("Your health metrics compared to standardized ranges")
                
                with chart_col2:
                    # Feature importance chart
                    if hasattr(ridge_model, 'coef_'):
                        st.plotly_chart(create_feature_importance_chart(ridge_model.coef_, feature_names), use_container_width=True)
                        st.caption("Which factors most influence diabetes progression")
                
                # Display input data summary with normal values
                st.markdown("### Your Input Summary")
                readable_data = {
                    "Age": f"{age_input} years",
                    "Sex": sex_selection,
                    "BMI": f"{bmi_normal:.1f}",
                    "Blood Pressure": f"{bp_normal:.1f} mmHg",
                    "Total Cholesterol": f"{s1_normal:.1f} mg/dL",
                    "LDL Cholesterol": f"{s2_normal:.1f} mg/dL",
                    "HDL Cholesterol": f"{s3_normal:.1f} mg/dL",
                    "Thyroid Function": f"{s4_normal:.1f} μIU/mL",
                    "Fasting Blood Sugar": f"{s5_normal:.1f} mg/dL",
                    "Post-Meal Blood Sugar": f"{s6_normal:.1f} mg/dL"
                }
                
                df = pd.DataFrame([readable_data])
                st.dataframe(df, use_container_width=True)
            
            with tab2:
                st.markdown("### Understanding Diabetes Risk Factors")
                
                # Create expandable sections for each health factor
                factors = [
                    {
                        "title": "Body Mass Index (BMI)",
                        "content": "Maintaining a healthy weight is crucial for diabetes prevention. Excess body fat, especially around the abdomen, increases insulin resistance. Even a 5-7% weight loss can significantly reduce diabetes risk.",
                        "ideal": "18.5-24.9",
                        "user_value": bmi_normal
                    },
                    {
                        "title": "Blood Pressure",
                        "content": "High blood pressure often accompanies diabetes and shares common risk factors. Controlling blood pressure reduces the risk of diabetes complications like heart disease and kidney damage.",
                        "ideal": "<120/80 mmHg",
                        "user_value": bp_normal
                    },
                    {
                        "title": "Cholesterol Levels",
                        "content": "Diabetes tends to lower 'good' HDL cholesterol and raise 'bad' LDL cholesterol and triglycerides. This combination increases heart disease risk, which is already higher in people with diabetes.",
                        "ideal": "Total: <200 mg/dL, LDL: <100 mg/dL, HDL: >60 mg/dL",
                        "user_value": f"Total: {s1_normal}, LDL: {s2_normal}, HDL: {s3_normal}"
                    },
                    {
                        "title": "Blood Sugar Levels",
                        "content": "Consistently high blood sugar levels can lead to diabetes diagnosis. Fasting levels between 100-125 mg/dL indicate prediabetes, while levels above 126 mg/dL suggest diabetes.",
                        "ideal": "Fasting: 70-100 mg/dL, Post-meal: <140 mg/dL",
                        "user_value": f"Fasting: {s5_normal}, Post-meal: {s6_normal}"
                    }
                ]
                
                for factor in factors:
                    with st.expander(factor["title"]):
                        st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
                        st.markdown(f"**Your value:** {factor['user_value']}")
                        st.markdown(f"**Ideal range:** {factor['ideal']}")
                        st.markdown(f"**Why it matters:** {factor['content']}")
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown("### Actionable Steps for Diabetes Prevention")
                
                tips = [
                    {"title": "🏃‍♂️ Regular Physical Activity", "content": "Aim for at least 150 minutes of moderate exercise per week. Activities like brisk walking, cycling, or swimming can improve insulin sensitivity."},
                    {"title": "🥗 Healthy Eating Patterns", "content": "Focus on whole foods, fiber-rich vegetables, lean proteins, and healthy fats. Limit processed foods, sugary drinks, and refined carbohydrates."},
                    {"title": "⚖️ Weight Management", "content": "If overweight, losing just 5-10% of your body weight can significantly reduce diabetes risk. Set realistic goals and celebrate small victories."},
                    {"title": "🩺 Regular Health Check-ups", "content": "Monitor key health metrics regularly. Early detection of prediabetes allows for interventions that can prevent or delay type 2 diabetes."},
                    {"title": "😴 Quality Sleep and Stress Management", "content": "Poor sleep and chronic stress can affect blood sugar levels and insulin sensitivity. Aim for 7-9 hours of quality sleep per night."},
                    {"title": "🚭 Avoid Tobacco and Limit Alcohol", "content": "Smoking increases diabetes risk and complications. If you drink alcohol, do so in moderation (up to one drink per day for women, two for men)."}
                ]
                
                for i, tip in enumerate(tips):
                    st.markdown(f'<div class="feature-card">', unsafe_allow_html=True)
                    st.markdown(f"#### {tip['title']}")
                    st.markdown(tip['content'])
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with tab4:
                st.markdown("### About This Diabetes Prediction Tool")
                
                st.markdown("""
                This tool uses machine learning to estimate diabetes progression risk based on the classic sklearn diabetes dataset.
                
                **How it works:**
                - The model was trained on health data from 442 diabetes patients
                - It considers 10 baseline variables including age, sex, BMI, blood pressure, and blood measurements
                - The prediction represents a quantitative measure of disease progression one year after baseline
                
                **Important limitations:**
                - This is an educational tool, not a diagnostic device
                - The prediction is based on statistical patterns, not individual medical assessment
                - Always consult healthcare professionals for medical advice
                
                **Dataset features:**
                - Age: Patient age
                - Sex: Patient biological sex
                - BMI: Body mass index
                - BP: Average blood pressure
                - S1-S6: Blood serum measurements including cholesterol and blood sugar levels
                """)
        
        # Add a footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
        <p>This tool is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p>Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
        </div>
        """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please check your code and try again.")

if __name__ == "__main__":
    main()