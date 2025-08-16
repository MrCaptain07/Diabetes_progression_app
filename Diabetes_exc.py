import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

# Load pre-trained model
@st.cache_data
def load_pretrained_model():
    try:
        # Load your pre-trained diabetes model
        # Option 1: If saved with pickle
        with open('diabetes.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        
        # Option 2: If saved with joblib (uncomment if using joblib)
        # loaded_data = joblib.load('diabetes.pkl')
        
        # Check if the loaded data is a tuple (model, scaler) or just the model
        if isinstance(loaded_data, tuple):
            # If it's a tuple, extract model and scaler
            if len(loaded_data) == 2:
                ridge_model, scaler = loaded_data
                st.info("Loaded model and scaler from tuple")
            else:
                # If tuple has different structure, extract just the model (first element)
                ridge_model = loaded_data[0]
                # Create a new scaler
                diabetes = load_diabetes()
                X, y = diabetes.data, diabetes.target
                scaler = StandardScaler()
                scaler.fit(X)
                st.info("Loaded model from tuple, created new scaler")
        else:
            # If it's just the model
            ridge_model = loaded_data
            # Create a new scaler
            diabetes = load_diabetes()
            X, y = diabetes.data, diabetes.target
            scaler = StandardScaler()
            scaler.fit(X)
            st.info("Loaded model only, created new scaler")
        
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

def predict_data(user_data, ridge_model, scaler):
    try:
        # Convert user data to numpy array in the correct order
        feature_order = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        user_array = np.array([[user_data[feature] for feature in feature_order]])
        
        # Scale the input data
        user_array_scaled = scaler.transform(user_array)
        
        # Make prediction
        prediction = ridge_model.predict(user_array_scaled)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return [0.0]

def main():
    try:
        st.title("Diabetes Progression Prediction")
        st.write("Enter your medical data to get a prediction for diabetes progression")
        st.write("*Note: This is for educational purposes only and should not replace professional medical advice*")
        
        # Load your pre-trained ridge model
        ridge_model, scaler, feature_names = load_pretrained_model()
        
        if ridge_model is None:
            st.error("Failed to load the model. Please check your setup.")
            return
        
        # Create input fields based on diabetes dataset features
        col1, col2 = st.columns(2)
        
        with col1:
            # Age input (0-100) with automatic scaling
            age_input = st.number_input("Age", min_value=1, max_value=100, value=25, step=1,
                                       help="Enter your age (1-100 years)")
            
            # Sex selection with automatic encoding
            sex_selection = st.selectbox("Sex", options=['Female', 'Male'], 
                                        help="Select your biological sex")
            
            bmi = st.number_input("BMI (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                 help="Standardized Body Mass Index")
            bp = st.number_input("Blood Pressure (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized average blood pressure")
            s1 = st.number_input("S1 - Total Cholesterol (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized total serum cholesterol")
        
        with col2:
            s2 = st.number_input("S2 - LDL Cholesterol (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized low-density lipoproteins")
            s3 = st.number_input("S3 - HDL Cholesterol (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized high-density lipoproteins")
            s4 = st.number_input("S4 - Thyroid Function (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized thyroid stimulating hormone")
            s5 = st.number_input("S5 - Blood Sugar (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized lamotrigine")
            s6 = st.number_input("S6 - Blood Sugar (standardized)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                                help="Standardized blood sugar measure")
        
        # Process the inputs after collecting them
        age = scale_age(age_input)
        sex_binary = encode_sex(sex_selection)
        sex = scale_sex_for_model(sex_binary)
        
        # Display processed values
        st.write(f"**Processed Values:** Age: {age:.3f}, Sex: {sex:.3f}")
        
        if st.button("Predict Diabetes Progression"):
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
            
            # Display readable input summary
            readable_data = {
                "Age": f"{age_input} years",
                "Sex": sex_selection,
                "BMI (standardized)": bmi,
                "Blood Pressure (standardized)": bp,
                "Total Cholesterol (standardized)": s1,
                "LDL Cholesterol (standardized)": s2,
                "HDL Cholesterol (standardized)": s3,
                "Thyroid Function (standardized)": s4,
                "Blood Sugar 1 (standardized)": s5,
                "Blood Sugar 2 (standardized)": s6
            }
            
            try:
                prediction = predict_data(user_data, ridge_model, scaler)
                
                # Display prediction
                st.success(f"Diabetes Progression Prediction: {round(prediction[0], 2)}")
                
                # Provide interpretation
                if prediction[0] < 100:
                    st.info("This suggests lower diabetes progression.")
                elif prediction[0] < 200:
                    st.warning("This suggests moderate diabetes progression.")
                else:
                    st.error("This suggests higher diabetes progression.")
                
                # Display input data summary
                st.subheader("Input Data Summary:")
                df = pd.DataFrame([readable_data])
                st.dataframe(df)
                
                # Optional: Save to database (uncomment if you have MongoDB setup)
                # user_data['prediction'] = round(float(prediction[0]), 2)
                # collection.insert_one(user_data)
                # st.success("Data saved to database!")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        
        # Add information about the dataset
        with st.expander("About the Diabetes Dataset"):
            st.write("""
            This prediction model uses the sklearn diabetes dataset, which contains:
            - 442 diabetes patients
            - 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements)
            - A quantitative measure of disease progression one year after baseline
            
            **Features:**
            - Age: Patient age (automatically scaled from your input)
            - Sex: Patient gender (automatically encoded)
            - BMI: Body mass index
            - BP: Average blood pressure
            - S1-S6: Six blood serum measurements
            
            All features are standardized (mean=0, std=1).
            The target variable represents disease progression (higher values = more progression).
            """)
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please check your code and try again.")

if __name__ == "__main__":
    main()