import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="ObesitySense: Obesity Level Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model, scaler, encoders
@st.cache_resource
def load_models():
    model = joblib.load('models/RandomForest.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('encoders/label_encoder_target.pkl')
    onehot_encoder = joblib.load('encoders/onehot_encoder_MTRANS.pkl')
    ordinal_encoder = joblib.load('encoders/ordinal_encoders.pkl')
    ptransformer = joblib.load('transformers/yeojohn_ptransformer.pkl')
    return model, scaler, label_encoder, onehot_encoder, ordinal_encoder, ptransformer

model, scaler, label_encoder, onehot_encoder, ordinal_encoder, ptransformer = load_models()

# Preprocessing function
ordinal_cols = ['FCVC', 'NCP', 'CAEC', 'CH2O', 'FAF', 'TUE', 'CALC', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'Gender']
onehot_cols = ['MTRANS']
yeo_john_cols = ['Age']
scale_cols = ['Age', 'Height', 'Weight']

def preprocess_user_input(user_input):
    user_input[scale_cols] = scaler.transform(user_input[scale_cols])
    user_input[yeo_john_cols] = ptransformer.transform(user_input[yeo_john_cols])
    for col in ordinal_cols:
        user_input[[col]] = ordinal_encoder[col].transform(user_input[[col]])
    onehot_transformed = onehot_encoder.transform(user_input[onehot_cols])
    onehot_df = pd.DataFrame(onehot_transformed, columns=onehot_encoder.get_feature_names_out(onehot_cols))
    user_input = user_input.drop(columns=onehot_cols).reset_index(drop=True)
    user_input = pd.concat([user_input, onehot_df], axis=1)
    return user_input

# Sidebar navigation
st.sidebar.title("ObesitySense")
page = st.sidebar.radio("Go to:", ("Home - Prediction", "Dashboard", "Model Information"))

# Home, Prediction Page
if page == "Home - Prediction":
    st.title("ObesitySense: Obesity level Analysis and Prediction")
    
    with st.form("prediction_form"):
        st.subheader("Enter Your Details")
        col1, col2 = st.columns(2)
        
        with col1:
            Gender = st.selectbox("Gender", ['Female', 'Male'], help="Select your gender.")
            Age = st.number_input("Age", min_value=1, max_value=100, value=25, help="Enter your age in years.")
            Height = st.number_input("Height (meters)", min_value=0.5, max_value=2.5, value=1.70, help="Enter height in meters.")
            Weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=70, help="Enter weight in kilograms.")
            family_history_with_overweight = st.selectbox("Family History with Overweight?", ["no", "yes"], help="Any family history of obesity?")
            FAVC = st.selectbox("Eat high-calorie food?", ["no", "yes"], help="Do you frequently consume high-calorie foods?")
            FCVC = st.selectbox("Vegetable Consumption", ["Never", "Sometimes", "Always"], help="How often do you eat vegetables?")
            NCP = st.selectbox("Number of Main Meals", ["Btwn 1 & 2", "3", "More than 3", "No Answer"], help="Daily main meals.")
        
        with col2:
            CAEC = st.selectbox("Eat Between Meals?", ['no', 'Sometimes', 'Frequently', "Always"], help="Do you snack between meals?")
            SMOKE = st.selectbox("Do You Smoke?", ["no", "yes"], help="Do you smoke?")
            CH2O = st.selectbox("Daily Water Intake", ['less than 1L', 'Btwn 1L & 2L', 'More than 2L'], help="Daily water consumption.")
            SCC = st.selectbox("Monitor Calories?", ["no", "yes"], help="Do you track calorie intake?")
            FAF = st.selectbox("Physical Activity", ['Never','1 to 2times','2 to 4 times', '4 or 5 times'], help="Weekly physical activity frequency.")
            TUE = st.selectbox("Time Using Technology", ['0 to 2h', '3 to 5h', 'More than 5h'], help="Daily technology usage.")
            CALC = st.selectbox("Alcohol Consumption", ['no', 'Sometimes', 'Frequently', 'Always'], help="Alcohol consumption frequency.")
            MTRANS = st.selectbox("Transportation", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"], help="Primary mode of transport.")
        
        submit = st.form_submit_button("Predict")
        
        if submit:
            user_input = pd.DataFrame({
                "Gender": [Gender], "Age": [Age], "Height": [Height], "Weight": [Weight],
                "family_history_with_overweight": [family_history_with_overweight], "FAVC": [FAVC],
                "FCVC": [FCVC], "NCP": [NCP], "CAEC": [CAEC], "SMOKE": [SMOKE], "CH2O": [CH2O],
                "SCC": [SCC], "FAF": [FAF], "TUE": [TUE], "CALC": [CALC], "MTRANS": [MTRANS]
            })
            preprocessed_input = preprocess_user_input(user_input)
            prediction = model.predict(preprocessed_input)
            prediction_class = label_encoder.inverse_transform(prediction)[0]
            
            st.subheader("Prediction Result")
            st.success(f"Predicted Obesity Level: {prediction_class}")

# Dashboard Page
if page == "Dashboard":
    st.title("Data Insights")
    
    @st.cache_data
    def load_data():
        df_og = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
        df_preprocessed = pd.read_csv('preprocessed_obesity.csv')
        return df_og, df_preprocessed
    
    df_og, df_preprocessed = load_data()
    
    st.subheader("Filter Data")
    col1, col2 = st.columns(2)
    with col1:
        gender_filter = st.multiselect("Gender", options=df_og['Gender'].unique(), default=df_og['Gender'].unique())
    with col2:
        age_range = st.slider("Age Range", min_value=int(df_og['Age'].min()), max_value=int(df_og['Age'].max()), value=(int(df_og['Age'].min()), int(df_og['Age'].max())))
    
    filtered_df = df_og[df_og['Gender'].isin(gender_filter) & (df_og['Age'].between(age_range[0], age_range[1]))]
    
    st.subheader("Obesity Level Distribution")
    fig_dist = px.histogram(
        filtered_df, x='NObeyesdad', color='NObeyesdad',
        title="Distribution of Obesity Levels",
        labels={'NObeyesdad': 'Obesity Level'},
        height=400
    )
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("Key Feature Distributions")
    feature_to_plot = st.selectbox("Select Feature", ['Weight', 'FCVC', 'FAF', 'CH2O'], help="Choose a feature to visualize its distribution.")
    if feature_to_plot == 'Weight':
        fig_feature = px.histogram(
            filtered_df, x='Weight', color='NObeyesdad',
            title="Weight Distribution by Obesity Level",
            labels={'Weight': 'Weight (kg)'},
            height=400
        )
    elif feature_to_plot == 'FCVC':
        fig_feature = px.histogram(
            filtered_df, x='FCVC', color='NObeyesdad',
            title="Vegetable Consumption Frequency by Obesity Level",
            labels={'FCVC': 'Vegetable Consumption'},
            height=400
        )
    elif feature_to_plot == 'FAF':
        fig_feature = px.histogram(
            filtered_df, x='FAF', color='NObeyesdad',
            title="Physical Activity Frequency by Obesity Level",
            labels={'FAF': 'Physical Activity'},
            height=400
        )
    else:  # CH2O
        fig_feature = px.histogram(
            filtered_df, x='CH2O', color='NObeyesdad',
            title="Water Intake by Obesity Level",
            labels={'CH2O': 'Daily Water Intake'},
            height=400
        )
    st.plotly_chart(fig_feature, use_container_width=True)
    
    st.subheader("Weight vs Height")
    fig_scatter = px.scatter(
        filtered_df, x='Height', y='Weight', color='NObeyesdad',
        hover_data=['Gender', 'Age'], size='Age',
        title="Weight vs Height by Obesity Level",
        labels={'Height': 'Height (m)', 'Weight': 'Weight (kg)'},
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Model Information Page
if page == "Model Information":
    st.title("Model Information")
    
    st.subheader("Data Preprocessing")
    st.markdown("""
    - **StandardScaler**: Standardized numerical features (`Age`, `Height`, `Weight`) .
    - **Yeo-Johnson Transformer**: Applied to `Age` to handle skewness and improve normality.
    - **Ordinal Encoder**: Encoded categorical features (e.g., `FCVC`, `CAEC`, `FAF`).
    - **One-Hot Encoder**: Encoded nominal feature `MTRANS` to create binary columns for each category.
    - **Label Encoder**: Encoded the target `NObeyesdad` into numerical labels.
    """)
    
    st.subheader("Random Forest Classifier")
    st.markdown("""
    - **Algorithm**: Random Forest, an ensemble of decision trees, was selected for its ability to handle complex, non-linear relationships and avoid overfitting.
    - **Training**: The model was trained on the preprocessed dataset, with hyperparameter tuning performed using RandomizedSearchCV to optimize parameters like `n_estimators` and `max_depth`.
    - **Performance**:
      - **Accuracy**: Achieved 94% without optimization.
      - **Improvement**: Hyperparameter tuning improved accuracy by 1%.
    """)
    
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Confusion Matrix**")
        st.image('confusion_matrix.png', use_column_width=True)
    with col2:
        st.markdown("**Performance Metrics (optimized)**")
        st.markdown("""
        - **Accuracy: 95%**
        - **Avg Precision: 95%**
        - **Avg Recall: 95%**
        - **Avg F1 Score: 95%**
        """)
    
    st.subheader("Feature Importance")
    st.markdown("The following plot shows the importance of each feature in the Random Forest model, highlighting which factors most influence obesity predictions.")
    feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS_Automobile', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']
    importances = model.feature_importances_
    feat_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig_fi = px.bar(
        feat_importance_df, x='Importance', y='Feature',
        title="Feature Importance in Random Forest Model",
        height=500
    )
    fig_fi.update_layout(xaxis_title="Importance Score", yaxis_title="Feature")
    st.plotly_chart(fig_fi, use_container_width=True)
    
