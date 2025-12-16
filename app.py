import streamlit as st
import pandas as pd
import joblib

# Function to load pretrained model and scaler
@st.cache_resource(show_spinner=True)
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to preprocess the input data
def preprocess_input(data):
    # Convert 2 distinct categorical value to boolean
    sex_map = {'Male': True, 'Female': False}
    fastingbs_map = {'Yes': True, 'No': False}
    exercise_angina_map = {'Yes': True, 'No': False}
    
    # Convert >2 distinct categorical value to one-hot encoding
    chestpain_map = {
        'Asymptomatic': 'ASY',
        'Atypical Angina': 'ATA',
        'Non-anginal Pain': 'NAP',
        'Typical Angina': 'TA'
    }
    restingecg_map = {
        'Left Ventricular Hypertrophy': 'LVH',
        'Normal': 'Normal',
        'ST-T wave abnormality': 'ST'
    }
    st_slope_map = {
        'Down': 'Down',
        'Flat': 'Flat',
        'Up': 'Up'
    }

    # Create DataFrame
    df = pd.DataFrame([{
        'Age': data[0],
        'Sex': sex_map[data[1]],
        'RestingBP': data[3],
        'Cholesterol': data[4],
        'FastingBS': fastingbs_map[data[5]],
        'MaxHR': data[7],
        'ExerciseAngina': exercise_angina_map[data[8]],
        'Oldpeak': data[9],
        'ChestPainType_ASY': chestpain_map[data[2]] == 'ASY',
        'ChestPainType_ATA': chestpain_map[data[2]] == 'ATA',
        'ChestPainType_NAP': chestpain_map[data[2]] == 'NAP',
        'ChestPainType_TA': chestpain_map[data[2]] == 'TA',
        'RestingECG_LVH': restingecg_map[data[6]] == 'LVH',
        'RestingECG_Normal': restingecg_map[data[6]] == 'Normal',
        'RestingECG_ST': restingecg_map[data[6]] == 'ST',
        'ST_Slope_Down': st_slope_map[data[10]] == 'Down',
        'ST_Slope_Flat': st_slope_map[data[10]] == 'Flat',
        'ST_Slope_Up': st_slope_map[data[10]] == 'Up'
        }])
    
    return df

# Function to scale the preprocessed data
def scale_input(df, scaler):
    scaled_array = scaler.transform(df)
    return scaled_array

# Load model and scaler
MODEL_PATH='model/bernoulli_naive_bayes_model.pkl'
SCALER_PATH='model/scaler.pkl'
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)

# ---------- 0. Streamlit page config ----------
st.set_page_config(
    page_title='Heart Disease Prediction DSS',
    page_icon=':material/cardiology:',
)

# ------------------ 1. Title and Description ------------------
st.markdown(
    """
    <style>
    .custom-card {
        width: 100%;
        margin: 0 auto 1.5em auto;
        padding: 0em 0em 0em 0em;
        border-radius: 16px;
        border: 1.5px solid var(--primary-border, #e0e0e0);
        background: var(--primary-bg, #fff);
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        transition: background 0.3s, border 0.3s;
    }
    @media (prefers-color-scheme: dark) {
        .custom-card {
            border: 1.5px solid var(--primary-border-dark, #333);
            background: var(--primary-bg-dark, #22272e);
        }
    }
    .material-symbols-rounded {
        font-family: 'Material Symbols Rounded';
        font-variation-settings:
            'FILL' 1,
            'wght' 400,
            'GRAD' 0,
            'opsz' 48;
        font-size: 2em;
        vertical-align: middle;
        color: #e53935;
    }
    </style>
    <div class="custom-card">
        <h1 style='text-align: center;'>
            <span class="material-symbols-rounded">cardiology</span>
            Heart Disease Prediction
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div style='text-align: justify;'>
    Enter your health data to predict the probability of heart disease using our machine learning model.
    This application will analyze the information you provide and estimate your risk of developing heart disease.
    For the best results, please ensure the data you enter accurately reflects your health condition.
    The model considers several factors such as age, gender, blood pressure, cholesterol level, maximum heart rate, and other medical examination results.
    <br><br>
    </div>
    ''',
    unsafe_allow_html=True
)

# ------------------ Expander: Input Variable Description ------------------
with st.expander('Glossary of Health Terms (See Details)'):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/heart-disease.jpg", use_container_width=True)
    with col2:
        st.markdown(
            """
            <ul>
                <li><b>Age</b>: Age of the patient in years.</li>
                <li><b>Sex</b>: Gender of the patient (Male or Female).</li>
                <li><b>Chest Pain Type</b>: Type of chest discomfort experienced:
                    <ul>
                        <li><b>Typical Angina:</b> Chest pain related to heart problems</li>
                        <li><b>Atypical Angina:</b> Chest pain with unusual characteristics</li>
                        <li><b>Non-anginal Pain:</b> Chest pain not related to the heart</li>
                        <li><b>Asymptomatic:</b> No chest pain symptoms</li>
                    </ul>
                </li>
                <li><b>Resting Blood Pressure</b>: Blood pressure measured while resting.</li>
                <li><b>Cholesterol</b>: Level of fat in the blood that may affect heart health.</li>
                <li><b>Fasting Blood Sugar</b>: Blood sugar level measured after fasting.</li>
                <li><b>Resting ECG</b>: Result of an electrical recording of the heart at rest.</li>
                <li><b>Maximum Heart Rate</b>: Highest heart rate reached during physical activity.</li>
                <li><b>Exercise-Induced Angina</b>: Chest pain that occurs during exercise.</li>
                <li><b>Oldpeak</b>: Change in heart electrical activity during exercise.</li>
                <li><b>ST Segment Slope</b>: Pattern of heart activity during peak exercise.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

# ------------------ 2. Input Data Form ------------------
st.subheader(':material/assignment: Input Health Data Below:')
with st.form("heart_disease_form", border=True):
    age = st.number_input('How old are you? (years)', min_value=0, max_value=120, value=30)
    sex = st.selectbox('What is your gender?', options=['Male', 'Female'], index=1)
    chestpain_type = st.selectbox(
        'What type of chest pain do you experience?',
        options=['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    )
    resting_bp = st.number_input('What is your resting blood pressure? (mm Hg)', min_value=80, max_value=200, value=110)
    cholesterol = st.number_input('What is your cholesterol level? (mg/dL)', min_value=100, max_value=600, value=150)
    fasting_bs = st.selectbox('Is your fasting blood sugar greater than 120 mg/dL?', options=['Yes', 'No'])
    resting_ecg = st.selectbox(
        'What is your resting electrocardiogram (ECG) result?',
        options=['Normal', 'ST-T wave abnormality', 'Left Ventricular Hypertrophy']
    )
    max_hr = st.number_input('What is your maximum heart rate achieved? (bpm)', min_value=60, max_value=220, value=150)
    exercise_angina = st.selectbox('Do you experience angina induced by exercise?', options=['Yes', 'No'], index=1)
    oldpeak = st.number_input('What is your ST depression value (Oldpeak 0.0 - 6.0)?', min_value=0.0, max_value=6.0, value=1.5, step=0.1)
    st_slope = st.selectbox(
        'What is the slope of your peak exercise ST segment?',
        options=['Down', 'Flat', 'Up']
    )

    # Submit button
    submitted = st.form_submit_button(':material/manufacturing: Predict The Outcome', use_container_width=True)

if submitted:
    # Prepare input data for prediction
    data_input_raw = [age, sex, chestpain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]
    processed_data = preprocess_input(data_input_raw)
    scaled_data = scale_input(processed_data, scaler)

    # Make prediction
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    # ------------------ 3. Prediction Result ------------------
    st.subheader(':material/manufacturing: Prediction Result')
    if prediction[0] == 1:
        st.error(
            f":material/heart_minus: The model indicates a **presence of heart disease** "
            f"with an estimated probability of **{prediction_proba[0][1]*100:.2f}%**."
        )
    else:
        st.success(
            f":material/heart_check: The model indicates an **absence of heart disease** "
            f"with an estimated probability of **{prediction_proba[0][0]*100:.2f}%**."
        )

    # ------------------ 4. Suggestion ------------------
    st.subheader(':material/stars_2: Clinical Recommendation')
    if prediction[0] == 1:
        st.info(
            "Based on the prediction results, it is strongly recommended that you consult "
            "a qualified cardiologist or healthcare professional for a comprehensive "
            "clinical evaluation. Further diagnostic tests may be required to confirm "
            "the presence and severity of potential heart disease. In addition, adopting "
            "appropriate lifestyle modifications is advised, including maintaining a "
            "balanced and heart-healthy diet, engaging in regular physical activity suited "
            "to your condition, managing stress levels, and avoiding risk factors such as "
            "smoking and excessive alcohol consumption. Regular medical follow-ups are "
            "important to monitor cardiovascular health and support early intervention."
        )
    else:
        st.info(
            "Based on the prediction results, no significant indication of heart disease "
            "is identified at this time. It is recommended to continue maintaining a "
            "healthy lifestyle by following a balanced diet, engaging in regular physical "
            "activity, and managing stress effectively. Periodic health check-ups and "
            "routine monitoring of key cardiovascular indicators, such as blood pressure "
            "and cholesterol levels, are encouraged to help sustain long-term heart health "
            "and support early detection of potential risks."
        )

    # Caption Disclaimer
    st.caption(
    "This result is generated by a machine learning model and serves only "
    "as a decision support tool. It should be interpreted with caution and must be "
    "considered alongside clinical judgment and professional medical evaluation."
    )
