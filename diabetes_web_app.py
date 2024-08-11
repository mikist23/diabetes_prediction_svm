import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
loaded_model = pickle.load(open("C:/Users/User/Documents/codebasics/diabetes_ml_project/trained_model.sav", 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_np_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 1:
        return 'The person is diabetic'
    else:
        return 'The person is non-diabetic'

def main():
    # Page Title
    st.markdown("<h1 style='text-align: center; color: #FF6347;'>Diabetes Prediction Web App</h1>", unsafe_allow_html=True)

    # Input from the user
    st.sidebar.header('User Input Parameters')
    
    Pregnancies = st.sidebar.number_input('Number of pregnancies', min_value=0, max_value=20, step=1, value=0)
    Glucose = st.sidebar.number_input('Glucose level', min_value=0, max_value=200, step=1, value=120)
    BloodPressure = st.sidebar.number_input('Blood Pressure value', min_value=0, max_value=150, step=1, value=70)
    SkinThickness = st.sidebar.number_input('Skin Thickness value', min_value=0, max_value=100, step=1, value=20)
    Insulin = st.sidebar.number_input('Insulin level', min_value=0, max_value=900, step=1, value=80)
    BMI = st.sidebar.number_input('BMI value', min_value=0.0, max_value=70.0, step=0.1, value=25.0)
    DiabetesPedigreeFunction = st.sidebar.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, step=0.01, value=0.5)
    Age = st.sidebar.number_input('Age of the person', min_value=1, max_value=120, step=1, value=25)

    # Code for prediction
    diagnosis = ''
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    if st.button('Get Diabetes Test Result'):
        diagnosis = diabetes_prediction(input_data)
        st.success(diagnosis)

        # Visualization
        st.header("User Input Data Visualization")

        # Convert input data into a DataFrame for visualization
        input_data_df = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

        # Display input data as a table
        st.write("Input Data:")
        st.write(input_data_df)

        # Line Plot: Glucose Level
        st.subheader("Glucose Level")
        fig, ax = plt.subplots()
        sns.lineplot(data=input_data_df, x=input_data_df.index, y='Glucose', marker='o', ax=ax)
        ax.set_xlabel("Index")
        ax.set_ylabel("Glucose Level")
        st.pyplot(fig)

        # Line Plot: BMI
        st.subheader("BMI")
        fig, ax = plt.subplots()
        sns.lineplot(data=input_data_df, x=input_data_df.index, y='BMI', marker='o', color='green', ax=ax)
        ax.set_xlabel("Index")
        ax.set_ylabel("BMI")
        st.pyplot(fig)

        # # Line Plot: Age
        # st.subheader("Age")
        # fig, ax = plt.subplots()
        # sns.histplot(data=input_data_df, kde=True, color='skyblue', ax=ax)
        # ax.set_xlabel("Index")
        # ax.set_ylabel("Age")
        # st.pyplot(fig)

    # Footer
    st.markdown("<footer style='text-align: center;'>© 2024 Diabetes Prediction App</footer>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
