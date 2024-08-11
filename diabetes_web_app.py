import numpy as np
import pickle
import streamlit as st

# loading saved model
loaded_model = pickle.load(open("C:/Users/User/Documents/codebasics/diabetes_ml_project/trained_model.sav",'rb'))

# creating a function for predcion
def diabetes_prediction(input_data):



    
    
    # Changing the data to numpy array
    input_data_as_np_array = np.asarray(input_data)

    # Reshape the array predicting one instance
    input_data_reshaped = input_data_as_np_array.reshape(1,-1)


    # Prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    
    if prediction[0] == 1:
        return 'The person is diabetic'
    else:
        return 'The person is non diabetic'

def main():

    # giving a title
    st.title("Diabetes Predicction web app")

    # input from the user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness= st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the person')


    # code for prediction
    diagnosis = ''

    # creating a pattern for prediction
    if st.button('Diabetes Test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()