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
    