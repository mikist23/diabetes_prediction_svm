import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
loaded_model = pickle.load(open("C:/Users/User/Documents/codebasics/diabetes_ml_project/trained_model.sav",'rb'))

input_data = (2,197,70,45,543,30.5,0.158,53)
# Changing the data to numpy array
input_data_as_np_array = np.asarray(input_data)

# Reshape the array predicting one instance
input_data_reshaped = input_data_as_np_array.reshape(1,-1)

# standarized the data
# std_data = scaler.transform(input_data_reshaped)
# print(std_data)


# Prediction
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


if prediction[0] == 1:
    print('The person is diabetic')
else:
    print('The person is non diabetic')