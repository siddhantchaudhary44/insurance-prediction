import streamlit as st
import joblib
import os
from src.prediction.prediction import predict_insurance

class App:
    """handles the insurance prediction app"""
    def __init__(self,encoder,scaler,model):
        self.encoder=encoder
        self.scaler=scaler
        self.model=model
    def run(self):
        """shows the app and takes user input"""
        st.title('Insurance Cost Prediction')
        st.write('Enter the details below to predict insurance charges')
        #user input
        age=st.text_input('Age',placeholder="Enter your age")
        weight=st.number_input('Weight(kg)',min_value=0.0,value=None)
        height=st.number_input("Height (cm)",min_value=0.0,value=None)
        
        children=st.number_input('Children',min_value=0,max_value=10)
        sex=st.selectbox('Sex',['Select','Female','Male'])
        smoker=st.selectbox('Smoker',['Select','Yes','No'])
        region=st.selectbox('Region',['Select','Northeast','Northwest','Southeast','Southwest'])
        #Run button
        if st.button("Run"):
            if age=="":
                st.warning("Please enter your age to proceed.")
            elif sex=="Select":
                st.warning("Please select your sex to proceed.")
            elif smoker=="Select":
                st.warning("Please select smoker to proceed.")
            elif region=="Select":
                st.warning("Please select your region to proceed.")
            else:
                user_input={
                "age":age,
                "height":height,
                "weight":weight,
                "sex":sex.lower(),#using lower
                "children":children,
                "smoker":smoker.lower(),
                "region":region.lower(),
                "encoder": self.encoder,
                "scaler": self.scaler,
                "model": self.model
                }    
                prediction_response=predict_insurance(user_input)
                if prediction_response:
                    status=prediction_response.get('status')
                    if status==200:
                        st.success(f"Predicted insurance charges: ₹ {prediction_response.get('prediction'):,.2f}")
                    else:
                        st.error(f"Error occured: {prediction_response.get('error')}")
                else:
                    st.warning("Could not get the prediction.")
        