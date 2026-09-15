import pandas as pd
import joblib

def predict_insurance(user_input):
    """takes user details as input and returns insurance charges with status code"""
    try:

        age = user_input.get("age")
        height = user_input.get("height")
        weight = user_input.get("weight")
        sex = user_input.get("sex")
        children = user_input.get("children")
        smoker = user_input.get("smoker")
        region = user_input.get("region")
        encoder = user_input.get("encoder")
        scaler = user_input.get("scaler")
        model = user_input.get("model")

        height_m=height*0.01
        bmi=weight/(height_m**2)
        data_dict={
            "age":age,
            "bmi":bmi,
            "sex":sex.lower(),#using lower
            "children":children,
            "smoker":smoker.lower(),
            "region":region.lower()
        }
        
        input_data=pd.DataFrame([data_dict])
        encoded_input=encoder.transform(input_data[['sex','smoker','region']])
        #combine numerical and encoded data
        encoded_df=pd.DataFrame(encoded_input,columns=encoder.get_feature_names_out(['sex','smoker','region']))
        #combining encoded categorical data and numerical data
        final_input=pd.concat([input_data[['age','bmi','children']],encoded_df],axis=1)
        #standardize input data
        scaled_input=scaler.transform(final_input)
        #make prediction
        prediction=model.predict(scaled_input)[0]
        return {
            "status":200,
            "prediction":prediction
        }
    except Exception as e:
        print(f"Prediction failed:",e)
        return{
            "status":500,
            "error":str(e)
        }

