import pandas as pd
import os
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
data = r'data\insurance.csv'

def training_model():
    """trains the insurance model using the dataset and saves the trained model files"""
    try:
        #Data Prepration
        df=pd.read_csv(data)
        #Encoding
        
        encoder=OneHotEncoder(sparse_output=False,drop="first")
        encoded=encoder.fit_transform(df[['sex','smoker','region']]
        )
        encoded_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(['sex','smoker','region']))

        #seperating x and y
        x=pd.concat([df[['age','bmi','children']],encoded_df],axis=1)
        #target variable
        #axis=1 combining column side by side
        y=df['charges']

        #train-test split
       
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

        #Standardization
       
        scaler=StandardScaler()
        #transform=coverting into standardize values
        x_train=scaler.fit_transform(x_train)
        #applying the learned scale in test data
        x_test=scaler.transform(x_test)

        #Model training
        #Creating model
        model=LinearRegression()
        model.fit(x_train,y_train)

        #prediction
        y_pred=model.predict(x_test)#predicting the 4 rows which model haven't seen

        #metrices
        mae=mean_absolute_error(y_test,y_pred)
        rmse=mean_squared_error(y_test,y_pred)**0.5
        r2=r2_score(y_test,y_pred) 
        #models path
        models_path=os.path.abspath("../../models") 
        joblib.dump(encoder,os.path.join(models_path,"encoder.pk1"))          
        joblib.dump(scaler,os.path.join(models_path,"scaler.pk1"))
        joblib.dump(model,os.path.join(models_path,"insurance_model"))
        print(f"models saved on ",models_path)
        print(f"Training RMSE:",rmse)
        print(f"Training r2:",r2)
        
    except Exception as e:
        print(f"Training failed:{e}")

if __name__ == "__main__":
    training_model()
