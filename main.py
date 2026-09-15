from src.app.app import App
import joblib 

encoder=joblib.load('models/encoder.pk1')
scaler=joblib.load('models/scaler.pk1')
model=joblib.load('models/insurance_model')
if __name__ == "__main__":
    st_app = App(encoder,scaler,model)
    st_app.run()