import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error,mean_squared_error

#page config
st.set_page_config(" Simple Linear Regression",layout="centered")

#load the css file
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)#to allow html tags by default streamlit does not allow  html tags ,so to enable it we have to use unsafe_allow_html=True
load_css("style.css")
#------------
#title
st.markdown(""" 
<div class="card">
            <h1> Simple Linear Regression App </h1>
            <p> <b>Build and predict the tip amount based on total bill using Simple Linear Regression</b> </p>
            <div>""",unsafe_allow_html=True)
#------------
#load dataset
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
data=load_data()
#data preview
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Dataset Preview ")
st.dataframe(data.head())
st.markdown('</div>',unsafe_allow_html=True)
#------------

#prepare data
x,y=data[["total_bill"]],data["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
#------------
#train model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

#------------
#model evaluation
r2=r2_score(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
mse=mean_squared_error(y_test,y_pred)
ajusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
#------------
#display evaluation metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Model Evaluation(Performance) Metrics ")
st.markdown(f"""
- R² Score: {r2:.4f}
- Mean Absolute Error (MAE): {mae:.4f}
- Root Mean Squared Error (RMSE): {rmse:.4f}
- Mean Squared Error (MSE): {mse:.4f}
- Adjusted R² Score: {ajusted_r2:.4f}
""")
st.markdown('</div>',unsafe_allow_html=True)

# st.markdown('<div class="card">',unsafe_allow_html=True)
# st.subheader("model performance on Test Data")
# c1,c2=st.columns(2)
# c1.metric("R² Score",f"{r2:.4f}")
# c2.metric("MAE",f"{mae:.4f}")
# st.markdown('</div>',unsafe_allow_html=True)

#------------
#visualize results
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader(" Total Bill vs Tip Amount (Actual vs Predicted) ")
fig,ax=plt.subplots()
# ax.scatter(x_test,y_test,color='blue',label='Actual Tips')
# ax.scatter(x_test,y_pred,color='red',label='Predicted Tips')
ax.scatter(data['total_bill'],data['tip'],alpha=0.7,label='Data Points',color='blue')
ax.plot(data['total_bill'],model.predict(scaler.transform(data[['total_bill']])),color='red',label='Regression Line')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.legend()
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)
#------------

#intercept and coefficient
st.markdown(f"""
            <div class="card">
            <h3> model Intercept and Coefficient </h3>
            <p> Intercept: {model.intercept_:.4f} </p>
            <p> Coefficient for Total Bill: {model.coef_[0]:.4f} </p>
            </div>""",unsafe_allow_html=True)
# ------------ Prediction Section ------------

st.markdown('<div class="card">', unsafe_allow_html=True)

total_bill = st.number_input(
    "Enter Total Bill Amount to Predict Tip:",
    min_value=float(data['total_bill'].min()),
    max_value=float(data['total_bill'].max()),
    value=30.0,
    step=1.0
)

tip = model.predict(
    scaler.transform([[total_bill]])
)[0]

st.markdown(
    f'''
    <div class="prediction-box">
        Predicted Tip Amount for Total Bill of {total_bill:.2f}
        <br>
        <b>{tip:.2f}</b>
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)







