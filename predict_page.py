import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title(" 2020 Software Developer Salary Prediction")

    st.write("""### Please provide some information to predict salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )
    
    # creating a selectbox...

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    
# creating a slider so that user can slide ( kinda fun ) can use a select box instead if you want  but change it to "experience = st.selectbox("Years of Experience", list(range(0, 51)))"
    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    
    # if we click on the button
    if ok:
        X = np.array([[country, education, expericence]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        
        #display 
        
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")