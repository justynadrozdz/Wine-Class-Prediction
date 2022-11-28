import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Wine Class Prediction App
""")

st.sidebar.header('Input Parameters')

#Function to get features from user
def user_input_features():
    alcohol = st.sidebar.slider('Alcohol', 11.0, 14.8, 13.0)
    malic_acid = st.sidebar.slider('Malic Acid', 0.74, 5.8, 2.34)
    ash = st.sidebar.slider('Ash', 1.36, 3.23, 2.36)
    alcalinity_of_ash = st.sidebar.slider("Alcalinity of ash", 10.6, 30.0, 19.5)
    magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0, 99.7)
    total_phenols = st.sidebar.slider('Total Phenols', 0.98, 3.88, 2.29)
    flavanoids = st.sidebar.slider('Flavanoids:', 0.34, 5.08, 2.03)
    nonflavanoid_phenols = st.sidebar.slider('Nonflavanoid Phenols:', 0.13, 0.66, 0.36)
    proanthocyanins = st.sidebar.slider('Proanthocyanins', 0.41, 3.58, 1.59)
    color_intensity = st.sidebar.slider('Colour Intensity', 1.3, 13.0, 5.1)
    hue = st.sidebar.slider('Hue', 0.48, 1.71, 0.96)
    od280_od315_of_diluted_wines = st.sidebar.slider('OD280/OD315 of diluted wines', 1.27, 4.0, 2.61)
    proline = st.sidebar.slider('proline', 0.98, 3.88, 2.29)

    data = {
        'alcohol': alcohol,
        'malic_acid': malic_acid,
        'ash': ash,
        "alcalinity_of_ash" : alcalinity_of_ash,
        'magnesium': magnesium,
        'total_phenols': total_phenols,
        'flavanoids': flavanoids,
        'nonflavanoid_phenols': nonflavanoid_phenols,
        'proanthocyanins': proanthocyanins,
        'color_intensity': color_intensity,
        'od280/od315_of_diluted_wines': od280_od315_of_diluted_wines,
        'hue': hue,
        'proline': proline,

    }

    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader('Chosen parameters')
st.write(df)

#Get wine datasets from sklearn datasets
wine = datasets.load_wine()
X = wine.data
Y = wine.target

#Random Forest Classifier - prediction of class of the wine and probability of prediction
rfc = RandomForestClassifier()
rfc.fit(X,Y)
class_prediction = rfc.predict(df)
prediction_probability = rfc.predict_proba(df)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Class labels:")
    st.write(wine.target_names)

    st.subheader("Prediction:")
    st.write(wine.target_names[class_prediction])

    st.subheader('Prediction Probability:')
    st.write(prediction_probability)

with col2:
    st.image("https://cdn.pixabay.com/photo/2015/11/07/12/00/alcohol-1031713_960_720.png", width = 250)