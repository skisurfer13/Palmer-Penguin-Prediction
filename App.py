
#import statements
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from PIL import Image,ImageFilter,ImageEnhance
import os

#start
st.write("""
# Penguin Prediction App
- This app predicts the species of Palmer penguins found in Antarctica using Machine Learning!
- Dataset credits: Dr.Kristen Gorman and Palmer Station, Antarctica LTER and Allison Horst.
- Note: User inputs for features are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of parameters can be changed from the sidebar.
""")

st.sidebar.header('User Input Features')

#inputs
def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()
  
st.subheader('User Input parameters')
st.write(input_df)
        
#read
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)


encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)





load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)



st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

@st.cache
def load_image(img):
    im =Image.open(os.path.join(img))
    return im
#images
if penguins_species[prediction] == 'Chinstrap':
    st.text("Showing Chinstrap Penguin")
    st.image(load_image('chinstrap.jpg'))
elif penguins_species[prediction] == 'Gentoo':
    st.text("Showing Gentoo Penguin")
    st.image(load_image('gentoo.jpg'))
elif penguins_species[prediction] == 'Adelie':
    st.text("Showing Adelie Penguin")
    st.image(load_image('adelie.jpg'))
#imp

st.write("Dataset citation: Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE 9(3):e90081. https://doi.org/10.1371/journal.pone.0090081")                     
st.write("Dataset License: Creative Commons 0")  
st.write("About the dataset: This dataset was created by Dr. Kristen Gorman and members of the Palmer Station, Antarctica (LTER). Palmer is one of the three US Antarctic Stations governed by the Antarctic Treaty of 1959. The Palmer Station is an interdisciplinary polar marine research program established in 1990.")

st.write("The dataset was uploaded by Allison Horst and it is available by CC-0 license in accordance with the Palmer Station LTER Data Policy and the LTER Data Access Policy for Type 1 data. The dataset contains data for 344 penguins. There are 3 diffrent species of penguins in this dataset, collected from 3 islands in the Palmer Archipelago, Antarctica.") 



      
