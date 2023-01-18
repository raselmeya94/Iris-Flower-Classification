# -*- coding: utf-8 -*-
"""
Spyder Editor

Author: Rasel Meya
"""

import numpy as np
import streamlit as st
import pickle

dt=pickle.load(open('C:/Users/rasel/.spyder-py3/Iiris_data/dt_model.sav','rb'))
log=pickle.load(open('C:/Users/rasel/.spyder-py3/Iiris_data/log_model.sav','rb'))
svm=pickle.load(open('C:/Users/rasel/.spyder-py3/Iiris_data/svc_model.sav','rb'))
kn=pickle.load(open('C:/Users/rasel/.spyder-py3/Iiris_data/kn_model.sav','rb'))

def main():
    st.title("")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision Tree','Logistic Regression','Support Vector Machine' , "K-Nearest Neighbor"]
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    sl=st.slider('Select Sepal Length', 0.0, 10.0)
    sw=st.slider('Select Sepal Width', 0.0, 10.0)
    pl=st.slider('Select Petal Length', 0.0, 10.0)
    pw=st.slider('Select Petal Width', 0.0, 10.0)
    
    inputs=[[sl,sw,pl,pw]]
    #print(inputs)
    if st.button('Classify'):
        if option=='Decision Tree':
            st.success(*dt.predict(inputs))
        elif option=='Logistic Regression':
            st.success(*log.predict(inputs))
        elif option=='K-Nearest Neighbor':
            st.success(*kn.predict(inputs))
        else:
            st.success(*svm.predict(inputs))


if __name__=='__main__':
    main()