import pandas as pd
import pickle as pickle
import streamlit as st
import plotly.graph_objects as go
import numpy as np


def get_data_clean():
    data = pd.read_csv("data/data.csv")
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data['diagnosis'] = data['diagnosis'].map({"M": 1, "B": 0})
    #print(data.head())
    return data


def get_scaled_data(input_dict):

    data = get_data_clean()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        X_min = X[key].min()
        X_max = X[key].max()
        X_scaled = (value - X_min)/X_max
        scaled_dict[key] = X_scaled

    #st.write(scaled_dict)
    return scaled_dict

def get_radar_chart(input_dict):
    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness',
                   'Concavity', 'Concave', 'Symetry', 'Fractal dimension' ]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_dict["radius_mean"], input_dict["texture_mean"], input_dict["perimeter_mean"],
           input_dict["area_mean"], input_dict["smoothness_mean"], input_dict["compactness_mean"],
           input_dict["concavity_mean"], input_dict["concave points_mean"], input_dict["symmetry_mean"],
           input_dict["fractal_dimension_mean"]],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_dict["radius_se"], input_dict["texture_se"], input_dict["perimeter_se"],
           input_dict["area_se"], input_dict["smoothness_se"], input_dict["compactness_se"],
           input_dict["concavity_se"], input_dict["concave points_se"], input_dict["symmetry_se"],
           input_dict["fractal_dimension_se"]],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_dict["radius_worst"], input_dict["texture_worst"], input_dict["perimeter_worst"],
           input_dict["area_worst"], input_dict["smoothness_worst"], input_dict["compactness_worst"],
           input_dict["concavity_worst"], input_dict["concave points_worst"], input_dict["symmetry_worst"],
           input_dict["fractal_dimension_worst"]],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )
    return fig

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_data_clean()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    input_dict = {}

    for labels, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            labels,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_prediction(input_dict):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = np.array(list(input_dict.values())).reshape(1, -1)

    scaled_input = scaler.transform(input_array)
    st.subheader("Cell cluster Prediction")
    st.write("The cell closter is :")
    prediction = model.predict(scaled_input)
    if prediction[0] == 0:
        st.write("<span class='class_benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='class_malicious'>Malicious</span>", unsafe_allow_html=True)
    
    st.write("Probability to be benign : ", model.predict_proba(scaled_input)[0][0])
    st.write("Probability to be Malicious : ", model.predict_proba(scaled_input)[0][1])

    st.write("please connect this app to your cytology lab to help dignosticate breast cancer from your tissue sample. this app may not replace professional diagnotic")

    return prediction

def main():
    st.set_page_config(page_title="Breast canser predictor",
                       page_icon = ":female-doctor:",
                       layout="wide",
                       initial_sidebar_state="expanded"
                    )
    with open("assets/style.css", 'r') as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_dict = add_sidebar()
    #st.write(input_dict)

    scaled_dict = get_scaled_data(input_dict)

    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("please connect this app to yout cytologylab to help diagnose breast cancer from your tissue sample. this app predicts using a machine learning model whether a breast mass is benign or malignant based on the mesurements it receives from your cytosislab. you can also updte the mesurements by hand using the slides in the sidebar.")

    col1, col2 = st.columns([4, 1])

    with col1:
        radar_chart = get_radar_chart(scaled_dict)
        st.plotly_chart(radar_chart)
    with col2:
        prediction = get_prediction(input_dict)


if __name__ == '__main__':
    main()