import streamlit as st
import re
import nltk
import joblib
import pandas as pd
import numpy as np 
import json
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

REMOVE_SPECIAL_CHARACTER = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))
punctuation = list(punctuation)
STOPWORDS.update(punctuation)

lemmatizer = WordNetLemmatizer()

# Configuración de página
st.set_page_config(page_title="Classification Deploy", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")
st.markdown('<style>body { background-color: white; }</style>', unsafe_allow_html=True)

# Imagen de portada
image = "https://cdn.britannica.com/25/93825-050-D1300547/collection-newspapers.jpg"
image2 ="https://makemynewspaper.com/pub/media/wysiwyg/ethnic-newspaper-main.jpg"
# Título y descripción

titulo = "Machine Learning Classification Deploy"
st.title(titulo)
st.markdown('<style>h1{color: white;}</style>', unsafe_allow_html=True)

# Agrega un cuadro de entrada para el heading
heading = st.text_input("Enter a heading for the description")

descripcion = """In this example, the intention is for the user to enter a brief description of some news, and after pressing the button, predict the category of the same using an artificial intelligence model trained for this purpose."""
st.markdown(f'<div style="color: white;">{descripcion}</div>', unsafe_allow_html=True)



# Cuadro de entrada de texto
escribe = "Type the description of the new"
st.markdown(f'<p style="color: white;">{escribe}</p>', unsafe_allow_html=True)
texto = st.text_area('Type the text', height=200)

# Estilo para el botón
boton_estilo = """ 
    <style> 
    div.stButton > button:first-child { 
        background-color: #4CAF50; 
        color: white; 
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer; 
    } 
    </style> 
"""

# Estilo para el recuadro de resultado
resultado_estilo = """
    <style>
    div.stMarkdown.stMarkdown-756 {
        color: white !important;
    }
    </style>
"""

#Tratar datos


def predict(data):
    clf = joblib.load("forest_clf_grid_search.pkl")
    tf_idf_vect = vect()
    data_prepared = tf_idf_vect.transform(data)

    return clf.predict(data_prepared)

def vect():

    list_dic = []

    with open('dataset/News_Category_Dataset_v3.json') as json_file:
        for the_file in json_file:
            list_dic.append(json.loads(the_file))

    len(list_dic)

    data = pd.DataFrame(list_dic)

    for i in range (0,209527):
        if(len(word_tokenize(data['short_description'][i])) == 0):
            data['short_description'][i] = np.nan
    
    data = data.dropna()
    data=data.reset_index(drop=True)

    data['short_description'] = data['headline'] + '. ' + data['short_description']
    data = data.drop(['link', 'headline','authors', 'date'], axis=1)

    data.loc[data['category']=='PARENTING', 'category'] = 'PARENTS'
    data.loc[data['category']=='THE WORLDPOST', 'category'] = 'WORLD NEWS'
    data.loc[data['category']=='BLACK VOICES', 'category'] = 'VOICES'
    data.loc[data['category']=='QUEER VOICES', 'category'] = 'VOICES'
    data.loc[data['category']=='LATINO VOICES', 'category'] = 'VOICES'
    data.loc[data['category']=='WEDDINGS', 'category'] = 'WEDDINGS & DIVORCE'
    data.loc[data['category']=='DIVORCE', 'category'] = 'WEDDINGS & DIVORCE'
    data.loc[data['category']=='HEALTHY LIVING', 'category'] = 'WELLNESS'
    data.loc[data['category']=='ARTS', 'category'] = 'ARTS & CULTURE'
    data.loc[data['category']=='CULTURE & ARTS', 'category'] = 'ARTS & CULTURE'

    clases = data['category'].unique()

    for i in clases:
        indexes = data.index[data['category'] == i].tolist()
        drop_i = indexes[:(len(indexes)-len(indexes)//20)]
        if (len(drop_i) > 0):
            data.drop(drop_i, axis=0, inplace=True)
            data=data.reset_index(drop=True)

    tf_idf_vect = TfidfVectorizer(ngram_range =(1,1))

    text = data['short_description'].apply(clean_text)

    return tf_idf_vect.fit(text)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # part 1
    text = text.lower() # lowering text
    text = REMOVE_SPECIAL_CHARACTER.sub('', text) # replace REPLACE_BY_SPACE symbols by space in text
    text = BAD_SYMBOLS.sub('', text) # delete symbols which are in BAD_SYMBOLS from text
    
    # part 2
    clean_text = []
    for w in word_tokenize(text):
        if w.lower() not in STOPWORDS:
            pos = pos_tag([w])
            new_w = lemmatizer.lemmatize(w, pos=get_simple_pos(pos[0][1]))
            clean_text.append(new_w)
    text = " ".join(clean_text)
    
    return text


# Botón para analizar el texto
if st.button("Analizar texto"):
    # Cargar los datos
    data =pd.DataFrame( {'new': [heading],'des': [texto]})

    #features = pd.DataFrame(data, index=[0])

    #data_prepared = full_pipeline.transform(features)

    #result = predict(data_prepared)
    #st.text(result[0])


    resultado = predict(data)
    st.markdown(boton_estilo, unsafe_allow_html=True)
    st.markdown(resultado_estilo, unsafe_allow_html=True)
    st.markdown(f'<div style="background-color:#000000;padding: 10px;color: white;">{resultado[0]}', unsafe_allow_html=True)




# Agrega un contenedor <div> para aplicar el fondo solo a este contenedor
with st.container():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2015/06/20/07/24/color-815546_1280.png");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True)
     