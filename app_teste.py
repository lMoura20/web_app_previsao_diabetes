import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#título
st.markdown("<h1 style='text-align: center; color: black;'>Prevendo Diabetes</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: black;'>Programa de demonstração desenvolvido por:</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Lincoln Moura</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Greici Capellari</h3>", unsafe_allow_html=True)

#dataset
df = pd.read_csv("diabetes.csv")

#cabeçalho
st.write("**Informações dos dados**")

#nomedousuário
user_input = st.sidebar.text_input("Digite seu nome")

#escrevendo o nome do usuário
st.write("Paciente:", user_input)

#dados de entrada
x = df.drop(['Outcome'],1)
y = df['Outcome']

#separa dados em treinamento e teste
x_train, x_text, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_text = scaler.transform(x_text)

#dados dos usuários com a função
def get_user_date():
    pregnancies = st.sidebar.slider("Gravidez",0, 15, 1)
    glicose = st.sidebar.slider("Glicose", 0, 200, 110)
    blood_pressure = st.sidebar.slider("Pressão Sanguínea", 0, 122, 72)
    skin_thickness = st.sidebar.slider("Espessura da pele", 0, 99, 20)
    insulin = st.sidebar.slider("Insulina", 0, 900, 30)
    bni= st.sidebar.slider("Índice de massa corporal", 0.0, 70.0, 15.0)
    dpf = st.sidebar.slider("Histórico familiar de diabetes", 0.0, 3.0, 0.0)
    age = st.sidebar.slider ("Idade", 15, 100, 21)

    #Criação de um dicionário para recebimento dessas informações.
    #dicionário para receber informações

    user_data = {'Gravidez': pregnancies,
                 'Glicose': glicose,
                 'Pressão Sanguínea': blood_pressure,
                 'Espessura da pele': skin_thickness,
                 'Insulina': insulin,
                 'Índice de massa corporal': bni,
                 'Histórico familiar de diabetes': dpf,
                 'Idade': age
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input_variables = get_user_date()

#Modelagem
dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(x_train, y_train)

#Modelo SVC
model_svc = SVC(gamma='auto')
model_svc.fit(x_train, y_train)

#Modelo KNN
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(x_train, y_train)

#Modelo RANDOM FOREST
model_rforest = RandomForestClassifier(random_state=0)
model_rforest.fit(x_train, y_train)

#previsão do resultado
prediction_tree = dtc.predict(user_input_variables)
prediction_SVC = model_svc.predict(user_input_variables)
prediction_KNN = model_knn.predict(user_input_variables)
prediction_rforest = model_rforest.predict(user_input_variables)

#Tabela Resultado
resultado = pd.DataFrame()
resultado['TREE'] = prediction_tree
resultado['SVC'] = prediction_SVC
resultado['KNN'] = prediction_KNN
resultado['RFOREST'] = prediction_rforest
resultado.index= ['resultado']
st.table(resultado)


#acurácia do modelo
st.subheader('Acurácia do modelo')
st.subheader('Tree')
st.write(metrics.accuracy_score(y_test, dtc.predict(x_text))*100)

#acurácia do modelo
st.subheader('SVC')
st.write(metrics.accuracy_score(y_test, model_svc.predict(x_text))*100)

#acurácia do modelo
st.subheader('KNN')
st.write(metrics.accuracy_score(y_test, model_knn.predict(x_text))*100)

#acurácia do modelo
st.subheader('RFOREST')
st.write(metrics.accuracy_score(y_test, model_rforest.predict(x_text))*100)

#grafico

graf = st.bar_chart(user_input_variables)


