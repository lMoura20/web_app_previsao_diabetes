import pandas as pd
import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

#título
st.markdown("<h1 style='text-align: center; color: black;'>Prevendo Diabetes</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: black;'>Programa de demonstração desenvolvido por:</h3>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Lincoln Moura e Greici Capellari</h3>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: green;'>Esta aplicação de inteligência artificial tem como objetivo fornecer uma ferramente de análise preditiva para auxílio a tomada de decisão dos profissionais. No lado esquerdo da tela, insira as variáveis referente as informações clínicas do paciente e verifique o resultado do modelo.</h5>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: green;'>Quando resultado for 1, o paciente possui alta probabilidade para desenvolver diabetes e para resultado 0 não possui. Você também pode verificar acurácia da predição de diabetes para cada modelo.</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: green;'>Você também pode verificar acurácia da predição de diabetes para cada modelo. As figuras apresentam alguns aspectos das variáveis utilizadas para o treinamento do modelo. </h5>", unsafe_allow_html=True)

#dataset
df = pd.read_csv("diabetes.csv")

#cabeçalho
st.markdown("<h3 style='text-align: left; color: black;'>Resultado da Previsão:</h3>", unsafe_allow_html=True)

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
st.subheader('Acurácia dos modelos:')

#Criando as colunas para cada modelo
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
col1.metric('Tree', str(np.around(metrics.accuracy_score(y_test, dtc.predict(x_text))*100,2))+' %')
#acurácia do modelo
col2.metric('SVC', str(np.around(metrics.accuracy_score(y_test, model_svc.predict(x_text))*100,2))+' %')
#acurácia do modelo
col3.metric('KNN', str(np.around(metrics.accuracy_score(y_test, model_knn.predict(x_text))*100,2))+' %')
#acurácia do modelo
col4.metric('RANDOM FOREST', str(np.around(metrics.accuracy_score(y_test, model_rforest.predict(x_text))*100,2))+' %')

#grafico
st.subheader('Composição dos dados utilizados para a previsão.')
user_input_variables.rename(index={0: 'valor'}, inplace= True)
fig = px.pie(user_input_variables.T, values='valor', names=user_input_variables.T.index)
st.plotly_chart(fig)

fig1 = px.bar(df, x='Age', y='Glucose',
              color='Outcome')
fig1.update_layout(showlegend=False,
                   title="Relação Idade x Glicose para determinação de Diabete",
                   title_x=0.5,
                   xaxis_title='Idade',
                   yaxis_title='Glicose')
st.plotly_chart(fig1)

fig2 = px.bar(df, x='Age', y='BMI',
              color='Outcome')
fig2.update_layout(showlegend=False,
                   title="Relação Idade x BMI para determinação de Diabete",
                   title_x=0.5,
                   xaxis_title='Idade',
                   yaxis_title='BMI')
st.plotly_chart(fig2)

col1,col2 = st.columns(2)
fig3 = px.pie(df, values='Age', names='Outcome', title='Proporção da Idade por Diabetes',width=380, height=400)
col1.plotly_chart(fig3)

fig4 = px.pie(df, values='BloodPressure', names='Outcome', title='Proporção da Pressão Sanguínea por Diabetes',width=380, height=400)
col2.plotly_chart(fig3)

fig5 = px.scatter(df, x="Age", y="BMI", marginal_y="violin",
                 marginal_x="box", trendline="ols",color="Outcome", template="simple_white")
st.plotly_chart(fig5)

fig5 = px.scatter(df, x="Age", y="BloodPressure",color="Outcome",  marginal_y="violin",
                  marginal_x="box", trendline="ols", template="simple_white")
st.plotly_chart(fig5)

fig6 = px.scatter_matrix(df, color="Outcome",width=800, height=800)
st.plotly_chart(fig6)

fig7 = px.bar_polar(df, r="Insulin", theta="Age", color="Outcome", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r,width=800, height=800)
st.plotly_chart(fig7)
