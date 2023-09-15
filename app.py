from flask import Flask, jsonify, request, render_template
import pandas as pd
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  

app = Flask(__name__)

# PAGINAS HTML
@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('algar/dashboard.html')

@app.route('/analytics', methods=['GET'])
def analytics():
    # Carregue o arquivo CSV em um DataFrame
    file_path = './database/base_clientes.csv'
    data = pd.read_csv(file_path, index_col=None)

    # Deleta a coluna bom pagador
    data = data.drop('default', axis=1)

    # Seleciona apenas as primeiras linhas
    data = data.head(246)

    # Converte em um json
    data_dict = data.to_dict(orient='records')
    return render_template('algar/analytics.html', data=data_dict)

@app.route('/table', methods=['GET'])
def table():
    # Carregue o arquivo CSV em um DataFrame
    file_path = './database/contratos_pendentes.csv'
    data = pd.read_csv(file_path, index_col=None)

    # Converte em um json
    data_dict = data.to_dict(orient='records')
    return render_template('algar/table.html', data=data_dict)






# MACHINE LEARNING
def update(id, new_prediction):

    # Carregue o arquivo CSV em um DataFrame
    file_path = './database/base_clientes.csv'
    data = pd.read_csv(file_path, index_col=None)

    id_number = int(id)

    # Atualiza o arquivo csv
    data.loc[data['id'] == id_number, 'predict'] = new_prediction

    # Salva o DataFrame atualizado 
    data.to_csv(file_path, index=False)
    


@app.route('/predict', methods=['POST'])
def predict():

    ##### MODELO #####

    # Carregar os dados
    data = pd.read_csv('./database/base_clientes.csv')

    # Dividir os dados em treinamento e teste
    X = data.drop('default', axis=1)
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir colunas categóricas e numéricas (como mencionado anteriormente)

    categorical_cols = [
        'sexo', 
        'escolaridade', 
        'estado_civil', 
        'salario_anual', 
        'tipo_cartao'
    ]
    numeric_cols = [
        'idade', 
        'dependentes'
    ]

    # Definir transformadores para colunas categóricas e numéricas
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combina os transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Definir o pipeline completo com pré-processamento e modelo
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                            ])

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Fazer previsões de probabilidades
    y_prob = model.predict_proba(X_test)


    ##### CARREGAR DADOS FRONT #####

    form = request.json  

    # Criar um DataFrame para o novo cliente
    user = pd.DataFrame({
        'idade': [form.get('idade', '')],
        'sexo': [form.get('sexo', '')],
        'dependentes': [form.get('dependentes', '')],
        'escolaridade': [form.get('escolaridade', '')],
        'estado_civil': [form.get('estado_civil', '')],
        'salario_anual': [form.get('salario_anual', '')],
        'tipo_cartao': [form.get('tipo_cartao', '')]
    })


    # Fazer previsão para o novo cliente
    new_prediction = model.predict(user)

    # Seleciona o id do usuario para atualizar a base de dados
    id = form.get('id', '')
    prediction = new_prediction[0]

    update(id, prediction)

    resultado = {"status": "sucesso", "mensagem": "Processamento bem-sucedido"}

    # Retorne uma resposta em formato JSON com o status
    return jsonify(resultado)




if __name__ == '__main__':
    app.run(debug=False, port=8163)