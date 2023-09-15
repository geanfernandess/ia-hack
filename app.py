from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # Importe o Pipeline aqui

app = Flask(__name__)

list_pendentes = [
    {"Nome": "Tiago Silva", "N° Contrato": 61, "Valor": "R$320.800", "Data": "25/04/2011"},
    {"Nome": "Luciana Oliveira", "N° Contrato": 63, "Valor": "R$170.750", "Data": "25/07/2011"},
    {"Nome": "Felipe Santos", "N° Contrato": 66, "Valor": "R$86.000", "Data": "12/01/2009"},
    {"Nome": "Carla Souza", "N° Contrato": 22, "Valor": "R$433.060", "Data": "29/03/2012"},
    {"Nome": "Marcos Costa", "N° Contrato": 33, "Valor": "R$162.700", "Data": "28/11/2008"},
    {"Nome": "Renata Lima", "N° Contrato": 61, "Valor": "R$372.000", "Data": "02/12/2012"},
    {"Nome": "Eduardo Pereira", "N° Contrato": 59, "Valor": "R$137.500", "Data": "06/08/2012"},
    {"Nome": "Mariana Fernandes", "N° Contrato": 55, "Valor": "R$327.900", "Data": "14/10/2010"},
    {"Nome": "Fernando Alves", "N° Contrato": 39, "Valor": "R$205.500", "Data": "15/09/2009"},
    {"Nome": "Sabrina Barbosa", "N° Contrato": 23, "Valor": "R$103.600", "Data": "13/12/2008"},
    {"Nome": "Ricardo Oliveira", "N° Contrato": 30, "Valor": "R$90.560", "Data": "19/12/2008"},
    {"Nome": "Larissa Costa", "N° Contrato": 22, "Valor": "R$342.000", "Data": "03/03/2013"},
    {"Nome": "Isabela Souza", "N° Contrato": 36, "Valor": "R$470.600", "Data": "16/10/2008"},
    {"Nome": "Roberto Santos", "N° Contrato": 43, "Valor": "R$313.500", "Data": "18/12/2012"},
    {"Nome": "Luisa Fernandes", "N° Contrato": 19, "Valor": "R$385.750", "Data": "17/03/2010"},
    {"Nome": "Pedro Lima", "N° Contrato": 66, "Valor": "R$198.500", "Data": "27/11/2012"},
    {"Nome": "Ana Rodrigues", "N° Contrato": 64, "Valor": "R$725.000", "Data": "09/06/2010"},
    {"Nome": "Fernando Silva", "N° Contrato": 59, "Valor": "R$237.500", "Data": "10/04/2009"}
]

@app.route('/tabela', methods=['GET'])
def tabela():
    return render_template('table.html', data=list_pendentes)


@app.route('/predict', methods=['POST'])
def predict():

    # Carregar os dados
    data = pd.read_csv('base_clientes.csv')

    # Dividir os dados em treinamento e teste
    X = data.drop('default', axis=1)
    y = data['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir colunas categóricas e numéricas (como mencionado anteriormente)

    categorical_cols = ['sexo', 'escolaridade', 'estado_civil', 'salario_anual', 'tipo_cartao']
    numeric_cols = ['idade', 'dependentes', 'meses_de_relacionamento', 'qtd_produtos', 'iteracoes_12m',
                    'meses_inativo_12m', 'limite_credito', 'valor_transacoes_12m', 'qtd_transacoes_12m']

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

    idade = int(request.form['idade'])
    sexo = request.form['sexo']
    dependentes = int(request.form['dependentes'])

    # Criar um DataFrame para o novo cliente
    new_data = pd.DataFrame({
        'idade': [idade],
        'sexo': [sexo],
        'dependentes': [dependentes],
        'escolaridade': ['ensino medio'],
        'estado_civil': ['casado'],
        'salario_anual': ['$60K - $80K'],
        'tipo_cartao': ['blue'],
        'meses_de_relacionamento': [30],
        'qtd_produtos': [4],
        'iteracoes_12m': [2],
        'meses_inativo_12m': [1],
        'limite_credito': [10000],
        'valor_transacoes_12m': [5000],
        'qtd_transacoes_12m': [20]
    })

    # Certifique-se de que as colunas correspondam às do seu conjunto de dados original

    # Fazer previsão para o novo cliente
    new_prediction = model.predict(new_data)

    # Converta a previsão para uma lista Python
    new_prediction_list = new_prediction.tolist()

    # A variável 'new_prediction' agora contém a previsão de inadimplência para o novo cliente (0 para adimplente, 1 para inadimplente)
    print(f'Previsão de Inadimplência: {new_prediction[0]}')

    # Retorne 'new_prediction_list' como JSON
    return jsonify({"Previsão de Inadimplência": new_prediction_list})

import random

@app.route('/teste', methods=['POST'])
def teste():
    data = request.json  # Obtenha os dados da requisição POST

    print(data)

    nome = data.get('nome', '')
    print(nome)
    # Faça qualquer processamento necessário com os dados aqui
    # ...

    # Suponha que o processamento tenha sido bem-sucedido
    resultado = {"status": "sucesso", "mensagem": "Processamento bem-sucedido"}

    # Retorne uma resposta em formato JSON com o status
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)