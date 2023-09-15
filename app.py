from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # Importe o Pipeline aqui

app = Flask(__name__)

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

@app.route('/random_data', methods=['GET'])
def random_data():
    try:
        # Gerar dados aleatórios para 10 pessoas
        adimplente = random.randint(0, 10)
        inadimplente = 10 - adimplente

        return jsonify({'Adimplente': adimplente, 'Inadimplente': inadimplente})
    except ValueError as e:
        # Em caso de erro, retorne uma mensagem de erro
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)