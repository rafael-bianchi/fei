#!/bin/bash

# Verifica se o pacote virtualenv está instalado
if ! pip3 show virtualenv > /dev/null; then
    echo "O pacote virtualenv não está instalado. Instalando..."
    pip3 install virtualenv
fi

# Gera um identificador aleatório para o ambiente virtual
ENV_ID=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 4 ; echo '')
VENV_NAME="tempenv_$ENV_ID"

# Cria e ativa o ambiente virtual
python3 -m virtualenv $VENV_NAME
source $VENV_NAME/bin/activate

# Instala as dependências a partir do arquivo requirements.txt
pip install -r requirements.txt

# Executa o arquivo main.py com os argumentos especificados
python main.py --return_X_y True --as_frame True --num_hidden 1 --learning_rate 0.01 --test_size 0.2 --random_state 42 --epochs 1000

# Desativa e remove o ambiente virtual
deactivate
rm -rf $VENV_NAME
