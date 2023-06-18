import argparse

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlp.mlp import MLP


def main(args):
    # Carregando o dataset Iris
    iris_ds = datasets.load_iris(return_X_y=args.return_X_y, as_frame=args.as_frame)

    # Separando os dados em features (X) e target (y)
    X, y = iris_ds[0], iris_ds[1]

    # Dividindo os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Normalizando os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Inicializando e treinando o modelo MLP
    mlp_model = MLP(X_train, y_train, num_hidden=args.num_hidden, learning_rate=args.learning_rate)
    mlp_model.train(epochs=args.epochs)

    # Fazendo previsões com o modelo treinado
    mlp_model.predict(X_test, y_test)

    # Imprimindo um resumo do modelo
    mlp_model.print_summary()

if __name__ == "__main__":
    # Definindo os argumentos para a linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument("--return_X_y", type=bool, default=True)
    parser.add_argument("--as_frame", type=bool, default=True)
    parser.add_argument("--num_hidden", type=int, nargs="*", default=[1])
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)

    # Coletando os argumentos da linha de comando
    args = parser.parse_args()

    # Executando a função principal
    main(args)
