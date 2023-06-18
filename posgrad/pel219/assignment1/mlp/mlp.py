import pprint

import numpy as np
from sklearn.preprocessing import OneHotEncoder


class MLP:

    def __init__(self, X, y, num_hidden, learning_rate):
        # Inicialização de variáveis
        self.X = X
        self.y = y
        self.learning_rate = learning_rate

        # One-hot encoding de y
        one_hot = OneHotEncoder(sparse_output=False)
        self.y_encoded = one_hot.fit_transform(y.values.reshape(-1, 1))

        # Inicialização de pesos e bias
        layers = [X.shape[1]] + num_hidden + [self.y_encoded.shape[1]]
        self.weights = [np.random.rand(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.random.rand(n) for n in layers[1:]]

    def sigmoid(self, x):
        # Função de ativação sigmoid
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivada da função sigmoid
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward(self, X):
        # Passo forward do MLP
        z_values = []
        a_values = [X]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(X, w) + b
            a = self.sigmoid(z)
            z_values.append(z)
            a_values.append(a)
            X = a

        return z_values, a_values

    def backward(self, z_values, a_values):
        # Passo backward do MLP
        delta_w = [np.zeros(w.shape) for w in self.weights]
        delta_b = [np.zeros(b.shape) for b in self.biases]

        error = self.y_encoded - a_values[-1]
        delta = error * self.sigmoid_derivative(z_values[-1])

        delta_w[-1] = np.dot(a_values[-2].T, delta)
        delta_b[-1] = np.sum(delta, axis=0)

        for l in range(2, len(self.weights) + 1):
            delta = np.dot(delta, self.weights[-l+1].T) * self.sigmoid_derivative(z_values[-l])
            delta_w[-l] = np.dot(a_values[-l-1].T, delta)
            delta_b[-l] = np.sum(delta, axis=0)

        return delta_w, delta_b

    def update_weights(self, delta_w, delta_b):
        # Atualização de pesos e bias
        self.weights = [w + self.learning_rate * dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [b + self.learning_rate * db for b, db in zip(self.biases, delta_b)]

    def train(self, epochs):
        # Treinamento do MLP
        for _ in range(epochs):
            z_values, a_values = self.forward(self.X)
            delta_w, delta_b = self.backward(z_values, a_values)
            self.update_weights(delta_w, delta_b)

            # Cálculo da perda
            loss = np.mean(np.abs(self.y_encoded - a_values[-1]))

            # Condição de parada
            if loss < 1e-5:
                break

    def predict(self, X, y):
        # Previsão do MLP
        _, a_values = self.forward(X)
        predictions = np.argmax(a_values[-1], axis=1)

        # Cálculo da acurácia
        accuracy = np.sum(predictions == y) / len(y)
        print(f'Acurácia de teste: {accuracy * 100:.2f}%')

        return predictions

    def print_summary(self):
        summary = {
            "Número de amostras": self.X.shape[0],
            "Número de features": self.X.shape[1],
            "Número de classes": self.y_encoded.shape[1],
            "Camada de entrada": self.X.shape[1],
            "Camada de saída": self.y_encoded.shape[1],
            "Número de camadas ocultas": len(self.weights) - 1,
            "Número de neurônios em cada camada oculta": [self.weights[i].shape[0] for i in range(len(self.weights) - 1)]
        }

        print('*************Resumo do MLP*************')
        pprint.pprint(summary)
        print('*************Fim do Resumo*************')