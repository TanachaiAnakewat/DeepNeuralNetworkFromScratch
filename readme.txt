This program was created on Google Colab.


Open this file on Google Colab and "Run all (Ctrl+F9) " to run all cell or press run button on the right side of each cell to run them.


Implement Neural Network as follow:


net = Network()
net.add(FCLayer(28*28, 100))# input_shape=(1, 28*28) output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))# input_shape=(1, 100)   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10)) # input_shape=(1, 50)    output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)
net.fit(x_train, y_train, x_test, y_test, epochs=35, learning_rate=0.1)
net.predict()


This program used the object-oriented program to implement layers and network. The base class Layer is for dealing with input, output, forward and backward methods and will be inherited by other classes such as ActivationLayer and FCLayer(Dense Layer). The class Network includes add method to append layer, use method to set loss function, predict method to predict output for given input, fit method to train the network, and see_hidden_layer to visualize outputs of the hidden layer.