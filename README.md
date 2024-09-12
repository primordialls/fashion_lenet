## Report by Assaf Cohen-Arazi and Elia Döhler

# Training Each Model Setting

1.	Base Model (No Dropout, No Batchnorm, No Weight Decay):
•	To train the base model:

```
lenet1 = Lenet5(dropoutp=0, bnorm=False, wdecay=0).to(device)
out[0] = trainLenet(lenet1, iterations, lr, batchsize)
torch.save(lenet1.state_dict(), "model0.pt")
```

2.	Model with Dropout (0.2):
•	To train the model with dropout:

```
lenet2 = Lenet5(dropoutp=0.2, bnorm=False, wdecay=0).to(device)
out[1] = trainLenet(lenet2, iterations, lr, batchsize)
torch.save(lenet2.state_dict(), "model1.pt")
```

3.	Model with Batch Normalization:
•	To train the model with batch normalization:

```
lenet3 = Lenet5(dropoutp=0, bnorm=True, wdecay=0).to(device)
out[2] = trainLenet(lenet3, iterations, lr, batchsize)
torch.save(lenet3.state_dict(), "model2.pt")
```

4.	Model with Weight Decay (1e-5):
•	To train the model with weight decay:

```
lenet4 = Lenet5(dropoutp=0, bnorm=False, wdecay=1e-4).to(device)
out[3] = trainLenet(lenet4, iterations, lr, batchsize)
torch.save(lenet4.state_dict(), "model3.pt")
```


# Testing the Model with Saved Weights

To test any of the models with saved weights, load the model and use the probe function:

	 model_test = Lenet5(dropoutp=0, bnorm=False, wdecay=1e-5).to(device)
	 model_test.load_state_dict(torch.load("model3.pt", map_location=device))
	 probe(model_test)

# Graphs: Train and Test Accuracy

You should plot training and testing accuracy for each model together in the same plot. The provided code already does this:

	 for i in range(4):
  	 	 plt.figure()
  	 	 plt.plot(out[i][2], out[i][0], label="Training Accuracy")
  	 	 plt.plot(out[i][2], out[i][1], label="Testing Accuracy")
  	 	 plt.xlabel("Iterations")
  	 	 plt.ylabel("Accuracy")
  	 	 plt.title(f"Accuracy vs. Iteration of Model: {modelnames[i]}")
  	 	 plt.legend(loc="lower right")

# Dropout Training Accuracy

For dropout, the training accuracy must be measured without dropout during evaluation. This is handled by setting model.eval() during evaluation, which ensures dropout is disabled.

LeNet-5 Architecture Modifications

LeNet-5 is originally used for MNIST (32x32 grayscale images), so it does not fit FashionMNIST, which has 28x28 grayscale images. The modified architecture is:

	1.	Conv Layer 1: 1 input channel, 6 output channels, kernel size 5x5.
	2.	Pooling Layer 1: Average pooling, 2x2 kernel.
	3.	Conv Layer 2: 6 input channels, 16 output channels, kernel size 5x5.
	4.	Pooling Layer 2: Average pooling, 2x2 kernel.
	5.	Conv Layer 3: 16 input channels, 120 output channels, kernel size 4x4 (scaled from the original 5x5).
	6.	Fully Connected Layer 1: 120 input features, 84 output features.
	7.	Fully Connected Layer 2: 84 input features, 10 output classes.

Activation function: tanh

Training Hyperparameters

	1.	Batch Size: Set to 128.
	2.	Learning Rate: Set to 0.001.
	3.	Optimizer: Adam optimizer with weight decay (where specified).

# Regularization Strategies

	1.	Dropout: Applied after pooling layers with a probability of 0.2.
	2.	Batch Normalization: Used to normalize the activations after each convolution.
	3.	Weight Decay: Set to 1e-4 for regularization, implemented via Adam’s weight decay parameter.
