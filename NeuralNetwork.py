import numpy as np

from src.Utils import Utils as ut


class NeuralNetwork:
    def __init__(self, input, output, validationInput=None, validationOutput=None, numhidden=100,
                 learningRate=1, alfa=0, lamb=0, errorMax=0, type="MSE"):
        # salva l'input del train set e da quello ricava il numero di esempi
        self.setInput(input)
        # salva l'output del train set
        self.setOutput(output)
        # salva il validation set

        self.setValidation(validationInput, validationOutput)
        # imposta i delta a zero per la prima iterazione per calcolare il momento
        self.deltaW = 0
        self.deltaW2 = 0
        self.deltaWB = 0
        self.deltaWB2 = 0
        # salva i parametri
        self.type = type
        self.numhidden = numhidden
        self.learningRate = learningRate
        self.errorMax = errorMax
        self.alfa = alfa
        self.lamb = lamb
        # inizializza i pesi
        self.setWeights()

    def setOutput(self, output):
        self.output = output
        self.dimOutput = output.shape[1]

    def setInput(self, input):
        self.input = input
        self.numExamples = input.shape[0]
        self.dimInput = self.input.shape[1]

    def setValidation(self, inputValidation, outputValidation):
        self.validationInput = inputValidation
        self.validationOutput = outputValidation

    def setWeights(self):
        self.weightsInput = (np.random.randn(self.dimInput, self.numhidden)) * 1.4 - 0.7
        self.biasInput = (np.random.randn(1, self.numhidden)) * 1.4 - 0.7

        # pesi hidden layer
        self.weightsHL1 = (np.random.rand(self.numhidden, self.dimOutput)) * 1.4 - 0.7
        self.biasHL1 = (np.random.rand(1, self.dimOutput)) * 1.4 - 0.7

    def feedforward(self):
        self.feed(self.input)

    def feed(self, input):
        self.outputLayer1 = ut.sigmoid(np.dot(input, self.weightsInput) + self.biasInput)
        self.outputNN = ut.sigmoid(np.dot(self.outputLayer1, self.weightsHL1) + self.biasHL1)

    def backpropagation(self):
        error = self.output - self.outputNN
        deltaK = error * ut.sigmoidDerivative(self.outputNN)

        error2 = np.dot(deltaK, self.weightsHL1.T)
        deltaJ = error2 * ut.sigmoidDerivative(self.outputLayer1)

        self.deltaW = self.alfa * self.deltaW + self.learningRate * self.input.T.dot(deltaJ)
        self.deltaWB = self.alfa * self.deltaWB +self.learningRate* np.sum(deltaJ)

        self.deltaW2 = self.alfa * self.deltaW2 +self.learningRate * self.outputLayer1.T.dot(deltaK)
        self.deltaWB2 = self.alfa * self.deltaWB2 + self.learningRate * np.sum(deltaK)

        self.weightsInput += self.deltaW - self.lamb * self.weightsInput
        self.weightsHL1 += self.deltaW2 - self.lamb * self.weightsHL1

        # non uso la regolarizzazione per i bias
        self.biasInput += self.deltaWB
        self.biasHL1 += self.deltaWB2

    def train(self, maxiter):
        self.maxiter = maxiter
        self.errorShow = np.empty(maxiter)
        self.errorValidShow = np.empty(maxiter)
        for i in range(0, maxiter):
            self.feedforward()
            self.backpropagation()
            self.calcError(i)

    def calcError(self, i):
        # calcola l'errore sul training set
        self.feedforward()
        self.errorShow[i] = ut.error(self.outputNN, self.output, self.type)
        # calcola l'errore sul validation set
        if(self.validationInput is not None):
            self.feed(self.validationInput)
            self.errorValidShow[i] = ut.error(self.outputNN, self.validationOutput, self.type)
