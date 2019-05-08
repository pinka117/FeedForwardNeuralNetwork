import numpy as np
from sklearn.utils import shuffle

from src import NeuralNetwork as nnBatch
from src.DB import NNDB as dbmongo


def kFold(k, input, output, numhidden, learningRate, alfa, lamb, maxiter=1000, type="MSE"):
    db = dbmongo.Database()
    input, output = shuffle(input, output, random_state=0)
    nn = nnBatch.NeuralNetwork(input, output, numhidden=numhidden, learningRate=learningRate, lamb=lamb, alfa=alfa,
                               type=type)

    # divido il dataset in k parti e segno il numero di pattern per ogni partizione
    sizePart = nn.numExamples // k

    errorTrain = np.empty(k)
    errorValidation = np.empty(k)

    for i in range(0, k):
        ##aggiungo una riga di zeri se no non posso concatenere con vstack
        setPass = np.zeros(shape=(1, nn.dimInput))
        setOutPass = np.zeros(shape=(1, nn.dimOutput))
        for j in range(0, k):
            if (i != j):
                # aggiungo la partizione al training set
                setPass = np.vstack((setPass, input[(j * sizePart):(sizePart * (j + 1))][:]))
                setOutPass = np.vstack((setOutPass, output[(j * sizePart):(sizePart * (j + 1))][:]))
            else:
                # passo la partizione alla rete neurale come validation set
                nn.setValidation(input[(i * sizePart):(sizePart * (i + 1))][:],
                                 output[(i * sizePart):(sizePart * (i + 1))][:])

        # rimuovo la linea di zeri iniziali quando passo il training set
        nn.setInput(setPass[1:][:])
        nn.setOutput(setOutPass[1:][:])
        # reinizializzo i pesi
        nn.setWeights()
        nn.train(maxiter)

        errorTrain[i] = nn.errorShow[-1]
        errorValidation[i] = nn.errorValidShow[-1]

    db.insert(np.mean(errorTrain), np.mean(errorValidation), int(numhidden), float(learningRate), float(alfa),
              float(lamb))
