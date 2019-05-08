import matplotlib.pyplot as plt
class plot:
    def __init__(self,errorTrain,errorValidation=None,type='MSE'):
        plt.ticklabel_format(style="plain")

        print(type+" train")
        print(errorTrain[-1])
        plt.plot(range(len(errorTrain)), errorTrain, color='green',label='Train')

        if(errorValidation is not None):
            print(type+" validation")
            print(errorValidation[-1])
            plt.plot(range(len(errorValidation)), errorValidation,'--', color='red',label='Test')
            if type=='Accuracy %':
                plt.legend(loc='lower right', fontsize=20)
            else:
                plt.legend(loc='upper right',fontsize=20)
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel(type, fontsize=16)
        plt.ticklabel_format(style="plain")
        plt.savefig("err")
        plt.show()


