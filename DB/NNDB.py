from pymongo import MongoClient


# database per salvare i risultati della gri search
class Database:
    def __init__(self):
        myclient = MongoClient('localhost', 27017,maxPoolSize=None)
        mydb = myclient["mydatabase"]
        self.mycol = mydb["grid"]

    def insert(self, trainErr, validErr, numhidden, learningRate, alfa, lamb):
        mydict = {"trainErr": trainErr, "validErr": validErr, "numhidden": numhidden, "learningRate": learningRate,
                  "alfa": alfa, "lamb": lamb}
        self.mycol.insert_one(mydict)

    # come risultati migliori passiamo i primi k con il validation error più basso
    def best(self, num):
        return self.mycol.find().sort("validErr").limit(num)
