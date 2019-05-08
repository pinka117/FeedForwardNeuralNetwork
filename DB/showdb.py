from pymongo import MongoClient

myclient = MongoClient('localhost', 27017)
mydb = myclient["mydatabase"]
mycol = mydb["grid"]

d=mycol.find().sort("validErr")
for row in d:
    print(row)

