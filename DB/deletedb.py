from pymongo import MongoClient

# script per cancellare il db
myclient = MongoClient('localhost', 27017)
mydb = myclient["mydatabase"]
mycol = mydb["grid"]

# delete
mycol.delete_many({})
mydoc = mycol.find()
for x in mydoc:
    print(x)
# drop collection
mycol.drop()
