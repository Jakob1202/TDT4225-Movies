from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()


class DbConnector:
    """
    Connects to the MongoDB server running in Docker.
    Connector needs HOST, DATABASE, USER and PASSWORD to connect.

    Example:
    HOST = "127.0.0.1" // Your server IP address/domain name
    DATABASE = "movies" // The database the user was created in
    USER = "movies" // This is the user you created and added privileges for
    PASSWORD = "test123" // The password you set for said user
    """

    def __init__(self,
                 DATABASE="movies",
                 HOST="127.0.0.1",
                 USER="movies",
                 PASSWORD=os.environ["MONGO_PASSWORD"]):
        uri = "mongodb://%s:%s@%s:27017/%s" % (USER, PASSWORD, HOST, DATABASE)
        # Connect to the database
        try:
            self.client = MongoClient(uri)
            self.db = self.client[DATABASE]
        except Exception as e:
            print("ERROR: Failed to connect to db:", e)
            raise

        # get database information
        print("You are connected to the database:", self.db.name)
        print("-----------------------------------------------\n")

    def close_connection(self):
        # close the DB connection
        self.client.close()
        print("\n-----------------------------------------------")
        print("Connection to %s-db is closed" % self.db.name)
