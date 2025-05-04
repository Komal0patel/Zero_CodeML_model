# db.py
import pymongo
from datetime import datetime

MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["New_DB"]
users_col = db["Users"]
logs_col = db["logs"]

def log_user_action(user_email, action, details=None):
    users_col.update_one(
        {"user_email": user_email},
        {
            "$push": {
                "actions": {
                    "action": action,
                    "details": details or {},
                    "timestamp": datetime.now()
                }
            }
        }
    )


def create_user(user_email, password):
    if users_col.find_one({"user_email": user_email}):
        return False  # Already exists
    users_col.insert_one({"user_email": user_email, "password": password})
    return True

def authenticate_user(user_email, password):
    user = users_col.find_one({"user_email": user_email, "password": password})
    return user is not None
