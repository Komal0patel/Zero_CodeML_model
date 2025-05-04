from pymongo import MongoClient
from hashlib import sha256

# Connect to MongoDB
MONGO_URI = "mongodb+srv://komal0mallaram:Qwerty%401234@cluster0.lhfjf0c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["New_DB"]
users_collection = db["Users"]

# Hashing password
def hash_password(password):
    return sha256(password.encode()).hexdigest()

def signup_user(user_email, password):
    if users_collection.find_one({"user_email": user_email}):
        return False, "User already exists."
    users_collection.insert_one({
        "user_email": user_email,
        "password": hash_password(password),
        "actions": []
    })
    return True, "Signup successful!"

def login_user(user_email, password):
    user = users_collection.find_one({"user_email": user_email})
    if not user:
        return False, "User not found."
    if user["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful!"
