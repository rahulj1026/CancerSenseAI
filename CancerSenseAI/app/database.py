import mysql.connector
from mysql.connector import Error
import bcrypt
import re
import json

class Database:
    def __init__(self):
        self.conn = None
        self.cursor = None
        try:
            self.conn = mysql.connector.connect(
                host="localhost",
                user="root",  # Replace with your MySQL username
                password="root",  # Replace with your MySQL password
                database="cancer_prediction_db",
                port=3309
            )
            if self.conn.is_connected():
                self.cursor = self.conn.cursor(buffered=True)
            else:
                raise Error("Failed to connect to database")
        except Error as e:
            raise Exception(f"Error connecting to MySQL: {e}")

    def ensure_connection(self):
        """Ensure the connection is active, reconnect if necessary"""
        try:
            if not self.conn or not self.conn.is_connected():
                self.__init__()
            if not self.cursor:
                self.cursor = self.conn.cursor(buffered=True)
        except Error as e:
            raise Exception(f"Database connection error: {e}")

    def validate_email(self, email):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return re.match(pattern, email) is not None

    def validate_password(self, password):
        # At least 8 characters, 1 uppercase, 1 lowercase, 1 number
        if (len(password) < 8 or not re.search(r'[A-Z]', password) or 
            not re.search(r'[a-z]', password) or not re.search(r'[0-9]', password)):
            return False
        return True

    def register_user(self, username, email, password):
        try:
            self.ensure_connection()
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            if not self.validate_password(password):
                return False, "Password must be at least 8 characters and contain uppercase, lowercase, and numbers"

            # Check if username or email already exists
            self.cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if self.cursor.fetchone():
                return False, "Username or email already exists"

            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Insert new user
            sql = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, (username, email, hashed_password))
            self.conn.commit()
            return True, "Registration successful"
        except Error as e:
            return False, f"Error: {str(e)}"

    def login_user(self, username, password):
        try:
            self.ensure_connection()
            self.cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = self.cursor.fetchone()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
                return True, "Login successful"
            return False, "Invalid username or password"
        except Error as e:
            return False, f"Database error: {str(e)}"

    def get_user_history(self, user_id):
        """Retrieve prediction history for a user"""
        try:
            self.ensure_connection()
            query = """
                SELECT id, prediction, confidence_benign, confidence_malicious, 
                       input_data, notes, timestamp 
                FROM prediction_history 
                WHERE user_id = %s 
                ORDER BY timestamp DESC
            """
            self.cursor.execute(query, (user_id,))
            return self.cursor.fetchall()
        except Error as e:
            print(f"Error retrieving history: {str(e)}")
            return []

    def save_prediction(self, user_id, prediction, confidence_benign, confidence_malicious, input_data, notes):
        """Save a prediction to the database"""
        try:
            self.ensure_connection()
            
            # Debug prints
            print(f"Saving prediction for user_id: {user_id}")
            print(f"Prediction: {prediction}")
            print(f"Confidence values: {confidence_benign}, {confidence_malicious}")
            
            # Convert numpy float64 to Python float
            confidence_benign = float(confidence_benign)
            confidence_malicious = float(confidence_malicious)
            
            # Convert all input_data values to Python float
            input_data = {k: float(v) for k, v in input_data.items()}
            
            query = """
                INSERT INTO prediction_history 
                (user_id, prediction, confidence_benign, confidence_malicious, input_data, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            params = (user_id, prediction, confidence_benign, confidence_malicious, json.dumps(input_data), notes)
            print(f"Query params: {params}")
            
            self.cursor.execute(query, params)
            self.conn.commit()
            return True, "Prediction saved successfully"
        except Error as e:
            print(f"Database error: {str(e)}")
            return False, f"Error saving prediction: {str(e)}"

    def get_user_id(self, username):
        """Get user ID from username"""
        try:
            self.ensure_connection()
            query = "SELECT id FROM users WHERE username = %s"
            self.cursor.execute(query, (username,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Error as e:
            print(f"Error getting user ID: {str(e)}")
            return None

    def __del__(self):
        if hasattr(self, 'conn') and self.conn.is_connected():
            self.cursor.close()
            self.conn.close() 