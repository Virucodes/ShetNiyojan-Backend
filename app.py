from flask import Flask, request, jsonify, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import json
from db import users_collection, yields_collection,activities_collection
from db import users_collection, yields_collection, activities_collection, db
from groq import Groq
from decouple import config
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from bson import ObjectId
from functools import wraps
import pickle as pkl
import re
from flask_cors import CORS
import os
from bson.objectid import ObjectId
import random
import requests
from urllib.parse import urlencode
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from haversine import haversine
import json
import google.generativeai as genai
import werkzeug.utils
from twilio.rest import Client as TwilioClient

base_dir  = os.path.dirname(os.path.abspath(__file__))

# ------------------ Twilio Configuration ------------------
TWILIO_ACCOUNT_SID = config('TWILIO_ACCOUNT_SID', default=os.getenv('TWILIO_ACCOUNT_SID', ''))
TWILIO_AUTH_TOKEN = config('TWILIO_AUTH_TOKEN', default=os.getenv('TWILIO_AUTH_TOKEN', ''))
TWILIO_VERIFY_SERVICE_SID = config('TWILIO_VERIFY_SERVICE_SID', default=os.getenv('TWILIO_VERIFY_SERVICE_SID', ''))

def get_twilio_client():
    return TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

UPLOAD_FOLDER = "uploads"
PLANT_DISEASE_GROQ_API_KEY = config('GROQ_API_KEY', default=os.getenv('GROQ_API_KEY', '')).strip()
PLANT_DISEASE_GROQ_MODEL = config('PLANT_DISEASE_GROQ_MODEL', default='llama-3.3-70b-versatile').strip()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Set up logging for transport optimizer
transport_logger = logging.getLogger("transport_optimizer")
file_handler = logging.FileHandler("transport_optimizer.log")
file_handler.setLevel(logging.INFO)
transport_logger.addHandler(file_handler)
transport_logger.setLevel(logging.INFO)

# Mandi API Configuration
MANDI_API_BASE_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
MANDI_API_KEY = "579b464db66ec23bdd0000010582a7863e934ee27f4b23c034396353"

# Predefined district coordinates for mapping and distance calculations
district_coordinates = {
    # Maharashtra Districts
    "Pune": (18.5204, 73.8567),
    "Mumbai": (19.0760, 72.8777),
    "Nashik": (19.9975, 73.7898),
    "Nagpur": (21.1458, 79.0882),
    "Kolhapur": (16.7050, 74.2433),
    "Solapur": (17.6599, 75.9064),
    "Aurangabad": (19.8762, 75.3433),
    "Ahmednagar": (19.0948, 74.7480),
    "Satara": (17.6805, 73.9979),
    "Jalgaon": (21.0077, 75.5626),
    # Gujarat Districts
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Rajkot": (22.3039, 70.8022),
    "Anand": (22.5645, 72.9289),
    "Mehsana": (23.6015, 72.3995),
    # Karnataka Districts
    "Bangalore": (12.9716, 77.5946),
    "Belgaum": (15.8497, 74.4977),
    "Dharwad": (15.4589, 75.0078),
    "Mysore": (12.2958, 76.6394),
    "Kolar": (13.1368, 78.1292),
    "Tumkur": (13.3409, 77.1006),
    # Madhya Pradesh Districts
    "Indore": (22.7196, 75.8577),
    "Bhopal": (23.2599, 77.4126),
    "Ujjain": (23.1760, 75.7885),
    "Jabalpur": (23.1815, 79.9864),
    "Dhar": (22.5978, 75.3040),
    "Dewas": (22.9623, 76.0508),
    # Uttar Pradesh Districts
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Agra": (27.1767, 78.0081),
    "Meerut": (28.9845, 77.7064),
    "Hapur": (28.7306, 77.7758)
} 

# Simulated city data with coordinates (latitude, longitude)
city_data = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639)
}

# Map city names to their corresponding states for API filtering
city_to_state = {
    "Mumbai": "Maharashtra",
    "Delhi": "NCT of Delhi",
    "Bangalore": "Karnataka",
    "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal"
}

# ------------------ Token Middleware ------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        print(f"=== Token Validation ===")
        print(f"Received token: {token}")
        
        if not token:
            print("No token provided")
            return jsonify({'error': 'Token is missing'}), 401

        # Try to find user with this token
        user = users_collection.find_one({"token": token})
        print(f"Token lookup result: {'User found' if user else 'No user found'}")
        
        # TEMPORARY WORKAROUND: If no user found with token, create a test user for development
        if not user:
            print("Creating test user for development")
            test_user = {
                "_id": ObjectId(),
                "fullname": "Test User",
                "mobileno": "9999999999",
                "phone": "9999999999",
                "token": token
            }
            # Don't actually insert this user - just use it for the current request
            user = test_user
        
        if user:
            print(f"User details: {user.get('fullname')}, {user.get('mobileno')}")
        
        if not token.strip():
            print("Empty token")
            return jsonify({'error': 'Invalid or expired token'}), 401

        return f(user, *args, **kwargs)
    return decorated

# ------------------ User Registration ------------------
@app.route('/api/users/register', methods=['POST'])
def register():
    data = request.json
    fullname = data.get('fullname')
    mobileno = data.get('mobileno')
    password = data.get('password')

    if not fullname or not mobileno or not password:
        return jsonify({'message': 'All fields are required'}), 400

    if users_collection.find_one({'mobileno': mobileno}):
        return jsonify({'message': 'User already exists'}), 400

    hashed_pw = generate_password_hash(password)
    users_collection.insert_one({
        'fullname': fullname,
        'mobileno': mobileno,
        'password': hashed_pw,
        'token': None
    })

    return jsonify({'message': 'Registration successful'}), 201

# ------------------ User Login ------------------
@app.route('/api/users/login', methods=['POST'])
def login():
    data = request.json
    mobileno = data.get('mobileno')
    password = data.get('password')

    if not mobileno or not password:
        return jsonify({'message': 'Mobile number and password are required'}), 400

    user = users_collection.find_one({'mobileno': mobileno})
    if not user:
        return jsonify({'message': 'Invalid credentials'}), 401

    stored_password = user.get('password', '')
    if not stored_password or not check_password_hash(stored_password, password):
        return jsonify({'message': 'Invalid credentials'}), 401

    token = str(uuid.uuid4())
    users_collection.update_one({'mobileno': mobileno}, {'$set': {'token': token}})

    return jsonify({'token': token}), 200

# ------------------ OTP - Send ------------------
@app.route('/api/otp/send', methods=['POST'])
def send_otp():
    data = request.json
    mobile = data.get('mobile', '').strip()

    if not mobile or len(mobile) != 10 or not mobile.isdigit():
        return jsonify({'message': 'A valid 10-digit mobile number is required'}), 400

    phone_e164 = f'+91{mobile}'  # India country code

    try:
        client = get_twilio_client()
        verification = client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID) \
            .verifications \
            .create(to=phone_e164, channel='sms')
        return jsonify({'message': 'OTP sent successfully', 'status': verification.status}), 200
    except Exception as e:
        print(f'[send_otp] Error: {e}')
        return jsonify({'message': f'Failed to send OTP: {str(e)}'}), 500

# ------------------ OTP - Verify ------------------
@app.route('/api/otp/verify', methods=['POST'])
def verify_otp():
    data = request.json
    mobile = data.get('mobile', '').strip()
    code = data.get('code', '').strip()

    if not mobile or len(mobile) != 10 or not mobile.isdigit():
        return jsonify({'message': 'A valid 10-digit mobile number is required'}), 400
    if not code or len(code) != 6 or not code.isdigit():
        return jsonify({'message': 'A valid 6-digit OTP code is required'}), 400

    phone_e164 = f'+91{mobile}'

    try:
        client = get_twilio_client()
        check = client.verify.v2.services(TWILIO_VERIFY_SERVICE_SID) \
            .verification_checks \
            .create(to=phone_e164, code=code)
        if check.status == 'approved':
            return jsonify({'message': 'OTP verified successfully', 'status': 'approved'}), 200
        else:
            return jsonify({'message': 'Invalid or expired OTP', 'status': check.status}), 400
    except Exception as e:
        print(f'[verify_otp] Error: {e}')
        return jsonify({'message': f'OTP verification failed: {str(e)}'}), 500

# ------------------ User Profile ------------------
@app.route('/api/users/profile', methods=['GET'])
@token_required
def profile(current_user):
    return jsonify({
        'fullname': current_user.get('fullname', ''),
        'mobileno': current_user.get('mobileno', ''),
        'email': current_user.get('email', ''),
        'address': current_user.get('address', ''),
        'total_acres': current_user.get('total_acres', 0),
        'preferred_crops': current_user.get('preferred_crops', []),
    })

@app.route('/api/users/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    data = request.json or {}
    update_fields = {}

    if 'email' in data:
        update_fields['email'] = data['email'].strip()
    if 'address' in data:
        update_fields['address'] = data['address'].strip()
    if 'total_acres' in data:
        try:
            update_fields['total_acres'] = float(data['total_acres'])
        except (ValueError, TypeError):
            return jsonify({'message': 'total_acres must be a number'}), 400
    if 'preferred_crops' in data:
        if isinstance(data['preferred_crops'], list):
            update_fields['preferred_crops'] = data['preferred_crops']
        else:
            return jsonify({'message': 'preferred_crops must be a list'}), 400
    if 'fullname' in data:
        update_fields['fullname'] = data['fullname'].strip()

    if not update_fields:
        return jsonify({'message': 'No fields to update'}), 400

    users_collection.update_one(
        {'mobileno': current_user['mobileno']},
        {'$set': update_fields}
    )

    updated_user = users_collection.find_one({'mobileno': current_user['mobileno']})
    return jsonify({
        'message': 'Profile updated successfully',
        'fullname': updated_user.get('fullname', ''),
        'mobileno': updated_user.get('mobileno', ''),
        'email': updated_user.get('email', ''),
        'address': updated_user.get('address', ''),
        'total_acres': updated_user.get('total_acres', 0),
        'preferred_crops': updated_user.get('preferred_crops', []),
    }), 200

# ------------------ plant-disease-analysis ------------------
@app.route('/api/plant-disease-analysis', methods=['POST'])
def plant_disease_analysis():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    safe_filename = werkzeug.utils.secure_filename(image.filename or "")
    if not safe_filename:
        safe_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    image.save(image_path)

    try:
        from PIL import Image
        import json
        
        # Load environment API Key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            app.logger.error("[plant_disease_analysis] Missing GEMINI_API_KEY environment variable")
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({"error": "Server configuration error: GEMINI_API_KEY is missing"}), 500

        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Initialize Gemini 2.5 Flash model
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """
        You are an expert plant pathologist and agricultural consultant.
        Analyze the attached plant leaf/crop image.
        Identify:
        1. The name of the plant/crop (e.g., Tomato, Wheat, Potato).
        2. Any disease present on the plant, or if it is healthy.
        
        Generate a structured JSON output with these exact keys:
        - "predictedDisease": The name of the predicted disease class (e.g. 'Tomato_Early_blight', 'Tomato_healthy', 'Potato_Late_blight', 'Wheat_Rust', etc.).
        - "confidence": A float confidence score between 0.0 and 1.0 (e.g. 0.95).
        - "analysis": A JSON object containing the detailed analysis with these exact keys:
            - "diseaseName": "Name of the disease"
            - "description": "Scientific and general overview of the disease"
            - "causes": "Possible causes or conditions for this disease"
            - "symptoms": "Visual and physical symptoms to look out for"
            - "recommendations": "Best practices to handle this disease"
            - "doses": "Pesticide/fertilizer doses and frequency if applicable"

        Return ONLY the raw valid JSON block, do not wrap in markdown or anything else.
        """
        
        # Open and load image using PIL context manager to release file lock on Windows
        print("Sending crop health image to Gemini 2.5 Flash...")
        with Image.open(image_path) as img:
            response = model.generate_content([prompt, img])
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini Vision API")
            
        text = response.text.strip()
        
        # Clean up any potential markdown delimiters
        text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()
        
        # Parse JSON to format it appropriately for the frontend
        try:
            data = json.loads(text)
        except Exception as json_err:
            print(f"Error parsing Gemini response text as JSON: {json_err}. Raw text: {text}")
            # Try to extract JSON using regex
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                raise json_err
                
        predicted_disease = data.get("predictedDisease", "Unknown")
        raw_conf = data.get("confidence", 0.95)
        # Convert confidence to percentage formatted to 2 decimals
        confidence = round(raw_conf * 100.0, 2) if raw_conf <= 1.0 else round(raw_conf, 2)
        
        # Ensure the analysis field is a structured dictionary/object
        analysis_val = data.get("analysis", {})
        if isinstance(analysis_val, str):
            try:
                analysis_val = json.loads(analysis_val)
            except Exception:
                analysis_val = {
                    "diseaseName": predicted_disease.replace("_", " "),
                    "description": analysis_val,
                    "causes": "N/A",
                    "symptoms": "N/A",
                    "recommendations": "N/A",
                    "doses": "N/A"
                }
            
        # Clean up the local image file
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return jsonify({
            "predictedDisease": predicted_disease,
            "confidence": confidence,
            "analysis": analysis_val
        }), 200

    except Exception as e:
        app.logger.exception(
            "[plant_disease_analysis] failed filename=%s error=%s",
            getattr(image, "filename", "<unknown>"),
            str(e),
        )
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": f"Failed to analyze image: {str(e)}"}), 500
# ------------------ Crop Recommendation ------------------
@app.route('/api/crop-recommendation', methods=['POST'])
def crop_recommendation():
    REQUIRED_FIELDS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "location"]
    request_id = uuid.uuid4().hex[:8]
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        app.logger.warning(
            "[crop_recommendation] request_id=%s invalid_json content_type=%s raw=%s",
            request_id,
            request.content_type,
            request.data[:200],
        )
        return jsonify({'error': 'Invalid JSON payload', 'request_id': request_id}), 400

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
        or data[field] is None
        or (isinstance(data[field], str) and not data[field].strip())
    ]
    if missing_fields:
        app.logger.warning(
            "[crop_recommendation] request_id=%s missing_fields=%s payload_keys=%s payload=%s",
            request_id,
            missing_fields,
            sorted(list(data.keys())),
            {k: data.get(k) for k in sorted(list(data.keys()))},
        )
        return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}', 'request_id': request_id}), 400

    app.logger.info(
        "[crop_recommendation] request_id=%s payload_keys=%s location=%s",
        request_id,
        sorted(list(data.keys())),
        data.get("location"),
    )

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""
        Given the following agricultural parameters, recommend suitable crops and provide detailed guidance:

        - Nitrogen: {data['N']}
        - Phosphorus: {data['P']}
        - Potassium: {data['K']}
        - Temperature: {data['temperature']}°C
        - pH: {data['ph']}
        - Rainfall: {data['rainfall']} mm
        - Humidity: {data['humidity']}%
        - Location: {data['location']}

        Please provide a response in the following STRICT JSON format and  the json keys should not have whitespace and they should be in camelcase:
        {{
            "Best Recommended Crop": "<single crop name>",
            "Alternative Crops": ["<crop1>", "<crop2>", "<crop3>"],
            "Recommendations": "<brief advice on farming practices>",
            "Estimated Yield": "<numeric value and unit>",
            "Environmental Suitability": "<concise sentence on climate suitability>",
            "Additional Crops": [
                {{
                    "Crop": "<name>",
                    "Suitability Score": "<1 to 100>",
                    "Water Requirement": "<low/medium/high>",
                    "Growth Period": "<e.g. 90-120 days>"
                }},
                {{
                    "Crop": "<name>",
                    "Suitability Score": "<1 to 100>",
                    "Water Requirement": "<low/medium/high>",
                    "Growth Period": "<e.g. 90-120 days>"
                }},
                {{
                    "Crop": "<name>",
                    "Suitability Score": "<1 to 100>",
                    "Water Requirement": "<low/medium/high>",
                    "Growth Period": "<e.g. 90-120 days>"
                }}
            ]
        }}

        Please return only valid JSON. Do not include any other text or explanation.
        """

        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # replace with your available Groq model
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert agricultural assistant providing recommendations for crop planning based on soil and climate data. You must return STRICTLY VALID JSON in the structure requested."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        # Get model output
        response_text = chat_completion.choices[0].message.content.strip()

        # Extract only JSON using regex
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return jsonify({'error': 'Failed to extract JSON from model response'}), 500

        clean_json = json_match.group(0)
        parsed_response = json.loads(clean_json)

        return jsonify(parsed_response), 200

    except Exception as e:
        app.logger.exception(
            "[crop_recommendation] request_id=%s failed with error=%s",
            request_id,
            str(e),
        )
        return jsonify({'error': str(e), 'request_id': request_id}), 500


# ------------------ Yield Management ------------------
@app.route('/api/yields', methods=['GET'])
@token_required
def get_yields(current_user):
    try:
        user_yields = list(yields_collection.find({"userId": current_user["_id"]}))
        
        # Convert ObjectId to string for JSON serialization
        for yield_item in user_yields:
            yield_item['id'] = str(yield_item.pop('_id'))
            if 'userId' in yield_item:
                yield_item['userId'] = str(yield_item['userId'])
            
        return jsonify(user_yields), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/yields/<yield_id>', methods=['GET'])
@token_required
def get_yield(current_user, yield_id):
    try:
        print(f"Fetching yield {yield_id} for user {current_user.get('_id')}")
        
        # Find the yield by ID
        yield_obj = yields_collection.find_one({"_id": ObjectId(yield_id)})
        
        if not yield_obj:
            print(f"Yield {yield_id} not found")
            return jsonify({"status": "error", "message": "Yield not found"}), 404
        
        print(f"Found yield data: {yield_obj}")
        
        # Prepare response by converting ObjectId to string
        yield_data = {k: str(v) if isinstance(v, ObjectId) else v for k, v in yield_obj.items()}
        
        # Ensure status is included in the response
        if 'status' not in yield_data or not yield_data['status']:
            yield_data['status'] = 'Active'  # Default to Active if not set
            
        # Ensure activityStatus is included and matches status
        yield_data['activityStatus'] = yield_data['status']
        
        print(f"Returning yield data with status: {yield_data['status']}")
        
        return jsonify(yield_data), 200
    except Exception as e:
        print(f"Error fetching yield: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/yields', methods=['POST'])
def create_yield():
    try:
        data = request.json
        print("Received yield creation request:", data)
        
        # Validate required fields
        if not data.get('name') or not data.get('acres') or not data.get('mobileno'):
            print("Missing required fields")
            return jsonify({"error": "Name, acres, and mobile number are required"}), 400
            
        # Find user by mobile number
        user = users_collection.find_one({'mobileno': data.get('mobileno')})
        if not user:
            print(f"User not found for mobile number: {data.get('mobileno')}")
            return jsonify({"error": "User not found with the provided mobile number"}), 404
            
        print(f"Found user with ID: {user['_id']}")
            
        # Create yield document with minimal required fields
        new_yield = {
            "name": data.get('name'),
            "acres": float(data.get('acres')),
            "status": "planning",
            "userId": user["_id"],
            "createdAt": datetime.now()
        }
        
        # Insert into database
        result = yields_collection.insert_one(new_yield)
        print(f"Inserted yield with ID: {result.inserted_id}")
        
        # Return the created yield with ID
        created_yield = new_yield.copy()
        created_yield['id'] = str(result.inserted_id)
        created_yield['userId'] = str(created_yield['userId'])  # Convert userId to string for JSON serialization
        if '_id' in created_yield:
            del created_yield['_id']
        
        return jsonify(created_yield), 201
    except Exception as e:
        print(f"Error creating yield: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/yields/<yield_id>', methods=['PUT'])
@token_required
def update_yield(current_user, yield_id):
    try:
        data = request.json
        print(f"Updating yield {yield_id} with data: {data}")
        
        if not yield_id:
            return jsonify({"status": "error", "message": "Yield ID is required"}), 400
        
        # Find the yield by ID - use the correct collection variable
        yield_obj = yields_collection.find_one({"_id": ObjectId(yield_id)})
        
        if not yield_obj:
            print(f"Yield {yield_id} not found")
            return jsonify({"status": "error", "message": "Yield not found"}), 404
        
        print(f"Original yield data: {yield_obj}")
        
        # Check if the yield belongs to the current user
        if str(yield_obj.get('userId')) != str(current_user.get('_id')):
            print(f"Authorization failed: Yield belongs to {yield_obj.get('userId')}, but current user is {current_user.get('_id')}")
            return jsonify({"status": "error", "message": "Unauthorized to update this yield"}), 403
        
        # Process update data
        update_data = {}
        
        # Handle status update specifically
        if 'status' in data:
            update_data['status'] = data['status']
            print(f"Updating yield status to: {data['status']}")
            
            # Also update the associated activityStatus
            update_data['activityStatus'] = data['status']
        
        # Handle other field updates
        allowed_fields = ['name', 'type', 'description', 'daysRemain', 'expense']
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
        
        # If no fields to update, return error
        if not update_data:
            print("No fields to update")
            return jsonify({"status": "error", "message": "No fields to update"}), 400
        
        # Add updatedAt timestamp
        update_data['updatedAt'] = datetime.now()
        
        print(f"Final update data: {update_data}")
        
        # Update the yield - use the correct collection variable
        result = yields_collection.update_one(
            {"_id": ObjectId(yield_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            print(f"Successfully updated yield {yield_id}, modified count: {result.modified_count}")
            # Fetch and return the updated yield data
            updated_yield = yields_collection.find_one({"_id": ObjectId(yield_id)})
            print(f"Updated yield data: {updated_yield}")
            return jsonify({
                "status": "success",
                "message": "Yield updated successfully",
                "data": {
                    "yield_id": yield_id,
                    "updated_fields": list(update_data.keys()),
                    "status": updated_yield.get("status", "Unknown")
                }
            }), 200
        else:
            print(f"Yield not modified. Result: {result}")
            return jsonify({"status": "error", "message": "Yield not modified"}), 400
            
    except Exception as e:
        print(f"Error updating yield: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/yields/<yield_id>', methods=['DELETE'])
@token_required
def delete_yield(current_user, yield_id):
    try:
        # Check if yield exists and belongs to user
        existing_yield = yields_collection.find_one({"_id": ObjectId(yield_id), "userId": current_user["_id"]})
        if not existing_yield:
            return jsonify({"error": "Yield not found or access denied"}), 404
            
        # Delete from database
        yields_collection.delete_one({"_id": ObjectId(yield_id)})
        
        return jsonify({"message": "Yield deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#activities



@app.route('/api/create_activity', methods=['POST'])
def create_activity():
    data = request.get_json()

    if not data or 'activity_type' not in data:
        return jsonify({"error": "Missing 'activity_type'"}), 400

    activity_type = data['activity_type']

    # Required fields per activity type
    if activity_type == 'fertilizer':
        required_fields = ['yield_id', 'mobileno', 'activity_name', 'summary', 'amount', 'fertilizer_name', 'quantity', 'bill_image']
    elif activity_type == 'pesticide':
        required_fields = ['yield_id', 'mobileno', 'activity_name', 'summary', 'amount', 'pesticide_name', 'quantity', 'bill_image']
    elif activity_type == 'financial':
        required_fields = ['yield_id', 'mobileno', 'activity_name', 'summary', 'amount', 'financial_category', 'payment_method', 'receipt']
    else:
        required_fields = ['yield_id', 'mobileno', 'activity_name', 'summary', 'amount']

    # Check for missing fields
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    # Get user by mobile number
    user = users_collection.find_one({'mobileno': data['mobileno']})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Validate yield ID and ownership
    try:
        yield_obj = yields_collection.find_one({
            '_id': ObjectId(data['yield_id']),
            'userId': user['_id']
        })
    except Exception as e:
        return jsonify({"error": f"Invalid yield ID format: {str(e)}"}), 400

    if not yield_obj:
        return jsonify({"error": "Yield not found for this user"}), 404

    # Base activity data
    activity = {
        'userId': user['_id'],
        'yieldId': yield_obj['_id'],
        'activity_type': data['activity_type'],
        'activity_name': data['activity_name'],
        'summary': data['summary'],
        'amount': data['amount'],
        'created_at': datetime.utcnow()
    }

    # Add extra fields
    if activity_type == 'fertilizer':
        activity['fertilizer_name'] = data['fertilizer_name']
        activity['quantity'] = data['quantity']
        activity['bill_image'] = data['bill_image']

    elif activity_type == 'pesticide':
        activity['pesticide_name'] = data['pesticide_name']
        activity['quantity'] = data['quantity']
        activity['bill_image'] = data['bill_image']

    elif activity_type == 'financial':
        activity['financial_category'] = data['financial_category']
        activity['payment_method'] = data['payment_method']
        activity['receipt'] = data['receipt']

    # Insert into activities_collection
    activities_collection.insert_one(activity)

    return jsonify({"message": f"{activity_type.capitalize()} activity created successfully."}), 201

@app.route('/api/activities',methods=['POST'])
def get_activities():
    data = request.get_json()
    mobileno = data.get('mobileno')
    yield_id = data.get('yield_id')
    print(mobileno +" "+yield_id)

    if not mobileno or not yield_id:
        return jsonify({"error": "Missing 'mobileno' or 'yield_id' in query parameters"}), 400

    # Find user by mobileno
    user = users_collection.find_one({'mobileno': mobileno})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Find yield with given yield_id and user ownership
    try:
        yield_obj = yields_collection.find_one({
            '_id': ObjectId(yield_id),
            'userId': user['_id']
        })
    except Exception as e:
        return jsonify({"error": f"Invalid yield ID format: {str(e)}"}), 400

    if not yield_obj:
        return jsonify({"error": "Yield not found for this user"}), 404
    print(yield_obj)
    # Fetch activities matching both user and yield
    activities = list(activities_collection.find({
        'userId': user['_id'],
        'yieldId': yield_obj['_id']
    }))
    # print(activities)

    # Convert ObjectId to string for JSON serializability
    for activity in activities:
        activity['_id'] = str(activity['_id'])
        activity['userId'] = str(activity['userId'])
        activity['yieldId'] = str(activity['yieldId'])
        activity['created_at'] = activity['created_at'].isoformat()

    return jsonify({"activities": activities}), 200

# ------------------ Chatbot API ------------------
@app.route('/api/chat', methods=['POST'])
def chatbot():
    try:
        data = request.json
        user_input = data.get('message', '')
        model = data.get('model', 'llama-3.3-70b-versatile')
        temperature = float(data.get('temperature', 0.7))
        max_tokens = int(data.get('max_tokens', 400))
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Initialize Groq client with API key
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Create the chat prompt with HTML formatting instructions and emphasis on brevity
        prompt = f"""
        As a helpful agricultural assistant, please respond to the following query: 
        
        {user_input}
        
        IMPORTANT INSTRUCTIONS:
        1. Be VERY CONCISE - limit your response to 3-4 short paragraphs maximum
        2. Focus only on the most relevant information
        3. Use simple, direct language
        4. Format using HTML for readability
        
        Use these HTML tags sparingly:
        - <h3> for a single main heading
        - <p> for paragraphs (keep them short)
        - <ul> with <li> for key points (limit to 3-5 items max)
        - <strong> for emphasis (use minimally)
        
        Do not include opening/closing HTML, body, or head tags - just the content HTML.
        """
        
        # Make the chat completion request
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable agricultural assistant. Provide BRIEF, CONCISE responses focused on farming practices. Format with minimal HTML for readability. Never exceed 400 tokens."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            stop=None,
            stream=False
        )
        
        # Extract the response
        response = chat_completion.choices[0].message.content.strip()
        
        return jsonify({'response': response}), 200
        
    except Exception as e:
        print(f"Error in chatbot API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Function to fetch real commodity prices from Mandi API
def fetch_mandi_prices(commodity: str, state: Optional[str] = None) -> Dict[str, float]:
    """
    Fetch current commodity prices from the Mandi API
    
    Args:
        commodity: The commodity to search for (e.g., "Rice", "Wheat")
        state: Optional state to filter results
        
    Returns:
        Dictionary mapping market/city names to prices
    """
    try:
        params = {
            "api-key": MANDI_API_KEY,
            "format": "json",
            "limit": 1000,  # Get a good number of records
            "filters[commodity]": commodity
        }
        
        if state:
            params["filters[state.keyword]"] = state
            
        # Build the URL with parameters
        query_string = urlencode(params)
        url = f"{MANDI_API_BASE_URL}?{query_string}"
        
        transport_logger.info(f"Fetching prices from Mandi API for {commodity}")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have valid records
            if "records" in data and data["records"]:
                prices = {}
                
                # Extract prices for each market
                for record in data["records"]:
                    market = record.get("market")
                    price = record.get("modal_price")
                    state_name = record.get("state")
                    
                    if market and price and state_name:
                        # Convert price to float
                        try:
                            price_float = float(price)
                            # Add to our prices dictionary
                            if market not in prices:
                                prices[market] = price_float
                            else:
                                # If market already exists, use the lower price (conservative)
                                prices[market] = min(prices[market], price_float)
                        except (ValueError, TypeError):
                            transport_logger.warning(f"Invalid price value for {market}: {price}")
                
                transport_logger.info(f"Fetched {len(prices)} market prices for {commodity}")
                
                # If we didn't find any prices, use simulated data
                if not prices:
                    transport_logger.warning(f"No price data found for {commodity}, using simulated data")
                    return simulate_crop_prices(commodity)
                
                # Map market prices to our city list based on state
                city_prices = map_market_to_city_prices(prices, commodity)
                return city_prices
            else:
                transport_logger.warning(f"No records found for {commodity} in Mandi API")
                return simulate_crop_prices(commodity)
        else:
            transport_logger.error(f"Mandi API request failed with status code {response.status_code}: {response.text}")
            return simulate_crop_prices(commodity)
    except Exception as e:
        transport_logger.error(f"Error fetching prices from Mandi API: {str(e)}")
        return simulate_crop_prices(commodity)

def map_market_to_city_prices(market_prices: Dict[str, float], commodity: str) -> Dict[str, float]:
    """
    Map market prices to our city list based on state information
    
    Args:
        market_prices: Dictionary of market name to price
        commodity: The commodity name
        
    Returns:
        Dictionary mapping city names to prices
    """
    city_prices = {}
    
    # First try to find direct market matches with our cities
    for city in city_data.keys():
        # Check if the city name is in market names
        for market, price in market_prices.items():
            if city.lower() in market.lower():
                city_prices[city] = price
                break
    
    # For cities without direct matches, use state average
    state_prices = defaultdict(list)
    
    # Group prices by state
    for city, state in city_to_state.items():
        for market, price in market_prices.items():
            # Simple heuristic - if market contains state name or vice versa
            if state.lower() in market.lower() or market.lower() in state.lower():
                state_prices[state].append(price)
    
    # Calculate state averages
    state_avg_prices = {}
    for state, prices in state_prices.items():
        if prices:
            state_avg_prices[state] = sum(prices) / len(prices)
    
    # Assign state average prices to cities without direct matches
    for city in city_data.keys():
        if city not in city_prices:
            state = city_to_state.get(city)
            if state and state in state_avg_prices:
                city_prices[city] = state_avg_prices[state]
    
    # For any remaining cities, generate simulated prices
    base_prices = {
        "Mumbai": 50.0, "Delhi": 55.0, "Bangalore": 52.0,
        "Chennai": 48.0, "Kolkata": 53.0
    }
    
    for city in city_data.keys():
        if city not in city_prices:
            # Use base price with small random variation
            city_prices[city] = base_prices[city] * (1 + random.uniform(-0.05, 0.05))
    
    transport_logger.info(f"Mapped market prices to cities for {commodity}: {city_prices}")
    return city_prices

# Simulated crop price API (fallback)
def simulate_crop_prices(crop: str) -> Dict[str, float]:
    try:
        base_prices = {
            "Mumbai": 50.0, "Delhi": 55.0, "Bangalore": 52.0,
            "Chennai": 48.0, "Kolkata": 53.0
        }
        prices = {city: price * (1 + random.uniform(-0.05, 0.05)) for city, price in base_prices.items()}
        transport_logger.info(f"Generated simulated prices for {crop}: {prices}")
        return prices
    except Exception as e:
        transport_logger.error(f"Failed to generate simulated crop prices: {str(e)}")
        return {city: 50.0 for city in city_data}

# Fetch crop prices - tries real API first, falls back to simulation
def fetch_crop_prices(crop: str) -> Dict[str, float]:
    try:
        # Try to get real prices from Mandi API
        prices = fetch_mandi_prices(crop)
        
        # If we got prices for all cities, return them
        if all(city in prices for city in city_data.keys()):
            return prices
        
        # Otherwise, use simulated prices
        transport_logger.warning(f"Incomplete price data for {crop}, using simulated data")
        return simulate_crop_prices(crop)
    except Exception as e:
        transport_logger.error(f"Failed to fetch crop prices: {str(e)}")
        return simulate_crop_prices(crop)

# Fetch dynamic fuel price (simulated)
def fetch_fuel_price() -> float:
    try:
        base_fuel_price = 1.20  # $ per liter
        fluctuation = random.uniform(-0.1, 0.1)  # ±10% variation
        fuel_price = base_fuel_price * (1 + fluctuation)
        transport_logger.info(f"Fetched fuel price: ${fuel_price:.2f}/liter")
        return fuel_price
    except Exception as e:
        transport_logger.error(f"Failed to fetch fuel price: {str(e)}")
        return 1.20

# Calculate transportation cost
def calculate_transport_cost(origin: str, destination: str, crop_weight_kg: float) -> float:
    try:
        # Validate inputs
        if origin not in city_data:
            transport_logger.error(f"Invalid origin city: {origin}")
            return 0.0
        if destination not in city_data:
            transport_logger.error(f"Invalid destination city: {destination}")
            return 0.0
        if crop_weight_kg <= 0:
            transport_logger.error(f"Invalid crop weight: {crop_weight_kg}")
            return 0.0
            
        origin_coords = city_data[origin]
        dest_coords = city_data[destination]
        distance_km = haversine(origin_coords, dest_coords)
        fuel_price = fetch_fuel_price()
        # Adjusted formula: Assume a truck with 5 km/liter efficiency and $0.50 per km base cost
        transport_cost = (distance_km / 5 * fuel_price * 100) + (distance_km * 0.50 * (crop_weight_kg / 100))  # Scaled for 100 kg
        transport_logger.info(f"Transport cost from {origin} to {destination}: ${transport_cost:.2f} for {crop_weight_kg} kg")
        return transport_cost
    except Exception as e:
        transport_logger.error(f"Error calculating transport cost: {str(e)}")
        return 0.0

# Dynamic transport optimizer
class TransportOptimizer:
    def __init__(self):
        self.current_city = None

    def optimize_transport(self, current_city: str, crop: str, crop_weight_kg: float) -> Dict:
        try:
            # Validate inputs
            if current_city not in city_data:
                transport_logger.error(f"Invalid current city: {current_city}")
                current_city = "Mumbai"  # Default to Mumbai if invalid
                
            if not crop or not isinstance(crop, str):
                transport_logger.error(f"Invalid crop: {crop}")
                crop = "Rice"  # Default to Rice if invalid
                
            if not crop_weight_kg or crop_weight_kg <= 0:
                transport_logger.error(f"Invalid crop weight: {crop_weight_kg}")
                crop_weight_kg = 100.0  # Default to 100kg if invalid
            
            self.current_city = current_city
            crop_prices = fetch_crop_prices(crop)

            results = defaultdict(dict)
            for city, price in crop_prices.items():
                if city == current_city:
                    transport_cost = 0.0  # No transport cost if selling in current city
                else:
                    transport_cost = calculate_transport_cost(current_city, city, crop_weight_kg)
                revenue = price * crop_weight_kg
                net_profit = revenue - transport_cost
                results[city] = {
                    "price_per_kg": price,
                    "transport_cost": transport_cost,
                    "net_profit": net_profit
                }

            results = dict(results)
            transport_logger.info(f"Calculated results: {results}")

            best_city = max(results.items(), key=lambda x: x[1]["net_profit"])[0]
            recommendation = {
                "current_city": current_city,
                "best_city": best_city,
                "best_net_profit": results[best_city]["net_profit"],
                "recommend_transport": best_city != current_city and results[best_city]["net_profit"] > results[current_city]["net_profit"],
                "city_details": results
            }
            transport_logger.info(f"Transport optimization result: {recommendation}")
            return recommendation
        except Exception as e:
            transport_logger.error(f"Optimization failed: {str(e)}")
            return {
                "current_city": current_city,
                "best_city": current_city,
                "best_net_profit": crop_weight_kg * 50.0,
                "recommend_transport": False,
                "city_details": {current_city: {"price_per_kg": 50.0, "transport_cost": 0.0, "net_profit": crop_weight_kg * 50.0}}
            }

# API endpoint to get available commodities from Mandi API
@app.route('/api/commodities', methods=['GET'])
def get_commodities():
    try:
        params = {
            "api-key": MANDI_API_KEY,
            "format": "json",
            "limit": 1000
        }
        
        # Try to get the data from Mandi API
        query_string = urlencode(params)
        url = f"{MANDI_API_BASE_URL}?{query_string}"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if "records" in data and data["records"]:
                # Extract unique commodities
                commodities = set()
                for record in data["records"]:
                    if "commodity" in record and record["commodity"]:
                        commodities.add(record["commodity"])
                
                # Return sorted list of commodities
                return jsonify({
                    "status": "success",
                    "commodities": sorted(list(commodities))
                }), 200
            else:
                # Fallback to default commodities
                default_commodities = ["Rice", "Wheat", "Maize", "Potato", "Onion", "Tomato"]
                return jsonify({
                    "status": "success",
                    "commodities": default_commodities,
                    "note": "Using default commodities as no data was found in API"
                }), 200
        else:
            # Fallback to default commodities
            default_commodities = ["Rice", "Wheat", "Maize", "Potato", "Onion", "Tomato"]
            return jsonify({
                "status": "success",
                "commodities": default_commodities,
                "note": f"Using default commodities due to API error: {response.status_code}"
            }), 200
            
    except Exception as e:
        transport_logger.error(f"Error fetching commodities: {str(e)}")
        # Fallback to default commodities
        default_commodities = ["Rice", "Wheat", "Maize", "Potato", "Onion", "Tomato"]
        return jsonify({
            "status": "success",
            "commodities": default_commodities,
            "note": "Using default commodities due to error"
        }), 200

# API endpoints for transport optimization
@app.route('/api/optimize-transport', methods=['POST'])
def optimize_transport():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        current_city = data.get("current_city", "Mumbai")
        crop = data.get("crop", "Rice")
        crop_weight_kg = float(data.get("crop_weight_kg", 100.0))

        # Validate inputs
        if current_city not in city_data:
            return jsonify({"error": f"Invalid city: {current_city}. Available cities: {list(city_data.keys())}"}), 400

        optimizer = TransportOptimizer()
        result = optimizer.optimize_transport(current_city, crop, crop_weight_kg)

        # Create map URL with query parameters
        map_url = f"/api/view-map?city={current_city}&best_city={result['best_city']}&crop={crop}&crop_weight_kg={crop_weight_kg}"

        response = {
            "optimization_result": result,
            "available_cities": list(city_data.keys()),
            "map_url": map_url
        }
        return jsonify(response), 200
    except Exception as e:
        transport_logger.error(f"Optimization failed: {str(e)}")
        return jsonify({"error": f"Optimization failed: {str(e)}", "status": "Error"}), 500

# ------------------ Dynamic Mandi API Endpoints ------------------
@app.route('/api/mandi/commodities', methods=['GET'])
def get_mandi_commodities():
    try:
        state = request.args.get("state", "Maharashtra")
        district = request.args.get("district", "Pune")
        
        params = {
            "api-key": MANDI_API_KEY,
            "format": "json",
            "limit": 100,
            "filters[State]": state,
            "filters[District]": district
        }
        
        url = MANDI_API_BASE_URL
        print(f"Fetching commodities from Mandi API for State={state}, District={district}...")
        response = requests.get(url, params=params, timeout=12)
        
        if response.status_code == 200:
            data = response.json()
            if "records" in data and data["records"]:
                commodities = set()
                for record in data["records"]:
                    if "commodity" in record and record["commodity"]:
                        commodities.add(record["commodity"])
                
                return jsonify({
                    "status": "success",
                    "commodities": sorted(list(commodities)),
                    "note": f"Loaded commodities from real-time Mandi API for {state}/{district}"
                }), 200
        
        # Fallback if status code is not 200 or no records
        print(f"Mandi API returned {response.status_code} or empty records. Using default commodities list.")
        default_commodities = ["Rice", "Wheat", "Maize", "Potato", "Onion", "Tomato", "Soybean", "Sugarcane", "Cotton", "Jowar", "Bajra"]
        return jsonify({
            "status": "success",
            "commodities": default_commodities,
            "note": "Using default commodities list (Mandi API fallback)"
        }), 200
        
    except Exception as e:
        print(f"Error fetching Mandi commodities: {str(e)}")
        default_commodities = ["Rice", "Wheat", "Maize", "Potato", "Onion", "Tomato", "Soybean", "Sugarcane", "Cotton", "Jowar", "Bajra"]
        return jsonify({
            "status": "success",
            "commodities": default_commodities,
            "note": "Using default commodities list (Mandi API error fallback)"
        }), 200


@app.route('/api/mandi/optimize-transport', methods=['POST'])
def optimize_mandi_transport():
    try:
        data = request.json or {}
        state = data.get("state", "Maharashtra")
        district = data.get("district", "Pune")
        crop = data.get("crop", "Rice")
        crop_weight_kg = float(data.get("crop_weight_kg", 100.0))
        
        # Determine starting coordinates
        start_coords = district_coordinates.get(district, (18.5204, 73.8567))
        
        # 1. Query Mandi API
        params = {
            "api-key": MANDI_API_KEY,
            "format": "json",
            "limit": 100,
            "filters[State]": state,
            "filters[District]": district
        }
        
        records = []
        api_success = False
        try:
            print(f"Querying Mandi API for transport optimization: State={state}, District={district}...")
            response = requests.get(MANDI_API_BASE_URL, params=params, timeout=12)
            if response.status_code == 200:
                data_json = response.json()
                if "records" in data_json and data_json["records"]:
                    records = data_json["records"]
                    api_success = True
                    print(f"Successfully retrieved {len(records)} records from Mandi API.")
        except Exception as err:
            print(f"Mandi API call failed or timed out: {str(err)}")
            
        # Filter records by commodity
        matching_records = []
        if api_success:
            for r in records:
                r_commodity = r.get("commodity", "")
                if r_commodity and r_commodity.lower() == crop.lower():
                    matching_records.append(r)
                    
        # 2. Process records if we have matching live data
        city_details = {}
        best_market = None
        best_net_profit = -999999.0
        
        # Predefined coordinates for common markets (offsets from district center)
        def get_deterministic_coords(market_name, base_coords):
            h = abs(hash(market_name))
            lat_offset = ((h % 100) / 400.0) - 0.125
            lng_offset = (((h // 100) % 100) / 400.0) - 0.125
            return (base_coords[0] + lat_offset, base_coords[1] + lng_offset)
            
        if matching_records:
            print(f"Found {len(matching_records)} matching records for crop '{crop}'")
            for r in matching_records:
                market = r.get("market", "Unknown Market")
                try:
                    price_per_kg = float(r.get("modal_price", 0.0)) / 100.0
                except (ValueError, TypeError):
                    price_per_kg = 0.0
                    
                if price_per_kg <= 0.0:
                    continue
                    
                lat, lng = get_deterministic_coords(market, start_coords)
                distance = haversine(start_coords, (lat, lng))
                
                if distance < 5:
                    transport_cost = 0.0
                else:
                    transport_cost = distance * 0.02 * crop_weight_kg
                    
                revenue = price_per_kg * crop_weight_kg
                net_profit = revenue - transport_cost
                
                city_details[market] = {
                    "price_per_kg": round(price_per_kg, 2),
                    "transport_cost": round(transport_cost, 2),
                    "net_profit": round(net_profit, 2),
                    "distance": round(distance, 1),
                    "coordinates": [round(lat, 4), round(lng, 4)]
                }
                
                if net_profit > best_net_profit:
                    best_net_profit = net_profit
                    best_market = market
        
        # 3. Fallback: If no API records matched or API timed out, generate simulated markets
        using_fallback = False
        if not city_details:
            using_fallback = True
            print("No matching Mandi records. Generating simulated markets as fallback.")
            
            base_prices = {
                "Rice": 42.0,
                "Wheat": 35.0,
                "Maize": 28.0,
                "Potato": 22.0,
                "Onion": 32.0,
                "Tomato": 25.0,
                "Soybean": 45.0,
                "Sugarcane": 3.5,
                "Cotton": 65.0,
                "Jowar": 30.0,
                "Bajra": 32.0
            }
            base_p = base_prices.get(crop, 30.0)
            
            sim_markets = [
                (f"{district} Center Mandi", 0.0, 0.95),
                (f"{district} North Market", 25.0, 1.05),
                (f"{district} East Plaza", 60.0, 1.15),
                (f"{district} Suburban Mandi", 15.0, 0.98)
            ]
            
            for m_name, dist, multiplier in sim_markets:
                price = base_p * multiplier
                lat, lng = get_deterministic_coords(m_name, start_coords)
                
                if dist < 5:
                    transport_cost = 0.0
                else:
                    transport_cost = dist * 0.02 * crop_weight_kg
                    
                revenue = price * crop_weight_kg
                net_profit = revenue - transport_cost
                
                city_details[m_name] = {
                    "price_per_kg": round(price, 2),
                    "transport_cost": round(transport_cost, 2),
                    "net_profit": round(net_profit, 2),
                    "distance": dist,
                    "coordinates": [round(lat, 4), round(lng, 4)]
                }
                
                if net_profit > best_net_profit:
                    best_net_profit = net_profit
                    best_market = m_name
                    
        starting_market = list(city_details.keys())[0]
        recommend_transport = best_market != starting_market and city_details[best_market]["net_profit"] > city_details[starting_market]["net_profit"]
        
        response_payload = {
            "status": "success",
            "current_city": starting_market,
            "best_city": best_market,
            "best_net_profit": round(best_net_profit, 2),
            "recommend_transport": recommend_transport,
            "city_details": city_details,
            "note": "Using simulated Mandi data due to timeout" if using_fallback else "Dynamic optimization based on live Mandi API"
        }
        
        return jsonify(response_payload), 200
        
    except Exception as e:
        transport_logger.error(f"Mandi optimization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Optimization failed: {str(e)}", "status": "Error"}), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    return jsonify({
        "cities": list(city_data.keys()),
        "map_info": {city: {"lat": coords[0], "lng": coords[1]} for city, coords in city_data.items()}
    }), 200

# Map template with Leaflet.js
MAP_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Crop Price Map with Route from {{ current_city }}</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
    
    <!-- Leaflet Routing Machine (for directions) -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.css" />
    
    <!-- Make sure you put the JS after the CSS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <script src="https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.js"></script>
    
    <style>
        #map { height: 600px; width: 100%; }
        .info-box {
            background-color: #fff;
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
            margin: 20px;
            padding: 15px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr.highlight {
            background-color: #e6f7ff;
            font-weight: bold;
        }
        .leaflet-popup-content {
            font-size: 14px;
        }
        /* Custom markers */
        .price-marker {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 120px;
            height: 30px;
            background-color: #2196F3;
            color: white;
            font-weight: bold;
            border-radius: 4px;
            text-align: center;
        }
        .best-price-marker {
            background-color: #4CAF50;
        }
        .data-source {
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            text-align: right;
        }
    </style>
</head>
<body>
    <h1>Crop Price Map with Route from {{ current_city }} to {{ best_city }}</h1>
    <div class="info-box">
        <h2>Transport Optimization Summary for {{ crop }} ({{ crop_weight_kg }} kg)</h2>
        <table>
            <tr>
                <th>City</th>
                <th>Price per kg</th>
                <th>Transport Cost</th>
                <th>Net Profit</th>
            </tr>
            {% for city, details in city_details.items() %}
            <tr {% if city == best_city %}class="highlight"{% endif %}>
                <td>{{ city }}</td>
                <td>${{ "%.2f"|format(details.price_per_kg) }}</td>
                <td>${{ "%.2f"|format(details.transport_cost) }}</td>
                <td>${{ "%.2f"|format(details.net_profit) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% if recommend_transport %}
        <p><strong>Recommendation:</strong> Transport to {{ best_city }} for maximum profit.</p>
        {% else %}
        <p><strong>Recommendation:</strong> Sell locally in {{ current_city }} for maximum profit.</p>
        {% endif %}
        <div class="data-source">Data source: Mandi API (data.gov.in)</div>
    </div>
    <div id="map"></div>
    
    <script>
        // Initialize the map
        function initMap() {
            try {
                // Create map centered on India
                const map = L.map('map').setView([22.5726, 78.9629], 5);
                
                // Add OpenStreetMap tiles
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                
                // Parse the data safely
                const cities = JSON.parse('{{ cities_json|safe }}');
                const prices = JSON.parse('{{ prices_json|safe }}');
                const cityDetails = JSON.parse('{{ city_details_json|safe }}');
                const currentCity = "{{ current_city }}";
                const bestCity = "{{ best_city }}";
                const distance = {{ distance }};
                const crop = "{{ crop }}";
                
                // Add markers for all cities
                for (const [city, data] of Object.entries(cities)) {
                    const isCurrentCity = city === currentCity;
                    const isBestCity = city === bestCity;
                    
                    // Create custom marker content
                    const customIcon = L.divIcon({
                        className: isBestCity ? 'price-marker best-price-marker' : 'price-marker',
                        html: `${city}: $${prices[city].toFixed(2)}/kg`,
                        iconSize: [120, 30],
                        iconAnchor: [60, 15]
                    });
                    
                    // Add marker
                    const marker = L.marker([data.lat, data.lng], {
                        icon: customIcon,
                        title: city
                    }).addTo(map);
                    
                    // Add popup with more details
                    marker.bindPopup(`
                        <b>${city}</b><br>
                        Crop: ${crop}<br>
                        Price: $${prices[city].toFixed(2)}/kg<br>
                        Transport Cost: $${cityDetails[city].transport_cost.toFixed(2)}<br>
                        Net Profit: $${cityDetails[city].net_profit.toFixed(2)}
                        ${isBestCity ? '<br><b>Best option for maximum profit!</b>' : ''}
                    `);
                }
                
                // Add route between current city and best city if they're different
                if (currentCity !== bestCity) {
                    const currentCityCoords = [cities[currentCity].lat, cities[currentCity].lng];
                    const bestCityCoords = [cities[bestCity].lat, cities[bestCity].lng];
                    
                    // Add straight line for reference
                    const straightLine = L.polyline([currentCityCoords, bestCityCoords], {
                        color: '#FF4500',
                        weight: 3,
                        opacity: 0.5,
                        dashArray: '10, 10',
                        lineJoin: 'round'
                    }).addTo(map);
                    
                    straightLine.bindPopup(`
                        <b>Direct distance</b><br>
                        From: ${currentCity}<br>
                        To: ${bestCity}<br>
                        ${distance.toFixed(2)} km
                    `);
                    
                    // Try to add routing if Leaflet Routing Machine is available
                    try {
                        if (typeof L.Routing !== 'undefined') {
                            const routing = L.Routing.control({
                                waypoints: [
                                    L.latLng(currentCityCoords[0], currentCityCoords[1]),
                                    L.latLng(bestCityCoords[0], bestCityCoords[1])
                                ],
                                routeWhileDragging: false,
                                showAlternatives: false,
                                fitSelectedRoutes: true,
                                lineOptions: {
                                    styles: [{color: '#0000FF', opacity: 0.7, weight: 5}]
                                }
                            }).addTo(map);
                            
                            // Catch routing errors
                            routing.on('routingerror', function(e) {
                                console.error('Routing error:', e);
                                alert('Could not calculate route. Showing direct line instead.');
                            });
                        } else {
                            console.warn('Leaflet Routing Machine not available');
                        }
                    } catch (error) {
                        console.error('Error setting up routing:', error);
                    }
                }
                
                // Fit the map to show all markers
                const bounds = [];
                for (const [city, data] of Object.entries(cities)) {
                    bounds.push([data.lat, data.lng]);
                }
                if (bounds.length > 0) {
                    map.fitBounds(bounds);
                }
            } catch (error) {
                console.error('Error initializing map:', error);
                document.getElementById('map').innerHTML = `<div style="padding: 20px; color: red;">Error loading map: ${error.message}</div>`;
            }
        }
        
        // Initialize map when DOM is loaded
        document.addEventListener('DOMContentLoaded', initMap);
    </script>
</body>
</html>
"""

@app.route('/api/view-map', methods=['GET'])
def view_map():
    try:
        current_city = request.args.get('city', 'Mumbai')
        best_city = request.args.get('best_city', 'Delhi')
        
        # Validate cities
        if current_city not in city_data:
            current_city = "Mumbai"  # Default if invalid
        if best_city not in city_data:
            best_city = "Delhi"  # Default if invalid
            
        crop = request.args.get('crop', 'Rice')
        crop_weight_kg = float(request.args.get('crop_weight_kg', 100.0))
        
        # Get optimization data
        optimizer = TransportOptimizer()
        result = optimizer.optimize_transport(current_city, crop, crop_weight_kg)
        
        city_details = result["city_details"]
        recommend_transport = result["recommend_transport"]
        
        # Prepare data for the map
        crop_prices = fetch_crop_prices(crop)
        
        # Format city data for JS
        cities_formatted = {}
        for city, coords in city_data.items():
            cities_formatted[city] = {"lat": coords[0], "lng": coords[1]}
        
        # Calculate direct distance
        current_coords = city_data[current_city]
        best_coords = city_data[best_city]
        distance_km = haversine(current_coords, best_coords)
        
        # Convert data to JSON for template
        cities_json = json.dumps(cities_formatted)
        prices_json = json.dumps(crop_prices)
        city_details_json = json.dumps(city_details)
        
        return render_template_string(
            MAP_TEMPLATE, 
            current_city=current_city,
            best_city=best_city,
            cities_json=cities_json,
            prices_json=prices_json,
            city_details_json=city_details_json,
            city_details=city_details,
            recommend_transport=recommend_transport,
            distance=distance_km,
            crop=crop,
            crop_weight_kg=crop_weight_kg
        )
    except Exception as e:
        transport_logger.error(f"Error rendering map: {str(e)}")
        error_template = """
        <!DOCTYPE html>
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Rendering Map</h1>
            <p>{{ error_message }}</p>
            <p><a href="/">Go Back to Home</a></p>
        </body>
        </html>
        """
        return render_template_string(error_template, error_message=str(e)), 500

# ------------------ Lease Marketplace API ------------------

@app.route('/api/lease-items', methods=['GET'])
def get_lease_items():
    try:
        lease_items = db.lease_items.find({})
        lease_items_list = []
        
        for item in lease_items:
            try:
                # Convert ObjectId to string
                item['_id'] = str(item['_id'])
                
                # Make sure ownerId is serialized properly
                if 'ownerId' in item and item['ownerId']:
                    item['ownerId'] = str(item['ownerId'])
                
                # Convert datetime objects to strings
                if 'createdAt' in item and item['createdAt']:
                    item['createdAt'] = item['createdAt'].isoformat()
                if 'updatedAt' in item and item['updatedAt']:
                    item['updatedAt'] = item['updatedAt'].isoformat()
                
                lease_items_list.append(item)
            except Exception as item_error:
                print(f"Error processing lease item: {str(item_error)}")
                # Skip this item and continue with the next
                continue
            
        print(f"Retrieved {len(lease_items_list)} lease items successfully")
        return jsonify({
            "status": "success",
            "data": lease_items_list
        }), 200
    except Exception as e:
        print(f"Error in get_lease_items: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/lease-items', methods=['POST'])
@token_required
def add_lease_item(current_user):
    try:
        data = request.json
        
        # Debug user authentication
        print("=== Debug Information ===")
        print(f"Headers received: {request.headers}")
        print(f"Token: {request.headers.get('x-access-token')}")
        print(f"Current user: {current_user}")
        
        # Validate required fields
        required_fields = ['name', 'description', 'imageUrl', 'category', 'pricePerHour', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Get the contact info from user or use a fallback
        owner_name = current_user.get('fullname', 'Equipment Owner')
        owner_contact = current_user.get('phone', current_user.get('mobileno', ''))
        
        if not owner_contact or owner_contact == '':
            print(f"Warning: No contact information found for user {owner_name}")
            # Use mobileno as fallback if phone is not available
            owner_contact = current_user.get('mobileno', '')

        # Create new lease item
        new_item = {
            "name": data['name'],
            "description": data['description'],
            "imageUrl": data['imageUrl'],
            "category": data['category'],
            "pricePerHour": float(data['pricePerHour']),
            "location": data['location'],
            "rating": 4.5,
            "reviews": data.get('reviews', 10),
            "available": True,
            "ownerId": current_user['_id'],
            "ownerName": owner_name,
            "ownerContact": owner_contact,
            "createdAt": datetime.now()
        }
        
        print(f"Creating lease item with owner: {owner_name}, contact: {owner_contact}")
        
        # Insert into database
        result = db.lease_items.insert_one(new_item)
        
        # Return success response with properly serialized data
        response_data = {
            "name": new_item["name"],
            "description": new_item["description"],
            "imageUrl": new_item["imageUrl"],
            "category": new_item["category"],
            "pricePerHour": new_item["pricePerHour"],
            "location": new_item["location"],
            "rating": new_item["rating"],
            "available": new_item["available"],
            "ownerName": new_item["ownerName"],
            "ownerContact": new_item["ownerContact"],
            "_id": str(result.inserted_id),
            "ownerId": str(new_item["ownerId"]),
            "createdAt": new_item["createdAt"].isoformat()
        }
        
        return jsonify({
            "status": "success",
            "message": "Lease item added successfully",
            "data": response_data
        }), 201
    except Exception as e:
        print(f"Error in add_lease_item: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/lease-items/<item_id>', methods=['GET'])
def get_lease_item(item_id):
    try:
        # Find item by ID
        item = db.lease_items.find_one({"_id": ObjectId(item_id)})
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
            
        # Convert ObjectId to string
        item['_id'] = str(item['_id'])
        item['ownerId'] = str(item['ownerId'])
        
        return jsonify({
            "status": "success",
            "data": item
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lease-items/<item_id>', methods=['PUT'])
@token_required
def update_lease_item(current_user, item_id):
    try:
        data = request.json
        
        # Find item
        item = db.lease_items.find_one({"_id": ObjectId(item_id)})
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
            
        # Check if user is the owner
        if str(item['ownerId']) != current_user['_id']:
            return jsonify({"error": "Not authorized to update this item"}), 403
            
        # Update fields
        update_data = {}
        allowed_fields = ['name', 'description', 'imageUrl', 'category', 'pricePerHour', 'location', 'available']
        
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
                
        if update_data:
            update_data['updatedAt'] = datetime.now()
            db.lease_items.update_one(
                {"_id": ObjectId(item_id)},
                {"$set": update_data}
            )
            
        # Get updated item
        updated_item = db.lease_items.find_one({"_id": ObjectId(item_id)})
        updated_item['_id'] = str(updated_item['_id'])
        updated_item['ownerId'] = str(updated_item['ownerId'])
        
        return jsonify({
            "status": "success",
            "message": "Item updated successfully",
            "data": updated_item
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lease-items/<item_id>', methods=['DELETE'])
@token_required
def delete_lease_item(current_user, item_id):
    try:
        # Find item
        item = db.lease_items.find_one({"_id": ObjectId(item_id)})
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
            
        # Check if user is the owner
        if str(item['ownerId']) != current_user['_id']:
            return jsonify({"error": "Not authorized to delete this item"}), 403
            
        # Delete item
        db.lease_items.delete_one({"_id": ObjectId(item_id)})
        
        return jsonify({
            "status": "success",
            "message": "Item deleted successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/lease-items/categories', methods=['GET'])
def get_lease_categories():
    # Return predefined categories for equipment
    categories = [
        "Tractor", 
        "Harvester", 
        "Seeder", 
        "Irrigation System", 
        "Sprayer", 
        "Thresher",
        "Cultivator",
        "Plow",
        "Other"
    ]
    
    return jsonify({
        "status": "success",
        "data": categories
    }), 200

# ------------------ Seed Demo Data ------------------
def seed_demo_data():
    try:
        # Check if we have any lease items
        items_count = db.lease_items.count_documents({})
        
        if items_count == 0:
            print("No lease items found, seeding demo data...")
            
            # Create a demo user if not exists
            demo_user = users_collection.find_one({'mobileno': '9999999999'})
            if not demo_user:
                demo_user_id = ObjectId()
                users_collection.insert_one({
                    '_id': demo_user_id,
                    'fullname': 'Demo User',
                    'mobileno': '9999999999',
                    'password': generate_password_hash('password'),
                    'token': str(uuid.uuid4())
                })
            else:
                demo_user_id = demo_user['_id']
                
            # Add sample equipment
            demo_items = [
                {
                    "name": "John Deere 5E Series Tractor",
                    "category": "Tractor",
                    "description": "Powerful 75HP tractor ideal for medium to large farms. Includes attachments for plowing.",
                    "pricePerHour": 400,
                    "location": "Nashik, Maharashtra",
                    "rating": 4.5,
                    "reviews": 24,
                    "imageUrl": "https://images.unsplash.com/photo-1605002123541-539772db692b?q=80&w=800",
                    "available": True,
                    "ownerId": demo_user_id,
                    "ownerName": "Demo User",
                    "ownerContact": "9999999999",
                    "createdAt": datetime.now()
                },
                {
                    "name": "CLAAS Harvester",
                    "category": "Harvester",
                    "description": "High-capacity combine harvester for wheat, rice, and other grain crops.",
                    "pricePerHour": 800,
                    "location": "Pune, Maharashtra",
                    "rating": 4.5,
                    "reviews": 18,
                    "imageUrl": "https://images.unsplash.com/photo-1591191425088-195b6e978259?q=80&w=800",
                    "available": True,
                    "ownerId": demo_user_id,
                    "ownerName": "Demo User",
                    "ownerContact": "9999999999",
                    "createdAt": datetime.now()
                },
                {
                    "name": "KisanKraft Irrigation System",
                    "category": "Irrigation System",
                    "description": "Complete drip irrigation system with controller for 2-acre farms.",
                    "pricePerHour": 150,
                    "location": "Satara, Maharashtra",
                    "rating": 4.5,
                    "reviews": 32,
                    "imageUrl": "https://images.unsplash.com/photo-1629793376581-8f4b9ee14537?q=80&w=800",
                    "available": True,
                    "ownerId": demo_user_id,
                    "ownerName": "Demo User",
                    "ownerContact": "9999999999",
                    "createdAt": datetime.now()
                }
            ]
            
            # Insert demo items
            db.lease_items.insert_many(demo_items)
            print(f"Added {len(demo_items)} demo items to the database")
        else:
            print(f"Found {items_count} existing items, skipping demo data seeding")
    except Exception as e:
        print(f"Error seeding demo data: {str(e)}")
        import traceback
        traceback.print_exc()

#  Get Groq API key from environment variables
GROQ_API_KEY = config('GROQ_API_KEY')

# Load the model
try:
    model = joblib.load("./models/adaboost_model_soil.pkl")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Function to get insights from Groq
def get_groq_insights(soil_params, soil_type, location, land_area, model_name="llama3-70b-8192"):
    # Create a prompt for Groq
    prompt = f"""
    As an agricultural expert, provide detailed insights and recommendations based on the following soil analysis:
    
    Location: {location}
    Land Area: {land_area} hectares
    Soil Type: {soil_type}
    
    Soil Parameters:
    - Nitrogen: {soil_params['N']} kg/ha
    - Phosphorus: {soil_params['P']} kg/ha
    - Potassium: {soil_params['K']} kg/ha
    - Temperature: {soil_params['temperature']}°C
    - Humidity: {soil_params['humidity']}%
    - pH: {soil_params['ph']}
    - Rainfall: {soil_params['rainfall']} mm/month
    
    Please provide:
    1. Top 3 most suitable crops with brief explanations
    2. Estimated yield potential for each recommended crop
    3. Specific farming recommendations (planting time, irrigation needs, fertilizer suggestions)
    4. Any soil health concerns and improvement strategies
    5. Sustainable farming practices that would work well with this soil profile
    
    Format your response as JSON with the following structure:
    {{
        "top_crops": [
            {{"name": "Crop1", "suitability": "95%", "water_requirement": "High/Medium/Low", "growth_period": "X-Y months"}},
            {{"name": "Crop2", "suitability": "87%", "water_requirement": "High/Medium/Low", "growth_period": "X-Y months"}},
            {{"name": "Crop3", "suitability": "79%", "water_requirement": "High/Medium/Low", "growth_period": "X-Y months"}}
        ],
        "best_crop": {{
            "name": "Crop1",
            "confidence": "95%",
            "environmental_suitability": "Excellent/Good/Moderate",
            "estimated_yield": "X-Y tons/hectare",
            "recommendation": "Brief planting and care recommendation"
        }},
        "soil_health": {{
            "status": "Excellent/Good/Needs improvement",
            "concerns": ["Concern1", "Concern2"],
            "improvement_strategies": ["Strategy1", "Strategy2", "Strategy3"]
        }},
        "sustainable_practices": ["Practice1", "Practice2", "Practice3"]
    }}
    """
    
    try:
        # Prepare the API request for Groq
        url = "https://api.groq.com/openai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert agricultural advisor with deep knowledge of soil science, crop selection, and sustainable farming practices."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        # Call Groq API
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Parse the response
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        insights = json.loads(content)
        return insights, None
    
    except Exception as e:
        error_message = f"Error getting insights: {str(e)}"
        return None, error_message

# Routes

@app.route('/api/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.get_json()

        required_fields = ['Temperature', 'Humidity', 'Soil Moisture', 'Soil Type',
                           'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorus']

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        soil_encoder = joblib.load("../models/soil_type_encoder.pkl")
        crop_encoder = joblib.load("../models/crop_type_encoder.pkl")
        fertilizer_encoder = joblib.load("../models/fertilizer_encoder.pkl")

        # Transform categorical features
        soil_type_encoded = soil_encoder.transform([data['Soil Type']])[0]
        crop_type_encoded = crop_encoder.transform([data['Crop Type']])[0]

        # Prepare input data
        input_data = pd.DataFrame([{
            'Temperature': float(data['Temperature']),
            'Humidity': float(data['Humidity']),
            'Soil Moisture': float(data['Soil Moisture']),
            'Soil Type': soil_type_encoded,
            'Crop Type': crop_type_encoded,
            'Nitrogen': int(data['Nitrogen']),
            'Potassium': int(data['Potassium']),
            'Phosphorus': int(data['Phosphorus']),
        }])

        xgb = joblib.load('../models/xgb_fertilizer_model.pkl')
        pred = xgb.predict(input_data)

        

        return jsonify({
            'recommended_fertilizer' : pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# farmwers activity via mobile

@app.route('/api/mobile-activity', methods=['POST'])
def mobile_activity():
    data = request.get_json()
    required_fields = ['mobileno', 'yield_id', 'text']

    # Check for missing required fields
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400

    text = data['text'].strip()
    if text == '':
        return jsonify({'message': 'No activity added'}), 400

    try:
        user = users_collection.find_one({'mobileno': data['mobileno']})
        if not user:
            return jsonify({'message': 'User not found'}), 404

        yieldObj = yields_collection.find_one({
            '_id': ObjectId(data['yield_id']),
            'userId': user['_id']
        })
        if not yieldObj:
            return jsonify({'message': 'Yield not found'}), 404

    except Exception as e:
        return jsonify({'message': f'Failed to retrieve user/yield: {str(e)}'}), 500

    prompt = f"""
    Categorize the following agricultural text input and extract structured activity details:

    "{text}"

    As an intelligent agricultural assistant, analyze the user's input and generate a structured JSON object with the following keys:
    {{
        "activity_type": "Category of the activity (e.g., Harvesting, Sowing, Irrigation, Fertilization, Expense, etc.)",
        "activity_name": "Short name or title of the activity",
        "summary": "Brief summary or explanation of the activity described in the input",
        "amount": "Extracted amount involved in the activity, if mentioned (in numeric form without currency symbol)"
    }}

    Please strictly return only the valid JSON. No extra explanation, no surrounding text — only a clean JSON object.
    """

    try:
        # Initialize Groq client
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Make the chat completion request
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Correct model ID
            messages=[
                {"role": "system", "content": "You are an intelligent agricultural assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=1.0
        )

        # Extract and parse JSON response
        detailed_info = chat_completion.choices[0].message.content.strip()
        activity_data = json.loads(detailed_info)

        # Save activity
        activities_collection.insert_one({
            'userId': user['_id'],
            'yieldId': yieldObj['_id'],
            'activity_name': activity_data.get("activity_name"),
            'activity_type': activity_data.get("activity_type"),
            'summary': activity_data.get("summary"),
            'amount': float(activity_data.get("amount", 0))
        })

        return jsonify({'message': 'Activity created successfully'}), 201

    except json.JSONDecodeError:
        return jsonify({'message': 'Invalid response format from AI model'}), 500
    except Exception as e:
        return jsonify({'message': f'Failed to create activity: {str(e)}'}), 500


# Call the seed function at startup
seed_demo_data()


# ------------------ Yield Forecast API ------------------
@app.route('/api/yields/<yield_id>/forecast', methods=['GET'])
@token_required
def get_yield_forecast(current_user, yield_id):
    try:
        print(f"Generating forecast for yield {yield_id} and user {current_user.get('_id')}")
        
        # 1. Fetch yield data
        yield_obj = yields_collection.find_one({"_id": ObjectId(yield_id)})
        if not yield_obj:
            return jsonify({"status": "error", "message": "Yield not found"}), 404
            
        # Check ownership (match update_yield logic, allow test user)
        if str(yield_obj.get('userId')) != str(current_user.get('_id')) and current_user.get('fullname') != "Test User":
            return jsonify({"status": "error", "message": "Unauthorized to access this yield forecast"}), 403

        # 2. Fetch activity list
        activities = list(activities_collection.find({"yieldId": ObjectId(yield_id)}))
        
        # Format activities list (to match frontend dates and amounts)
        activities_with_dates = []
        for act in activities:
            created_at = act.get('created_at') or act.get('date')
            if isinstance(created_at, str):
                try:
                    # Try to parse ISO date string
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except Exception:
                    created_at = datetime.utcnow()
            elif not isinstance(created_at, datetime):
                created_at = datetime.utcnow()
            
            activities_with_dates.append({
                'type': act.get('activity_type', 'Other'),
                'amount': float(act.get('amount', 0)),
                'date': created_at
            })
            
        # Sort activities by date
        activities_with_dates.sort(key=lambda x: x['date'])
        
        now = datetime.now()
        current_year = now.year
        current_month = now.month
        
        # Group actual expenses and income by month
        actual_points = []
        
        # Helper to format month-year key: e.g., "Jun 26"
        def format_month_key(dt):
            return dt.strftime("%b %y")
            
        if activities_with_dates:
            earliest_date = activities_with_dates[0]['date']
            # Limit history to 6 months ago
            six_months_ago = datetime(now.year - (1 if now.month <= 6 else 0), (now.month - 6) if now.month > 6 else (now.month + 6), 1)
            start_date = max(earliest_date, six_months_ago)
            
            # Generate calendar months from start_date up to the month before current
            curr = datetime(start_date.year, start_date.month, 1)
            history_months = []
            while curr < datetime(current_year, current_month, 1):
                history_months.append(curr)
                # Next month
                if curr.month == 12:
                    curr = datetime(curr.year + 1, 1, 1)
                else:
                    curr = datetime(curr.year, curr.month + 1, 1)
                    
            # Sum amounts for each history month
            for m_dt in history_months:
                m_start = datetime(m_dt.year, m_dt.month, 1)
                if m_dt.month == 12:
                    m_end = datetime(m_dt.year + 1, 1, 1)
                else:
                    m_end = datetime(m_dt.year, m_dt.month + 1, 1)
                    
                m_activities = [a for a in activities_with_dates if m_start <= a['date'] < m_end]
                
                investment = sum(a['amount'] for a in m_activities if a['amount'] > 0)
                revenue = sum(abs(a['amount']) for a in m_activities if a['amount'] < 0)
                profit = revenue - investment
                
                actual_points.append({
                    "month": format_month_key(m_dt),
                    "investment": int(round(investment)),
                    "revenue": int(round(revenue)),
                    "profit": int(round(profit)),
                    "isProjected": False
                })

        # 3. Future Projections (Current month + next 5 months = 6 months)
        projected_points = []
        assumptions = []
        
        # Calculate overall averages from actual expenses
        expense_activities = [a for a in activities_with_dates if a['amount'] > 0]
        income_activities = [a for a in activities_with_dates if a['amount'] < 0]
        
        overall_avg_expense = 5000.0  # Default baseline if no history
        if expense_activities:
            overall_avg_expense = sum(a['amount'] for a in expense_activities) / len(expense_activities)
            
        # Get historical monthly average for the last 3 months
        monthly_expense_totals = {}
        for a in expense_activities:
            key = (a['date'].year, a['date'].month)
            monthly_expense_totals[key] = monthly_expense_totals.get(key, 0.0) + a['amount']
            
        historical_expenses = list(monthly_expense_totals.values())
        if historical_expenses:
            recent_monthly_avg = sum(historical_expenses[-3:]) / min(3, len(historical_expenses))
        else:
            recent_monthly_avg = overall_avg_expense
            
        # Crop Lifecycle Policy Setup
        crop_type = (yield_obj.get('type') or yield_obj.get('name') or "Crop").strip().lower()
        acres = float(yield_obj.get('acres') or 1.0)
        days_remain = int(yield_obj.get('daysRemain') or 180)
        remaining_months = max(1, min(6, int(round(days_remain / 30.0))))
        
        # Crop Lifecycle configurations (standard cost per acre for pending stages)
        crop_lifecycles = {
            "sugarcane": {
                "duration": 12,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3500, "sowing": 3000, "fertilizer": 5000, "irrigation": 3000, "pesticide": 2500, "harvesting": 6000, "financial": 1000}
            },
            "wheat": {
                "duration": 5,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3000, "sowing": 2500, "fertilizer": 4000, "irrigation": 2000, "pesticide": 2000, "harvesting": 5000, "financial": 800}
            },
            "rice": {
                "duration": 4,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3200, "sowing": 2800, "fertilizer": 4500, "irrigation": 3500, "pesticide": 3000, "harvesting": 5000, "financial": 800}
            },
            "paddy": {
                "duration": 4,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3200, "sowing": 2800, "fertilizer": 4500, "irrigation": 3500, "pesticide": 3000, "harvesting": 5000, "financial": 800}
            },
            "cotton": {
                "duration": 6,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3000, "sowing": 2500, "fertilizer": 4500, "irrigation": 2500, "pesticide": 5000, "harvesting": 5500, "financial": 1000}
            },
            "tomato": {
                "duration": 4,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3000, "sowing": 2500, "fertilizer": 4000, "irrigation": 2500, "pesticide": 4000, "harvesting": 4500, "financial": 800}
            }
        }
        
        # Match crop type
        matched_crop = None
        for k in crop_lifecycles.keys():
            if k in crop_type:
                matched_crop = k
                break
                
        if matched_crop:
            lifecycle = crop_lifecycles[matched_crop]
            assumptions.append(f"Applying server-side lifecycle policy for crop type: {matched_crop.capitalize()}.")
        else:
            # Default fallback policy
            lifecycle = {
                "duration": 6,
                "stages": ["cultivation", "sowing", "fertilizer", "irrigation", "pesticide", "harvesting", "financial"],
                "costs": {"cultivation": 3000, "sowing": 2500, "fertilizer": 4000, "irrigation": 2500, "pesticide": 3000, "harvesting": 5000, "financial": 800}
            }
            assumptions.append("Applying standard crop lifecycle baseline policies.")
            
        # Determine completed vs pending stages
        completed_stages = set(a['type'].lower() for a in activities_with_dates)
        pending_stages = [stage for stage in lifecycle["stages"] if stage not in completed_stages]
        
        if pending_stages:
            assumptions.append("Pending lifecycle stages identified: " + ", ".join(s.capitalize() for s in pending_stages) + ".")
        else:
            assumptions.append("All primary lifecycle activity stages have already been logged.")
            
        # Check weather risk factor (if set by backend, or default)
        user_weather_risk = float(yield_obj.get('weatherRiskFactor') or 1.0)
        
        # Generate 6 projected months starting from current month
        projected_months = []
        curr = datetime(current_year, current_month, 1)
        for _ in range(6):
            projected_months.append(curr)
            if curr.month == 12:
                curr = datetime(curr.year + 1, 1, 1)
            else:
                curr = datetime(curr.year, curr.month + 1, 1)
                
        # Simulate Weather Policy for the next 6 months
        weather_forecast = []
        monthly_weather_risks = []
        
        for p_dt in projected_months:
            m_idx = p_dt.month
            if m_idx in [11, 12, 1, 2]:
                temp = random.randint(16, 22)
                rain = random.randint(5, 20)
                humi = random.randint(45, 55)
                season = "Winter"
                risk_lvl = "Low"
                risk_f = 0.9 * user_weather_risk
            elif m_idx in [3, 4, 5]:
                temp = random.randint(34, 41)
                rain = random.randint(5, 15)
                humi = random.randint(30, 40)
                season = "Summer"
                risk_lvl = "Medium"
                risk_f = 1.15 * user_weather_risk
            else:
                temp = random.randint(25, 30)
                rain = random.randint(200, 380)
                humi = random.randint(75, 88)
                season = "Monsoon"
                risk_lvl = "High" if matched_crop in ["tomato", "cotton"] else "Medium"
                risk_f = 1.25 * user_weather_risk
                
            weather_forecast.append({
                "month": format_month_key(p_dt),
                "temperature": temp,
                "rainfall": rain,
                "humidity": humi,
                "season": season,
                "riskLevel": risk_lvl,
                "riskFactor": risk_f
            })
            monthly_weather_risks.append(risk_f)

        # Distribute pending stage costs over the remaining months slots
        allocated_stage_costs = [0.0] * 6
        
        for idx, stage in enumerate(pending_stages):
            stage_history = [a['amount'] for a in activities_with_dates if a['type'].lower() == stage and a['amount'] > 0]
            if stage_history:
                hist_avg = sum(stage_history) / len(stage_history)
                estimated_cost = (hist_avg * 0.6) + (lifecycle["costs"].get(stage, 3000) * acres * 0.4)
            else:
                estimated_cost = lifecycle["costs"].get(stage, 3000) * acres
                
            slot = min(5, int((idx * remaining_months) / max(1, len(pending_stages))))
            allocated_stage_costs[slot] += estimated_cost

        # Calculate revenue projection basis
        expected_price = float(yield_obj.get('expectedPricePerUnit') or 0.0)
        expected_yield = float(yield_obj.get('expectedYield') or 0.0)
        
        if expected_yield <= 0.0 and float(yield_obj.get('expectedYieldPerAcre') or 0.0) > 0.0:
            expected_yield = float(yield_obj['expectedYieldPerAcre']) * acres
            
        revenue_basis_note = ""
        total_projected_revenue = 0.0
        
        if expected_price > 0.0 and expected_yield > 0.0:
            total_projected_revenue = expected_price * expected_yield
            revenue_basis_note = f"Revenue projected from user-supplied expected yield ({expected_yield} units) and price (₹{expected_price}/unit)."
        elif matched_crop:
            baselines = {
                "sugarcane": {"yield_acre": 35.0, "price": 3200.0, "units": "tonnes"},
                "wheat": {"yield_acre": 22.0, "price": 2275.0, "units": "quintals"},
                "rice": {"yield_acre": 20.0, "price": 2200.0, "units": "quintals"},
                "paddy": {"yield_acre": 20.0, "price": 2200.0, "units": "quintals"},
                "cotton": {"yield_acre": 12.0, "price": 7000.0, "units": "quintals"},
                "tomato": {"yield_acre": 120.0, "price": 1500.0, "units": "quintals"}
            }
            base = baselines[matched_crop]
            total_projected_revenue = base["yield_acre"] * acres * base["price"]
            revenue_basis_note = f"Expected yield not specified. Using regional baseline for {matched_crop.capitalize()} ({base['yield_acre']} {base['units']}/acre at ₹{base['price']}/{base['units']}) as fallback."
        elif len(income_activities) >= 2:
            avg_income = sum(abs(a['amount']) for a in income_activities) / len(income_activities)
            total_projected_revenue = avg_income * 6
            revenue_basis_note = "Expected yield not specified. Revenue projected from historical income activities average."
            
        if revenue_basis_note:
            assumptions.append(revenue_basis_note)
            
        # Fallback check
        if len(expense_activities) < 2 and not matched_crop:
            return jsonify({
                "available": False,
                "points": [],
                "summary": None,
                "reason": "Forecast needs at least 2 logged expense activities or a specified crop type/acres to apply regional lifecycle templates.",
                "assumptions": ["Add more real farm records or update crop details to unlock forecast insights."]
            }), 200

        # Build monthly projection values
        for i, p_dt in enumerate(projected_months):
            risk_f = monthly_weather_risks[i]
            
            base_inv = recent_monthly_avg * risk_f
            stage_inv = allocated_stage_costs[i] * risk_f
            investment = int(round(base_inv + stage_inv))
            
            revenue = 0.0
            if "historical income" in revenue_basis_note:
                revenue = total_projected_revenue / 6.0 * risk_f
            elif total_projected_revenue > 0.0:
                harvest_slot = min(5, remaining_months - 1)
                pre_harvest_slot = max(0, harvest_slot - 1)
                
                if harvest_slot == pre_harvest_slot:
                    if i == harvest_slot:
                        revenue = total_projected_revenue * risk_f
                else:
                    if i == pre_harvest_slot:
                        revenue = total_projected_revenue * 0.2 * risk_f
                    elif i == harvest_slot:
                        revenue = total_projected_revenue * 0.8 * risk_f
                        
            revenue = int(round(revenue))
            profit = int(round(revenue - investment))
            
            projected_points.append({
                "month": format_month_key(p_dt),
                "investment": max(0, investment),
                "revenue": max(0, revenue),
                "profit": profit,
                "isProjected": True
            })
            
        combined_points = actual_points + projected_points
        
        total_investment = sum(p["investment"] for p in projected_points)
        total_profit = sum(p["profit"] for p in projected_points)
        
        highest_inv_point = max(combined_points, key=lambda x: x["investment"])
        highest_profit_point = max(combined_points, key=lambda x: x["profit"])
        
        roi = 0.0
        if total_investment > 0:
            roi = (float(total_profit) / float(total_investment)) * 100.0
            
        summary = {
            "totalInvestment": total_investment,
            "totalProfit": total_profit,
            "highestInvestmentMonth": {
                "month": highest_inv_point["month"],
                "investment": highest_inv_point["investment"],
                "revenue": highest_inv_point["revenue"],
                "profit": highest_inv_point["profit"]
            },
            "highestProfitMonth": {
                "month": highest_profit_point["month"],
                "investment": highest_profit_point["investment"],
                "revenue": highest_profit_point["revenue"],
                "profit": highest_profit_point["profit"]
            },
            "roiPercent": round(roi, 2)
        }
        
        avg_risk = sum(monthly_weather_risks) / 6.0
        if avg_risk > 1.1:
            assumptions.append("Weather risk is elevated for the projection period. Projections adjusted accordingly.")
        else:
            assumptions.append("Weather conditions are projected to be standard. Baseline risk levels applied.")
            
        return jsonify({
            "available": True,
            "points": combined_points,
            "summary": summary,
            "assumptions": assumptions,
            "weatherForecast": weather_forecast
        }), 200
        
    except Exception as e:
        print(f"Error generating forecast: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# Call the seed function at startup
seed_demo_data()

# ------------------ Cost Reduction Suggestions API ------------------
@app.route('/api/cost-reduction-suggestions', methods=['POST'])
@token_required
def get_cost_reduction_suggestions(current_user):
    try:
        print("Received cost reduction suggestion request")
        data = request.json
        print(f"Request data: {data}")
        expense_data = data.get('expenseData', [])
        yield_name = data.get('yieldName', 'your crop')
        total_expense = data.get('totalExpense', 0)
        
        if not expense_data:
            print("No expense data provided")
            return jsonify({
                "status": "error",
                "message": "No expense data provided"
            }), 400
        
        # Format expense data for the Gemini prompt
        expense_text = ""
        for category in expense_data:
            expense_text += f"- {category['name']}: ₹{category['value']} ({category.get('percentage', 0)}% of total)\n"
        
        print(f"Formatted expense data: {expense_text}")
        
        # Create the prompt for Gemini
        prompt = f"""
        As an agricultural financial advisor, analyze the following expense data for {yield_name} and provide 4-6 specific cost reduction suggestions.
        
        Total Expense: ₹{total_expense}
        
        Expense Breakdown:
        {expense_text}
        
        For each suggestion:
        1. Provide a concise title (5-7 words)
        2. Write a detailed explanation (2-3 sentences) with specific savings potential where possible
        3. Categorize it as either "General" or specific to the highest expense category
        
        Format your response as valid JSON with this structure:
        {{
            "suggestions": [
                {{
                    "title": "Suggestion title",
                    "description": "Detailed explanation with savings potential",
                    "category": "Category name or General"
                }},
                // More suggestions...
            ]
        }}
        
        Return ONLY the JSON object without any additional explanation or markdown formatting.
        """
        
        print("Prompt created for Gemini")
        
        # Generate suggestions with mock data for testing purpose
        # This is a fallback if Gemini API is not available
        mock_data = {
            "suggestions": [
                {
                    "title": "Optimize fertilizer application",
                    "description": "Use soil testing to determine exact nutrient needs. This can reduce fertilizer costs by 15-20% while maintaining or improving yields.",
                    "category": "Fertilizer"
                },
                {
                    "title": "Implement water-saving irrigation techniques",
                    "description": "Switch to drip irrigation or moisture sensors to reduce water usage. This approach can save 30-50% on irrigation costs and prevent yield loss from over-watering.",
                    "category": "Irrigation"
                },
                {
                    "title": "Form equipment sharing cooperatives",
                    "description": "Share expensive machinery with neighboring farmers to divide acquisition and maintenance costs. Equipment sharing can reduce capital expenses by 40-60%.",
                    "category": "General"
                },
                {
                    "title": "Use integrated pest management",
                    "description": "Combine biological controls with targeted chemical applications. IPM can reduce pesticide costs by 30-40% while maintaining effective pest control.",
                    "category": "Pesticide"
                }
            ]
        }
        
        # Call Gemini API to generate suggestions
        try:
            # Initialize Gemini client with API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Gemini API key not configured, using mock data")
                return jsonify({
                    "status": "success",
                    "data": mock_data
                }), 200
            
            print(f"Using Gemini API key (first few chars): {api_key[:5]}...")
            
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # Set up the model
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
            }
            
            # Initialize the model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config
            )
            
            print("Making request to Gemini API using Python client")
            response = model.generate_content(prompt)
            
            if not response:
                print("Empty response from Gemini API")
                return jsonify({
                    "status": "success",
                    "data": mock_data,
                    "note": "Using fallback data due to empty API response"
                }), 200
            
            print("Received response from Gemini API")
            
            # Extract the text from Gemini's response
            try:
                suggestions_text = response.text
                print(f"Raw suggestions text (first 100 chars): {suggestions_text[:100]}...")
            except Exception as e:
                print(f"Error accessing response text: {e}")
                return jsonify({
                    "status": "success",
                    "data": mock_data,
                    "note": "Using fallback data due to API response format issue"
                }), 200
            
            # Clean the JSON content by removing any markdown formatting
            cleaned_json_content = suggestions_text.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
            print(f"Cleaned JSON content (first 100 chars): {cleaned_json_content[:100]}...")
            
            # Parse the suggestions JSON
            try:
                suggestions_data = json.loads(cleaned_json_content)
                print(f"Successfully parsed JSON with keys: {list(suggestions_data.keys())}")
                return jsonify({
                    "status": "success",
                    "data": suggestions_data
                }), 200
            except json.JSONDecodeError as e:
                print(f"Error parsing Gemini response as JSON: {e}")
                print(f"Raw response: {suggestions_text}")
                # Return mock data as fallback
                return jsonify({
                    "status": "success",
                    "data": mock_data,
                    "note": "Using fallback data due to JSON parsing error"
                }), 200
                
        except Exception as gemini_error:
            print(f"Error calling Gemini API: {str(gemini_error)}")
            import traceback
            traceback.print_exc()
            
            # Attempt to call Groq API as an intelligent fallback
            try:
                print("Gemini failed. Attempting to call Groq API as fallback...")
                groq_api_key = os.getenv("GROQ_API_KEY")
                if groq_api_key:
                    groq_client = Groq(api_key=groq_api_key)
                    chat_completion = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an agricultural financial advisor. Return ONLY a valid JSON object matching the requested schema. Do not include markdown code block formatting (like ```json) or explanation."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.2,
                        max_completion_tokens=600
                    )
                    groq_response = chat_completion.choices[0].message.content.strip()
                    print(f"Raw Groq response (first 100 chars): {groq_response[:100]}...")
                    
                    cleaned_groq_json = groq_response.replace('```json\n', '').replace('```\n', '').replace('```', '').strip()
                    try:
                        suggestions_data = json.loads(cleaned_groq_json)
                        print("Successfully parsed Groq response as JSON")
                        return jsonify({
                            "status": "success",
                            "data": suggestions_data,
                            "note": "Generated dynamically using Groq LLM"
                        }), 200
                    except json.JSONDecodeError as json_err:
                        print(f"Error parsing Groq JSON: {json_err}")
                else:
                    print("Groq API key not configured")
            except Exception as groq_err:
                print(f"Error calling Groq API fallback: {str(groq_err)}")
                traceback.print_exc()
                
            # Fallback to mock data if both LLMs fail
            return jsonify({
                "status": "success",
                "data": mock_data,
                "note": "Using fallback static data due to LLM errors"
            }), 200
    
    except Exception as e:
        print(f"Error in cost reduction suggestions: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
