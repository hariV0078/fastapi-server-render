from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from groq import Groq
import fitz
import re
import json
import base64
import os
import logging
import tempfile
from pydantic import BaseModel
from typing import Optional

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()

# Allow CORS (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys (Replace with ENV variables for security)
GOOGLE_API_KEY = "AIzaSyB9_uEfeyLvJ1O-PrT8Qlj8PlOG-p_MvsU"
GROQ_API_KEY = "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"

# Initialize API Clients
client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)


def clean_json(markdown_json_string):
    """Extracts JSON content from a markdown response."""
    json_content = re.search(r'```json\n(.*?)```', markdown_json_string, re.DOTALL)
    if json_content:
        try:
            return json.loads(json_content.group(1))
        except json.JSONDecodeError:
            raise ValueError("Extracted JSON is not valid.")
    raise ValueError("Could not extract JSON from the response.")


class OCRTextLoader:
    """Extracts text from PDFs."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text(self):
        try:
            text = ""
            with fitz.open(self.file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {e}")


def analyze_image(file_path):
    """
    Analyzes the food image using the Groq Cloud model "llama-3.2-90b-vision-preview"
    and returns the generated message as a dictionary.
    """
    prompt = (
        "Analyze the given food image and return a structured JSON with:\n"
        "- food_detected: List of food items.\n"
        "- calories: Estimated calories.\n"
        "- nutritional_info: { carbs, proteins, fats } in grams.\n"
        "- health_warnings: Warnings related to health.\n"
        "- alternatives: Healthier alternatives."
    )
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]

        completion = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=1,
            top_p=1
        )

        message = completion.choices[0].message
        return {"role": message.role, "content": message.content}
    except Exception as e:
        logger.error(f"Groq Cloud Error in nutrition analysis: {e}")
        return {"error": f"Failed to process with Groq Cloud: {e}"}


@app.post("/nutrition")
async def nutrition(file: UploadFile = File(...)):
    """
    Handles image file upload for nutrition analysis using Groq Cloud model.
    """
    try:
        filename = file.filename.lower()
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            raise HTTPException(status_code=400, detail="Only image files (JPG, PNG) are supported")

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as temp_file:
            temp_file.write(await file.read())
            file_path = temp_file.name

        logger.info(f"Processing file for nutrition: {file_path}")

        response_data = analyze_image(file_path)
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Error processing nutrition file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)


class RecipeRequest(BaseModel):
    user_data: dict


@app.post("/recipe")
async def recipe(request: RecipeRequest):
    """
    Generate a health-focused South Indian recipe based on user health data.
    """
    try:
        prompt = (
            "Given the user's medical report, suggest a South Indian recipe that supports their health.\n"
            "Output JSON should contain:\n"
            "- recipe_name\n"
            "- ingredients (list)\n"
            "- calories (per serving)\n"
            "- diet_friendly (Low Sodium, High Fiber, etc.)\n"
            "- instructions\n"
            "- health_benefits"
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, json.dumps(request.user_data)]
        )
        parsed_output = clean_json(response.text)
        return JSONResponse(content=parsed_output)
    except Exception as e:
        logger.error(f"Error generating recipe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def extract_health_info(text: str):
    """Extract structured health information from a given text report."""
    try:
        health_data = {}
        
        # Extract Patient Information
        health_data["patient_info"] = {
            "name": re.search(r"Name:\s*(.*)", text).group(1),
            "age": int(re.search(r"Age:\s*(\d+)", text).group(1)),
            "gender": re.search(r"Gender:\s*(.*)", text).group(1),
            "report_date": re.search(r"Report Date:\s*(.*)", text).group(1),
            "patient_id": re.search(r"Patient ID:\s*(.*)", text).group(1),
            "doctor": re.search(r"Doctor:\s*(.*)", text).group(1),
        }
        
        # Extract Vital Signs & Lab Results
        health_data["vital_signs"] = {
            "blood_pressure": re.search(r"Blood Pressure:\s*(.*)\s*\(.*\)", text).group(1),
            "fasting_blood_sugar": int(re.search(r"Fasting Blood Sugar:\s*(\d+)", text).group(1)),
            "post_meal_blood_sugar": int(re.search(r"Post-Meal Blood Sugar:\s*(\d+)", text).group(1)),
            "hba1c_level": float(re.search(r"HbA1c Level:\s*(\d+\.\d+)", text).group(1)),
            "cholesterol": int(re.search(r"Cholesterol:\s*(\d+)", text).group(1)),
            "bmi": float(re.search(r"BMI:\s*(\d+\.?\d*)", text).group(1)),
        }
        
        # Extract Doctor's Observations
        observations_match = re.search(r"Doctor's Observations & Diagnosis\n(.*?)(?:\n\n|Medication & Treatment Plan)", text, re.DOTALL)
        health_data["doctor_observations"] = observations_match.group(1).strip() if observations_match else ""
        
        # Extract Medication & Treatment Plan
        medications_match = re.findall(r"- (.*?)- Once daily", text)
        health_data["medication_plan"] = medications_match if medications_match else []
        
        # Extract Dietary & Lifestyle Recommendations
        recommendations_match = re.search(r"Dietary & Lifestyle Recommendations\n(.*?)(?:\n\n|Follow-up & Next Appointment)", text, re.DOTALL)
        health_data["dietary_recommendations"] = recommendations_match.group(1).strip().split('\n') if recommendations_match else []
        
        # Extract Follow-up & Next Appointment
        health_data["follow_up"] = {
            "next_appointment": re.search(r"Next Appointment:\s*(.*)", text).group(1),
            "recommended_tests": re.findall(r"Recommended Tests:\s*(.*)\n", text),
            "doctor_contact": re.search(r"Doctor's Contact:\s*(.*)", text).group(1),
        }
        
        return health_data
    except Exception as e:
        raise ValueError(f"Error extracting health information: {e}")


@app.post("/extract_health")
async def extract_health(file: UploadFile = File(...)):
    """API endpoint to extract health information from a document."""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(await file.read())
            file_path = temp_file.name
        
        ocr_loader = OCRTextLoader(file_path)
        extracted_text = ocr_loader.extract_text()
        structured_data = extract_health_info(extracted_text)
        
        return JSONResponse(content={"extracted_health_info": structured_data})
    except Exception as e:
        logger.error(f"Error extracting health information: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


class ChatbotRequest(BaseModel):
    user_query: str
    user_info: Optional[str] = None


@app.post("/chatbot")
async def chatbot(request: ChatbotRequest):
    """
    A chatbot API that answers nutrition-related questions.
    """
    try:
        prompt = (
            "You are a chatbot for a nutrition-based application.\n"
            "Answer only nutrition-related questions. If a question is out of scope, ask the user to stay on topic.\n"
            "Consider user's health data for personalized responses."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, request.user_query, request.user_info]
        )
        return JSONResponse(content={"response": response.text})
    except Exception as e:
        logger.error(f"Error in chatbot response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


import uvicorn

if __name__ != "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
