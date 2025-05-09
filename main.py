from fastapi import FastAPI, HTTPException, status, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI(
    title="Medical Prediction API",
    description="API for predicting medical conditions using XGBoost model",
    version="1.0.0"
)

# Load templates directory
templates = Jinja2Templates(directory="templates")

# Load the model
try:
    model = joblib.load("xgb_model.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file 'xgb_model.pkl' not found.")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    gender: int = Form(...),  # 0 for Female, 1 for Male
    hemoglobin: float = Form(...),
    rbc: float = Form(...),
    wbc: float = Form(...),
    hematocrit: float = Form(...),
    platelet_count: float = Form(...),
    fasting_blood_sugar: float = Form(...),
    hemoglobin_a1c: float = Form(...),
    serum_creatinine: float = Form(...),
    blood_urea_nitrogen: float = Form(...),
    glomerular_filtration_rate: float = Form(...),
    alt: float = Form(...),  # Alanine Aminotransferase
    ast: float = Form(...),  # Aspartate Aminotransferase
    alp: float = Form(...),  # Alkaline Phosphatase
    total_bilirubin: float = Form(...),
    total_cholesterol: float = Form(...),
    ldl: float = Form(...),  # Low-Density Lipoprotein
    hdl: float = Form(...),  # High-Density Lipoprotein
    triglycerides: float = Form(...),
    sodium: float = Form(...),
    potassium: float = Form(...),
    calcium: float = Form(...),
    iron: float = Form(...),
    ferritin: float = Form(...),
    urine_protein: int = Form(...),
    urine_glucose: int = Form(...),
    red_blood_cells_in_urine: float = Form(...)
):
    try:
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Hemoglobin (g/dL)': hemoglobin,
            'RBC (million/µL)': rbc,
            'WBC (cells/µL)': wbc,
            'Hematocrit (%)': hematocrit,
            'Platelet Count (platelets/µL)': platelet_count,
            'Fasting Blood Sugar (mg/dL)': fasting_blood_sugar,
            'Hemoglobin A1C (%)': hemoglobin_a1c,
            'Serum Creatinine (mg/dL)': serum_creatinine,
            'Blood Urea Nitrogen (BUN) (mg/dL)': blood_urea_nitrogen,
            'Glomerular Filtration Rate (GFR) (mL/min/1.73m²)': glomerular_filtration_rate,
            'Alanine Aminotransferase (ALT) (U/L)': alt,
            'Aspartate Aminotransferase (AST) (U/L)': ast,
            'Alkaline Phosphatase (ALP) (U/L)': alp,
            'Total Bilirubin (mg/dL)': total_bilirubin,
            'Total Cholesterol (mg/dL)': total_cholesterol,
            'LDL (mg/dL)': ldl,
            'HDL (mg/dL)': hdl,
            'Triglycerides (mg/dL)': triglycerides,
            'Sodium (mEq/L)': sodium,
            'Potassium (mEq/L)': potassium,
            'Calcium (mg/dL)': calcium,
            'Iron (µg/dL)': iron,
            'Ferritin (ng/mL)': ferritin,
            'Urine Protein': urine_protein,
            'Urine Glucose': urine_glucose,
            'Red Blood Cells in Urine (cells/HPF)': red_blood_cells_in_urine,
        }])

        prediction = model.predict(input_data)[0]
        message = "⚠️ Critical Condition Detected!" if prediction == 1 else "✅ Condition is Normal"
        return templates.TemplateResponse("form.html", {
            "request": request,
            "result": int(prediction),
            "message": message
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


