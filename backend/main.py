from fastapi import FastAPI
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
from backend.database import engine, SessionLocal
from backend import models, database
from backend.routers import admin, authentication, technician
from model.data.test_data_generator import generate_data
from model.inference.rul_prediction_inference import predict, save_inference_report, CONFIG_PATH, OUTPUT_PATH
import pandas as pd
import threading
import time
import os
import joblib

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

model_config = joblib.load(CONFIG_PATH)

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(authentication.router)
app.include_router(admin.router)
app.include_router(technician.router)

def generate_alerts(db: Session):
    df = pd.read_csv(OUTPUT_PATH)
    
    for _, row in df.iterrows():
        if row['health_status'] == 'DEGRADING':
            alert = models.Alert(
                machine_id=row['machine_id'],
                severity=models.Severity.MEDIUM
            )

            db.query(models.Machine).filter(
                models.Machine.machine_id == row['machine_id']
            ).update({models.Machine.health_status: models.HealthStatus.DEGRADING})
        elif row['health_status'] == 'CRITICAL':
            alert = models.Alert(
                machine_id=row['machine_id'],
                severity=models.Severity.HIGH
            )

            db.query(models.Machine).filter(
                models.Machine.machine_id == row['machine_id']
            ).update({models.Machine.health_status: models.HealthStatus.CRITICAL})
        else:
            db.query(models.Machine).filter(
                models.Machine.machine_id == row['machine_id']
            ).update({models.Machine.health_status: models.HealthStatus.HEALTHY})
            continue
        
        existing_issue = db.query(models.Alert).filter(
            models.Alert.machine_id == row['machine_id'],
            models.Alert.closed == 0
        ).first()
        
        if existing_issue:
            continue
        
        db.add(alert)

def alert_automation():
    while True:
        db = SessionLocal()
        try:
            generate_data(verbose=0)
            time.sleep(0.5)
            summary_df = predict(model_config=model_config)
            save_inference_report(summary_df, verbose=0)
            generate_alerts(db)
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"Error: {e}")
        finally:
            db.close()
            
        time.sleep(5)
        
@app.on_event("startup")
def start_maintenance_system():
    if os.getenv("UVICORN_RELOAD") != "true":
        thread = threading.Thread(target=alert_automation, daemon=True)
        thread.start()