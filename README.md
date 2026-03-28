# ML Reliability & Deployment System (MLOps Project)

An end-to-end MLOps project that serves a machine learning model and ensures its reliability in production using data drift detection and monitoring.

---

## Live Demo

👉 API Docs (Swagger UI):  
https://mlops-ml-reliability-system.onrender.com/docs

---

## GitHub Repository

👉 https://github.com/Yuvraj-Dixit6265/mlops-ml-reliability-system

---

##  Problem Statement

In production, ML models may continue running without errors, but their prediction quality degrades due to changes in input data (data drift). Traditional monitoring tools do not detect this.

---

## Solution

Built a system that:
- Logs incoming data and predictions  
- Compares production data with training data  
- Detects drift using statistical methods  
- Alerts when model behavior becomes unreliable  

---

## Features

- FastAPI-based model serving  
-  Input validation using Pydantic  
-  Data logging (CSV-based)  
-  Data drift detection  
-  Alert system  
-  Automated testing using Pytest  
-  Docker containerization  
-  CI pipeline using GitHub Actions  
-  Cloud deployment using Render  

---

##  Architecture

User → FastAPI → ML Model → Prediction  
          ↓  
        Logging System  
          ↓  
        Drift Detection  
          ↓  
        Alert System  

---

## Tech Stack

- Python  
- FastAPI  
- Scikit-learn  
- Pandas  
- Pydantic  
- Pytest  
- Docker  
- GitHub Actions (CI/CD)  
- Render (Deployment)  

---

## How to Run Locally

```bash
git clone https://github.com/Yuvraj-Dixit6265/mlops-ml-reliability-system
cd mlops-ml-reliability-system

pip install -r requirements.txt

uvicorn app.main:app --reload