import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from detection.detect import detect_problem, load_detection_model
from duplicate.similarity import is_possible_duplicate, load_clip_model
from hotspots.clustering import find_hotspots
from priority.scoring import SEVERITY_SCORES
from verification.compare import verify_resolution

KNOWN_PROBLEM_TYPES = set(SEVERITY_SCORES.keys())

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("تحميل الموديلات...")
    ml_models["yolo_model"] = load_detection_model()
    ml_models["clip_model"], ml_models["clip_preprocess"] = load_clip_model()
    print("تم تحميل كل الموديلات بنجاح.")

    yield

    ml_models.clear()


app = FastAPI(lifespan=lifespan)


class ReportForDuplicateCheck(BaseModel):
    image_path: str
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    timestamp: str
    problem_type: str


class CheckDuplicateRequest(BaseModel):
    report_a: ReportForDuplicateCheck
    report_b: ReportForDuplicateCheck


class ReportPoint(BaseModel):
    id: str
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class VerifyResolutionRequest(BaseModel):
    before_image_path: str
    after_image_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    file_extension = os.path.splitext(file.filename)[1]
    safe_temp_filename = f"{uuid.uuid4().hex}{file_extension}"
    temp_path = os.path.join(tempfile.gettempdir(), safe_temp_filename)

    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        try:
            detections = detect_problem(temp_path, ml_models["yolo_model"])
        except Exception:
            raise HTTPException(status_code=400, detail="الصورة غير صالحة أو تالفة")

        return {"detections": detections}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/ai/check-duplicate")
def check_duplicate(request: CheckDuplicateRequest):
    for report in (request.report_a, request.report_b):
        if report.problem_type not in KNOWN_PROBLEM_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"نوع المشكلة '{report.problem_type}' غير معروف. الأنواع المسموحة: {sorted(KNOWN_PROBLEM_TYPES)}",
            )
        if not os.path.exists(report.image_path):
            raise HTTPException(status_code=404, detail=f"الصورة غير موجودة: {report.image_path}")

    report_a = {
        "image_path": request.report_a.image_path,
        "lat": request.report_a.lat,
        "lng": request.report_a.lng,
        "timestamp": datetime.fromisoformat(request.report_a.timestamp),
        "problem_type": request.report_a.problem_type,
    }
    report_b = {
        "image_path": request.report_b.image_path,
        "lat": request.report_b.lat,
        "lng": request.report_b.lng,
        "timestamp": datetime.fromisoformat(request.report_b.timestamp),
        "problem_type": request.report_b.problem_type,
    }

    return is_possible_duplicate(
        report_a, report_b, ml_models["clip_model"], ml_models["clip_preprocess"]
    )


@app.post("/ai/hotspots")
def hotspots(reports: List[ReportPoint]):
    reports_as_dicts = [report.model_dump() for report in reports]
    return find_hotspots(reports_as_dicts)


@app.post("/ai/verify-resolution")
def verify_resolution_endpoint(request: VerifyResolutionRequest):
    if not os.path.exists(request.before_image_path):
        raise HTTPException(status_code=404, detail=f"الصورة غير موجودة: {request.before_image_path}")
    if not os.path.exists(request.after_image_path):
        raise HTTPException(status_code=404, detail=f"الصورة غير موجودة: {request.after_image_path}")

    return verify_resolution(
        request.before_image_path,
        request.after_image_path,
        ml_models["clip_model"],
        ml_models["clip_preprocess"],
    )
