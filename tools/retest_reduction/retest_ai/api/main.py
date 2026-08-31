from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router
from ..models.service import MLService

app = FastAPI(
    title="ATE Retest-Benefit Prediction AI",
    description="Option B Supervised ML API for predicting P(RETEST_BENEFICIAL) in semiconductor ATE testing.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

@app.on_event("startup")
def startup_event():
    """Initializes the ML model artifacts on application startup."""
    print("Initializing Shared ML Service for FastAPI...")
    MLService.get_instance()
    print("Shared ML Service initialized successfully.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
