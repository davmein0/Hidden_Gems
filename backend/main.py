from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.predict import router as predict_router
from backend.routers.midcaps import router as midcaps_router
from backend.routers.search import router as search_router
from backend.routers.watchlist import router as watchlist_router
from backend.routers.features import router as features_router

app = FastAPI(title="Hidden Gems Stock Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(midcaps_router)
app.include_router(search_router)
app.include_router(watchlist_router)
app.include_router(features_router)

@app.get("/")
def root():
    return {"message": "Hidden Gems API is running"}