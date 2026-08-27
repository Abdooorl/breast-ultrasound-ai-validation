"""FastAPI entry point for the research demo.

This is intentionally minimal during Phase 1. The inference endpoint will be
implemented after the external-validation pipeline is stable.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Breast Ultrasound AI Validation API",
    version="0.1.0",
    description="Research prototype only — not intended for clinical diagnosis.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "use": "research-only"}
