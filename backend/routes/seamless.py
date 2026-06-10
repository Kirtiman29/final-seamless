import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from services.seamless_generator import make_seamless

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"


@router.post("/generate-seamless")
async def generate_seamless(request: Request, image: UploadFile | None = File(default=None)):
    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "message": "Image missing",
            },
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{unique_id}.png"
    output_path = OUTPUT_DIR / f"{unique_id}_seamless.png"
    preview_path = OUTPUT_DIR / f"{unique_id}_preview.png"

    contents = await image.read()

    with open(upload_path, "wb") as upload_file:
        upload_file.write(contents)

    base_url = str(request.base_url).rstrip("/")

    try:
        result = make_seamless(str(upload_path), str(output_path), str(preview_path))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": str(exc),
                "code": "generation_failed",
            },
        ) from exc

    validation = result["validation"]
    if not validation.get("ai_inpaint_used"):
        raise HTTPException(
            status_code=502,
            detail={
                "success": False,
                "message": (
                    "Vertex AI inpainting did not run. "
                    f"{validation.get('vertex_failure_message') or 'Check Google Cloud authentication.'}"
                ),
                "code": "vertex_ai_inpainting_not_used",
                "validation": validation,
            },
        )

    response = {
        "success": True,
        "original_url": f"{base_url}/uploads/{upload_path.name}",
        "tile_url": f"{base_url}/outputs/{output_path.name}",
        "preview_url": f"{base_url}/outputs/{preview_path.name}",
        "validation": validation,
    }

    return response


@router.post("/generate-seamless-custom")
async def generate_seamless_custom(
    request: Request,
    image: UploadFile = File(...),
    horizontal_band: int = Form(...),
    vertical_band: int = Form(...)
):
    if image is None:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "message": "Image missing",
            },
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unique_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{unique_id}.png"
    output_path = OUTPUT_DIR / f"{unique_id}_seamless.png"
    preview_path = OUTPUT_DIR / f"{unique_id}_preview.png"

    contents = await image.read()

    with open(upload_path, "wb") as upload_file:
        upload_file.write(contents)

    base_url = str(request.base_url).rstrip("/")

    try:
        result = make_seamless(
            str(upload_path),
            str(output_path),
            str(preview_path),
            custom_horizontal_band=horizontal_band,
            custom_vertical_band=vertical_band
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": str(exc),
                "code": "generation_failed",
            },
        ) from exc

    validation = result["validation"]
    if not validation.get("ai_inpaint_used"):
        raise HTTPException(
            status_code=502,
            detail={
                "success": False,
                "message": (
                    "Vertex AI inpainting did not run. "
                    f"{validation.get('vertex_failure_message') or 'Check Google Cloud authentication.'}"
                ),
                "code": "vertex_ai_inpainting_not_used",
                "validation": validation,
            },
        )

    response = {
        "success": True,
        "original_url": f"{base_url}/uploads/{upload_path.name}",
        "tile_url": f"{base_url}/outputs/{output_path.name}",
        "preview_url": f"{base_url}/outputs/{preview_path.name}",
        "validation": validation,
    }

    return response
