from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from code_qa_api.api.routes import router as api_router
from code_qa_api.core.config import settings
from code_qa_api.core.lifespan import lifespan

app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default ReDoc
    lifespan=lifespan,  # Use the custom lifespan context manager
)

# Include API routes
app.include_router(api_router, prefix=settings.api_prefix)


# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    # Point to the actual docs path which FastAPI generates
    # internally even if docs_url is None
    # Alternatively, serve custom Swagger HTML here if needed
    return RedirectResponse(url="/docs")


# Custom route for Swagger UI if docs_url is None
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request) -> HTMLResponse:
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=app.title + " - Swagger UI",
    )


# Expose the OpenAPI schema
@app.get(app.openapi_url if app.openapi_url else "/openapi.json", include_in_schema=False)
async def get_open_api_endpoint() -> JSONResponse:
    return JSONResponse(
        get_openapi(
            title=app.title,
            version=app.version,
            routes=app.routes,
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
