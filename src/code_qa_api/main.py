from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi_mcp import FastApiMCP

from code_qa_api.api.routes import router as api_router
from code_qa_api.core.config import settings
from code_qa_api.core.lifespan import lifespan

app = FastAPI(
    title=settings.project_name,
    version=settings.project_version,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url=None, 
    redoc_url=None,
    lifespan=lifespan, 
)
mcp = FastApiMCP(app) 
mcp.mount()
app.include_router(api_router, prefix=settings.api_prefix)
mcp.setup_server()


# Redirect root to docs
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request) -> HTMLResponse:
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url, title=f"{app.title} - Swagger UI"
    )

@app.get(app.openapi_url or "/openapi.json", include_in_schema=False)
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
