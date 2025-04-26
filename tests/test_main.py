import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Setting up revised test parameters
# Format: (param_openapi_url, param_version, expected_request_url, expected_status_code, test_id)
test_params = [
    (None, "0.1.0", "/openapi.json", 200, "default_url"), # Test default URL
    ("/custom/path/openapi.json", "1.2.3", "/custom/path/openapi.json", 200, "custom_url"), # Test custom URL
    (None, "v1.0-beta", "/openapi.json", 200, "beta_version"), # Test beta version, default URL
    ("/different.json", "0.5.0", "/different.json", 200, "another_custom_url") # Another custom URL test
]

test_ids = [p[4] for p in test_params]


@pytest.mark.parametrize(
    "param_openapi_url, param_version, expected_request_url, expected_status_code",
    [p[:4] for p in test_params], # Pass only the necessary params to the test function
    ids=test_ids
)
def test_get_open_api_endpoint(
    param_openapi_url: str | None, param_version: str, expected_request_url: str, expected_status_code: int
):
    # Arrange
    test_app = FastAPI(
        title="Test API",
        version=param_version,
        openapi_url="/openapi.json" if param_openapi_url is None else param_openapi_url,
    )
    client = TestClient(test_app) 

    @test_app.get("/")
    async def read_root():
        return {"message": "Hello World"}

    # Act
    response = client.get(expected_request_url) 

    # Assert
    assert response.status_code == expected_status_code
    if response.status_code == 200:
        assert response.headers["content-type"] == "application/json"
        openapi_data = response.json()
        assert openapi_data["info"]["title"] == test_app.title
        assert openapi_data["info"]["version"] == param_version

