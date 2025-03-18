import pytest
import os
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api_server import app

@pytest.fixture
def test_config():
    return {
        "cache_duration_days": 7,
        "max_results_per_page": 100,
        "supported_mime_types": [
            "application/vnd.google-apps.document",
            "application/vnd.google-apps.spreadsheet",
            "text/plain"
        ],
        "processor_url": "http://localhost:8001"
    }

@pytest.fixture
def mock_credentials():
    return {
        "token": "mock_token",
        "refresh_token": "mock_refresh_token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "mock_client_id",
        "client_secret": "mock_client_secret",
        "scopes": ["https://www.googleapis.com/auth/drive.readonly"]
    }

@pytest.fixture
async def test_client(test_config, mock_credentials, tmp_path):
    # Create mock config and credential files
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f)
    
    token_path = tmp_path / "token.json"
    with open(token_path, "w") as f:
        json.dump(mock_credentials, f)
    
    # Set environment variables
    os.environ["CONFIG_PATH"] = str(config_path)
    os.environ["GOOGLE_TOKEN_PATH"] = str(token_path)
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_list_documents(test_client, mocker):
    # Mock Google Drive API response
    mock_files = {
        "files": [
            {
                "id": "doc1",
                "name": "Test Document 1",
                "mimeType": "application/vnd.google-apps.document",
                "createdTime": "2025-03-18T00:00:00.000Z",
                "modifiedTime": "2025-03-18T01:00:00.000Z"
            },
            {
                "id": "doc2",
                "name": "Test Document 2",
                "mimeType": "text/plain",
                "createdTime": "2025-03-18T02:00:00.000Z",
                "modifiedTime": "2025-03-18T03:00:00.000Z"
            }
        ],
        "nextPageToken": None
    }
    
    # Mock the drive service
    mocker.patch("api_server.build").return_value.files().list().execute.return_value = mock_files
    
    # Test document listing
    response = await test_client.get("/documents")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["documents"]) == 2
    assert data["documents"][0]["id"] == "doc1"
    assert data["documents"][1]["id"] == "doc2"
    assert data["next_page_token"] is None
    assert data["total_count"] == 2

@pytest.mark.asyncio
async def test_get_status(test_client, mocker):
    # Mock Google Drive API responses
    mock_total_files = {"files": [{"id": "doc1"}, {"id": "doc2"}, {"id": "doc3"}]}
    mock_classified_files = {"files": [{"id": "doc1"}, {"id": "doc2"}]}
    
    # Set up mock responses
    drive_mock = mocker.patch("api_server.build").return_value.files()
    drive_mock.list().execute.side_effect = [
        mock_total_files,
        mock_classified_files
    ]
    
    # Test status endpoint
    response = await test_client.get("/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total_documents"] == 3
    assert data["classified_count"] == 2
    assert data["pending_count"] == 1

@pytest.mark.asyncio
async def test_get_report(test_client, mocker):
    # Mock classified documents
    mock_files = {
        "files": [
            {
                "id": "doc1",
                "name": "Document 1",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T00:00:00.000Z",
                    "overall_confidence": "90",
                    "categories": "confidential,sensitive"
                }
            },
            {
                "id": "doc2",
                "name": "Document 2",
                "properties": {
                    "classified": "true",
                    "classification_date": "2025-03-18T01:00:00.000Z",
                    "overall_confidence": "75",
                    "categories": "internal"
                }
            }
        ]
    }
    
    # Mock the drive service
    mocker.patch("api_server.build").return_value.files().list().execute.return_value = mock_files
    
    # Test report generation
    response = await test_client.get("/report?format=json")
    assert response.status_code == 200
    
    data = response.json()
    assert data["format"] == "json"
    assert "report_content" in data
    
    # Parse report content
    report = json.loads(data["report_content"])
    assert len(report) == 2
    assert report[0]["id"] == "doc1"
    assert report[1]["id"] == "doc2"

@pytest.mark.asyncio
async def test_pagination(test_client, mocker):
    # Mock paginated responses
    mock_page1 = {
        "files": [{"id": "doc1", "name": "Document 1"}],
        "nextPageToken": "page2_token"
    }
    mock_page2 = {
        "files": [{"id": "doc2", "name": "Document 2"}],
        "nextPageToken": None
    }
    
    # Set up mock responses
    drive_mock = mocker.patch("api_server.build").return_value.files()
    drive_mock.list().execute.side_effect = [mock_page1, mock_page2]
    
    # Test first page
    response = await test_client.get("/documents?page_size=1")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["next_page_token"] == "page2_token"
    
    # Test second page
    response = await test_client.get("/documents?page_size=1&page_token=page2_token")
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["documents"]) == 1
    assert data["next_page_token"] is None

@pytest.mark.asyncio
async def test_error_handling(test_client, mocker):
    # Mock API error
    def mock_api_error(*args, **kwargs):
        raise Exception("API Error")
    
    mocker.patch("api_server.build").return_value.files().list().execute.side_effect = mock_api_error
    
    # Test error response
    response = await test_client.get("/documents")
    assert response.status_code == 500
    assert "error" in response.json()

if __name__ == "__main__":
    pytest.main(["-v", __file__])