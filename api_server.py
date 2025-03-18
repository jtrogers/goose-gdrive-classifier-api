from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os
import json
from datetime import datetime, timedelta
import aiohttp
import asyncio

app = FastAPI()

# Models
class DocumentList(BaseModel):
    documents: List[dict]
    next_page_token: Optional[str]
    total_count: int

class StatusResponse(BaseModel):
    total_documents: int
    classified_count: int
    pending_count: int
    last_update: str

class ReportResponse(BaseModel):
    report_content: str
    generated_at: str
    format: str

# Configuration
class Config:
    def __init__(self):
        self.cache_duration_days = int(os.getenv('CACHE_DURATION_DAYS', '7'))
        self.max_results_per_page = int(os.getenv('MAX_RESULTS_PER_PAGE', '100'))
        self.supported_mime_types = os.getenv('SUPPORTED_MIME_TYPES', '').split(',')
        self.processor_url = os.getenv('PROCESSOR_URL')

config = Config()

# Google Drive client setup
def get_drive_service():
    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
    if not os.path.exists(token_path):
        raise HTTPException(status_code=500, detail="Google OAuth token not found")
    
    with open(token_path, 'r') as token:
        token_data = json.load(token)
        credentials = Credentials.from_authorized_user_info(token_data)
        return build('drive', 'v3', credentials=credentials)

# Routes
@app.get("/documents", response_model=DocumentList)
async def list_documents(
    folder_id: Optional[str] = None,
    page_token: Optional[str] = None,
    page_size: Optional[int] = Query(default=50, le=1000),
    include_processed: bool = False
):
    """List documents in Google Drive that can be classified."""
    service = get_drive_service()
    
    # Build query
    query_parts = []
    
    # Filter by folder
    if folder_id:
        query_parts.append(f"'{folder_id}' in parents")
    
    # Filter by mime types
    if config.supported_mime_types:
        mime_types = [f"mimeType = '{mime}'" for mime in config.supported_mime_types]
        query_parts.append(f"({' or '.join(mime_types)})")
    
    # Filter out processed documents unless included
    if not include_processed:
        query_parts.append("not properties has { key='classified' and value='true' }")
    
    query = " and ".join(query_parts) if query_parts else None
    
    try:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name, mimeType, createdTime, modifiedTime, owners, size)',
            pageToken=page_token,
            pageSize=min(page_size, config.max_results_per_page)
        ).execute()
        
        return DocumentList(
            documents=response.get('files', []),
            next_page_token=response.get('nextPageToken'),
            total_count=len(response.get('files', []))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current classification status."""
    service = get_drive_service()
    
    try:
        # Get total documents
        total_response = service.files().list(
            q=" or ".join(f"mimeType = '{mime}'" for mime in config.supported_mime_types),
            fields="files(id)"
        ).execute()
        
        # Get classified documents
        classified_response = service.files().list(
            q="properties has { key='classified' and value='true' }",
            fields="files(id)"
        ).execute()
        
        total_count = len(total_response.get('files', []))
        classified_count = len(classified_response.get('files', []))
        
        return StatusResponse(
            total_documents=total_count,
            classified_count=classified_count,
            pending_count=total_count - classified_count,
            last_update=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report", response_model=ReportResponse)
async def get_report(
    format: str = Query(default="markdown", enum=["markdown", "json"]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Get a classification report."""
    service = get_drive_service()
    
    try:
        # Build query for classified documents
        query_parts = ["properties has { key='classified' and value='true' }"]
        
        if start_date:
            query_parts.append(f"modifiedTime >= '{start_date}'")
        if end_date:
            query_parts.append(f"modifiedTime <= '{end_date}'")
            
        query = " and ".join(query_parts)
        
        # Get classified documents
        response = service.files().list(
            q=query,
            fields="files(id, name, properties)",
            pageSize=1000
        ).execute()
        
        files = response.get('files', [])
        
        # Generate report
        if format == "markdown":
            report_content = _generate_markdown_report(files)
        else:
            report_content = json.dumps(files, indent=2)
            
        return ReportResponse(
            report_content=report_content,
            generated_at=datetime.now().isoformat(),
            format=format
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_markdown_report(files: List[dict]) -> str:
    """Generate a markdown format report."""
    report_parts = [
        "# Document Classification Report",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\nTotal Documents: {len(files)}",
        "\n## Summary"
    ]
    
    # Aggregate statistics
    categories = {}
    confidence_levels = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    
    for file in files:
        props = file.get('properties', {})
        for cat in props.get('categories', '').split(','):
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
        
        confidence = int(props.get('overall_confidence', 0))
        if confidence >= 90:
            confidence_levels['HIGH'] += 1
        elif confidence >= 70:
            confidence_levels['MEDIUM'] += 1
        else:
            confidence_levels['LOW'] += 1
    
    # Add statistics to report
    report_parts.extend([
        "\n### Categories",
        *[f"- {cat}: {count}" for cat, count in categories.items()],
        "\n### Confidence Levels",
        *[f"- {level}: {count}" for level, count in confidence_levels.items()]
    ])
    
    return "\n".join(report_parts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))