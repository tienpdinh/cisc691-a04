#!/usr/bin/env python3
"""
Document Upload Script

Uploads all documents from the files/ directory to the RAG API
to populate ChromaDB with vector embeddings.
"""

from __future__ import annotations

import requests
import sys
from pathlib import Path
import time

def upload_documents(api_base="http://localhost:8001", files_dir="files"):
    """Upload all documents from files directory to RAG API."""
    
    print("🚀 DOCUMENT UPLOAD PROCESS")
    print("=" * 50)
    
    files_path = Path(files_dir)
    if not files_path.exists():
        print(f"❌ Directory {files_dir} does not exist")
        return False
    
    # Get all PDF and DOCX files
    pdf_files = list(files_path.glob("*.pdf"))
    docx_files = list(files_path.glob("*.docx"))
    all_files = pdf_files + docx_files
    
    print(f"📁 Found {len(pdf_files)} PDF files and {len(docx_files)} DOCX files")
    print(f"📄 Total files to upload: {len(all_files)}")
    
    if not all_files:
        print("❌ No PDF or DOCX files found to upload")
        return False
    
    # Check if API is available
    try:
        response = requests.get(f"{api_base}/health", timeout=10)
        if response.status_code != 200:
            print(f"❌ RAG API not available at {api_base}")
            print("💡 Make sure to run: docker-compose up -d")
            return False
        print(f"✅ RAG API is available at {api_base}")
    except Exception as e:
        print(f"❌ Cannot connect to RAG API: {e}")
        print("💡 Make sure to run: docker-compose up -d")
        return False
    
    print("\n📤 Starting document upload...")
    
    successful_uploads = 0
    failed_uploads = []
    
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] Uploading: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as file:
                files = {'file': (file_path.name, file, 'application/pdf' if file_path.suffix == '.pdf' else 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')}
                
                response = requests.post(
                    f"{api_base}/upload-document",
                    files=files,
                    timeout=60  # Allow 60 seconds for upload and processing
                )
                
                if response.status_code == 200:
                    print(f"    ✅ Successfully uploaded {file_path.name}")
                    successful_uploads += 1
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    print(f"    ❌ Failed to upload {file_path.name}: {error_msg}")
                    failed_uploads.append((file_path.name, error_msg))
                    
        except Exception as e:
            print(f"    ❌ Error uploading {file_path.name}: {e}")
            failed_uploads.append((file_path.name, str(e)))
        
        # Small delay between uploads to avoid overwhelming the API
        if i < len(all_files):
            time.sleep(2)
    
    print(f"\n📊 UPLOAD SUMMARY")
    print(f"✅ Successful uploads: {successful_uploads}")
    print(f"❌ Failed uploads: {len(failed_uploads)}")
    
    if failed_uploads:
        print(f"\n❌ Failed files:")
        for filename, error in failed_uploads:
            print(f"    - {filename}: {error}")
    
    return successful_uploads > 0

def main():
    """Main function."""
    success = upload_documents()
    
    if success:
        print(f"\n🎉 Document upload completed!")
        print(f"💡 Now run the verification script to check ingestion:")
        print(f"   python benchmarks/scripts/check_document_ingestion.py")
    else:
        print(f"\n❌ Document upload failed!")
        print(f"💡 Check that your RAG API is running with:")
        print(f"   docker-compose up -d")
        print(f"   docker-compose ps")
        sys.exit(1)

if __name__ == "__main__":
    main()