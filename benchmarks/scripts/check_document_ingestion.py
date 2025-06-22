#!/usr/bin/env python3
"""
Document Ingestion Verification Script

Checks if documents are properly ingested into ChromaDB and provides
detailed information about the current state of the vector database.
"""

import sys
from pathlib import Path
import chromadb
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_chromadb_collections(host="localhost", port=8000):
    """Check ChromaDB collections and document counts."""
    print("🔍 CHROMADB DOCUMENT INGESTION STATUS")
    print("=" * 50)
    
    try:
        # Connect to ChromaDB using v2 API
        client = chromadb.HttpClient(host=host, port=port)
        print(f"✅ Connected to ChromaDB at {host}:{port}")
        
        # List all collections
        collections = client.list_collections()
        print(f"📚 Found {len(collections)} collections")
        
        if not collections:
            print("❌ No collections found in ChromaDB")
            print("\n💡 This means no documents have been ingested yet.")
            return False
        
        # Check each collection
        total_documents = 0
        for collection in collections:
            try:
                count = collection.count()
                total_documents += count
                print(f"    📁 Collection '{collection.name}': {count} documents")
                
                # Show sample data if documents exist
                if count > 0:
                    # Get a few sample documents
                    sample = collection.peek(limit=3)
                    print(f"        📄 Sample documents:")
                    for i, doc_id in enumerate(sample.get('ids', [])[:3]):
                        metadata = sample.get('metadatas', [{}])[i] if i < len(sample.get('metadatas', [])) else {}
                        source = metadata.get('source', 'unknown')
                        text_preview = metadata.get('text', '')[:100] + '...' if metadata.get('text') else 'No text'
                        print(f"            [{i+1}] ID: {doc_id}")
                        print(f"                Source: {source}")
                        print(f"                Text: {text_preview}")
                else:
                    print(f"        ⚠️  Collection '{collection.name}' is empty")
                    
            except Exception as e:
                print(f"        ❌ Error checking collection '{collection.name}': {e}")
        
        print(f"\n📊 TOTAL DOCUMENTS ACROSS ALL COLLECTIONS: {total_documents}")
        
        if total_documents == 0:
            print("\n❌ NO DOCUMENTS FOUND IN ANY COLLECTION")
            print("This explains why RAG queries return 'no relevant information'")
            return False
        else:
            print(f"\n✅ Found {total_documents} documents in ChromaDB")
            return True
            
    except Exception as e:
        print(f"❌ Failed to connect to ChromaDB: {e}")
        print("\n💡 Make sure ChromaDB container is running:")
        print("   docker-compose up chromadb")
        return False

def check_document_directories():
    """Check document directories for source files."""
    print("\n🗂️  DOCUMENT DIRECTORIES CHECK")
    print("=" * 50)
    
    # Common document directories to check
    dirs_to_check = [
        "files",
        "data", 
        "documents",
        "src/data",
        "cleaned_text",
        "embeddings"
    ]
    
    found_dirs = []
    
    for dir_name in dirs_to_check:
        dir_path = project_root / dir_name
        if dir_path.exists():
            files = list(dir_path.glob("*"))
            pdf_files = list(dir_path.glob("*.pdf"))
            txt_files = list(dir_path.glob("*.txt"))
            json_files = list(dir_path.glob("*.json"))
            
            print(f"📁 {dir_name}/")
            print(f"    📄 Total files: {len(files)}")
            print(f"    📕 PDF files: {len(pdf_files)}")
            print(f"    📝 TXT files: {len(txt_files)}")
            print(f"    📋 JSON files: {len(json_files)}")
            
            if files:
                print(f"    📋 Sample files:")
                for file in files[:5]:  # Show first 5 files
                    print(f"        - {file.name}")
                if len(files) > 5:
                    print(f"        ... and {len(files) - 5} more")
            
            found_dirs.append(dir_name)
            print()
    
    if not found_dirs:
        print("❌ No document directories found")
        print("💡 You may need to add documents to ingest")
    else:
        print(f"✅ Found {len(found_dirs)} document directories: {', '.join(found_dirs)}")
    
    return found_dirs

def check_ingestion_pipeline():
    """Check if ingestion pipeline components exist."""
    print("⚙️  INGESTION PIPELINE CHECK")
    print("=" * 50)
    
    # Check for ingestion-related files
    pipeline_files = [
        "src/embedding_loader.py",
        "src/text_processor.py", 
        "src/document_processor.py",
        "main.py",
        "config.json",
        "config.local.json"
    ]
    
    found_files = []
    
    for file_name in pipeline_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"✅ Found: {file_name}")
            found_files.append(file_name)
        else:
            print(f"❌ Missing: {file_name}")
    
    print(f"\n📊 Pipeline files found: {len(found_files)}/{len(pipeline_files)}")
    
    # Check configuration
    config_path = project_root / "config.local.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"\n⚙️  Configuration Check:")
            print(f"    ChromaDB Host: {config.get('chromadb', {}).get('host', 'Not set')}")
            print(f"    ChromaDB Port: {config.get('chromadb', {}).get('port', 'Not set')}")
            print(f"    Collection Name: {config.get('chromadb', {}).get('collection_name', 'Not set')}")
            
        except Exception as e:
            print(f"⚠️  Could not read config file: {e}")
    
    return found_files

def provide_ingestion_guidance(has_documents, found_dirs, pipeline_files):
    """Provide guidance on how to fix document ingestion."""
    print("\n💡 INGESTION GUIDANCE")
    print("=" * 50)
    
    if not has_documents:
        print("🔧 TO FIX DOCUMENT INGESTION:")
        print()
        
        if found_dirs:
            print("1. ✅ You have document directories with files")
            print("2. 🔄 Run the document ingestion process:")
            print("   • Check if there's a main.py or ingestion script")
            print("   • Look for commands like:")
            print("     python main.py --ingest")
            print("     python src/embedding_loader.py")
            print("     docker-compose exec rag-api python main.py")
            print()
            
        else:
            print("1. 📁 First, add documents to ingest:")
            print("   • Create a 'files/' or 'data/' directory")
            print("   • Add your PDF or text files there")
            print()
            
        print("3. 🔍 Check the API for ingestion endpoints:")
        print("   curl -X POST http://localhost:8001/upload")
        print("   curl -X GET http://localhost:8001/docs")
        print()
        
        print("4. 🐳 Ensure all containers are running:")
        print("   docker-compose up -d")
        print("   docker-compose ps")
        print()
        
        print("5. 📊 Re-run this script to verify ingestion:")
        print("   python benchmarks/scripts/check_document_ingestion.py")
        
    else:
        print("✅ Documents are properly ingested!")
        print("Your RAG system should work correctly now.")

def main():
    """Main function to check document ingestion status."""
    print("🚀 DOCUMENT INGESTION VERIFICATION")
    print("=" * 80)
    print()
    
    # Check ChromaDB collections
    has_documents = check_chromadb_collections()
    
    # Check document directories
    found_dirs = check_document_directories()
    
    # Check ingestion pipeline
    pipeline_files = check_ingestion_pipeline()
    
    # Provide guidance
    provide_ingestion_guidance(has_documents, found_dirs, pipeline_files)
    
    print("\n" + "=" * 80)
    if has_documents:
        print("✅ INGESTION STATUS: DOCUMENTS FOUND")
    else:
        print("❌ INGESTION STATUS: NO DOCUMENTS FOUND")
    print("=" * 80)

if __name__ == "__main__":
    main()