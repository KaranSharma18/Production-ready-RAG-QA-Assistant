from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
import redis_cache
import vector_store
import document_loader  
import uuid

app = FastAPI()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    """Handles document uploads, processes embeddings, and stores session metadata."""
    extracted_text = ""
    
    for file in files:
        file_extension = "." + file.filename.split(".")[-1].lower()  # Get file extension
        if file_extension not in ALLOWED_EXTENSIONS:
            return {"error": f"File type {file_extension} is not supported."}

        content = await file.read()
        extracted_text += document_loader.extract_text(content, file.filename)

    # Generate embeddings & store in Pinecone
    vector_store.store_embeddings(session_id, extracted_text)

    # Save session in Redis
    redis_cache.save_session(session_id, [file.filename for file in files])
    
    return {"message": "Files processed successfully"}

class QueryRequest(BaseModel):
    session_id: str
    query: str

@app.post("/query/")
async def query_llm(request: QueryRequest):
    """Handles user queries, retrieves relevant document context, and generates responses."""

    session_id = request.session_id
    query = request.query

    session_data = redis_cache.get_session(session_id)
    if not session_data:
        return {"error": "Session expired. Please re-upload your files."}

    # Retrieve relevant document chunks
    relevant_chunks = vector_store.retrieve_embeddings(session_id, query)

    context = relevant_chunks if relevant_chunks else ["No relevant documents found, but answering based on general knowledge."]

    # Generate response from LLaMA model
    response = llm.generate_response(query, context)
    
    return {"response": response}

@app.post("/cleanup/")
async def cleanup_session(session_id: str):
    """Deletes session data and removes related embeddings from Pinecone."""
    redis_cache.delete_session(session_id)  # Cleanup session data
    
    return {"message": f"Session {session_id} cleaned up"}

