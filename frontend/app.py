import streamlit as st
import requests
import uuid
import json

st.title("Chat with Documents")

# Generate a new session ID if not already present
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]

uploaded_files = st.file_uploader(
    "Upload PDFs, DOCX, or TXT files", type=["pdf", "docx", "txt"], accept_multiple_files=True
) 

BACKEND_URL = "http://localhost:8000"

if uploaded_files:
    files = [("files", (file.name, file.getvalue())) for file in uploaded_files]
    response = requests.post(f"{BACKEND_URL}/upload/", files=files, data={"session_id": session_id})

    if response.status_code == 200:
        st.success("Files uploaded successfully!")
    else:
        st.error(f"Failed to upload files. Error: {response.text}")

query = st.text_input("Ask a question")
if query:
    # response = requests.post(f"{BACKEND_URL}/query/", json={"session_id": session_id, "query": query})

    response = requests.post(
    f"{BACKEND_URL}/query/",
    data=json.dumps({"session_id": str(session_id), "query": str(query)}),  
    headers={"Content-Type": "application/json"}  
)

    print("response", response)
    
    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            st.error(response_data["error"])
        else:
            st.write(response_data["response"])
    else:
        st.error("Session expired. Please re-upload files.")

if st.button("End Session"):
    requests.post(f"{BACKEND_URL}/cleanup/", json={"session_id": session_id})
    st.session_state["session_id"] = str(uuid.uuid4())  # Reset session
    st.experimental_rerun()
