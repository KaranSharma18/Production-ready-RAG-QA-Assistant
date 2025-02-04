import streamlit as st
import requests
import uuid
import json

st.set_page_config(page_title="Chat with Documents", layout="wide")
st.title("Chat with Documents")

BACKEND_URL = "http://localhost:8000"

# Generate a session ID if not already present
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

session_id = st.session_state["session_id"]

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDFs, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    file_data = [("files", (file.name, file.getvalue())) for file in uploaded_files]
    response = requests.post(f"{BACKEND_URL}/upload/", files=file_data, data={"session_id": session_id})

    if response.status_code == 200:
        st.success("Files uploaded successfully!")
    else:
        st.error(f"Failed to upload files. Error: {response.text}")

def get_chat_history(session_id):
    """Fetch chat history from backend."""
    response = requests.post(f"{BACKEND_URL}/chat_history/", json={"session_id": session_id})
    if response.status_code == 200:
        return response.json().get("chat_history", [])  # Return chat history or empty list
    return []

# Display chat history
st.subheader("Chat History")
chat_container = st.container()

chat_history = get_chat_history(session_id)  # Fetch chat history from backend

with chat_container:
    if chat_history:
        for chat in chat_history:
            chat_data = json.loads(chat)  # Convert JSON string to dictionary
            st.markdown(f"**You:** {chat_data['question']}")
            st.markdown(f"**Bot:** {chat_data['answer']}")
            st.markdown("---")
    else:
        st.markdown("*No chat history yet.*")  # Display message if chat history is empty

# User query input
query = st.text_input("Ask a question")

if query:
    response = requests.post(
        f"{BACKEND_URL}/query/",
        json={"session_id": session_id, "query": query},
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            st.error(response_data["error"])
        else:
            st.session_state["answer"] = response_data.get("response", "No response received.")
            st.experimental_rerun()
    else:
        st.error("Session expired. Please re-upload files.")

# Display the latest answer
if st.session_state["answer"]:
    st.subheader("Response")
    st.write(st.session_state["answer"])

# End session button
if st.button("End Session"):
    requests.post(f"{BACKEND_URL}/cleanup/", json={"session_id": session_id})
    delete_session(session_id)  # Delete session & embeddings
    st.session_state["session_id"] = str(uuid.uuid4())  # Reset session
    st.experimental_rerun()
