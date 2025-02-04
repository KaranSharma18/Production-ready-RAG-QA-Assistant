import streamlit as st
import requests
import uuid
import json
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentChatApp:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        """
        Initialize the Streamlit document chat application.
        
        Args:
            backend_url (str): Base URL for the backend API
        """
        self.BACKEND_URL = backend_url
        self._initialize_session_state()

    def _initialize_session_state(self):
        """
        Initialize or reset session state variables.
        """
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = str(uuid.uuid4())
        
        # Initialize other session state variables with default values
        session_state_defaults = {
            "answer": "",
            "query": "",
            "last_query": "",  # Add this new state variable
            "uploaded_files": [],
            "error": None
        }
        
        for key, default_value in session_state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def upload_files(self, uploaded_files: List) -> bool:
        """
        Upload files to the backend.
        
        Args:
            uploaded_files (List): List of uploaded files
        
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            file_data = [("files", (file.name, file.getvalue())) for file in uploaded_files]
            response = requests.post(
                f"{self.BACKEND_URL}/upload/", 
                files=file_data, 
                data={"session_id": st.session_state["session_id"]}
            )
            
            if response.status_code == 200:
                st.success("Files uploaded successfully!")
                st.session_state["uploaded_files"] = uploaded_files
                return True
            else:
                st.error(f"Failed to upload files. Error: {response.text}")
                return False
        except requests.RequestException as e:
            st.error(f"Network error during file upload: {e}")
            return False

    def get_chat_history(self) -> List[Dict]:
        """
        Fetch chat history from backend.
        
        Returns:
            List[Dict]: List of chat history entries
        """
        try:
            response = requests.post(
                f"{self.BACKEND_URL}/chat_history/", 
                json={"session_id": st.session_state["session_id"]}
            )
            
            if response.status_code == 200:
                return response.json().get("chat_history", [])
            return []
        except requests.RequestException as e:
            logger.error(f"Error fetching chat history: {e}")
            return []

    def submit_query(self, query: str) -> Optional[str]:
        """
        Submit a query to the backend.
        
        Args:
            query (str): User's query
        
        Returns:
            Optional[str]: Response from backend or None
        """
        try:
            response = requests.post(
                f"{self.BACKEND_URL}/query/",
                json={
                    "session_id": st.session_state["session_id"], 
                    "query": query
                },
                headers={"Content-Type": "application/json"},
                timeout=30  # Add a timeout to prevent hanging
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                    return None
                return response_data.get("response", "No response received.")
            else:
                st.error("Session expired. Please re-upload files.")
                return None
        except requests.RequestException as e:
            st.error(f"Network error during query submission: {e}")
            return None

    def end_session(self):
        """
        End the current session and reset session state.
        """
        try:
            requests.post(
                f"{self.BACKEND_URL}/cleanup/", 
                json={"session_id": st.session_state["session_id"]}
            )
        except requests.RequestException as e:
            logger.warning(f"Error during session cleanup: {e}")
        
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_session_state()
        st.rerun()  # Only rerun after completely clearing the session

    def render_chat_history(self):
        """
        Render the chat history in the Streamlit app.
        """
        st.subheader("Chat History")
        chat_container = st.container()

        with chat_container:
            chat_history = self.get_chat_history()
            
            if chat_history:
                for chat in chat_history:
                    try:
                        chat_data = json.loads(chat)
                        st.markdown(f"**You:** {chat_data['question']}")
                        st.markdown(f"**Bot:** {chat_data['answer']}")
                        st.markdown("---")
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing chat history: {e}")
            else:
                st.markdown("*No chat history yet.*")

    def run(self):
        """
        Main method to run the Streamlit application.
        """
        st.set_page_config(page_title="Chat with Documents", layout="wide")
        st.title("Chat with Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDFs, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files and uploaded_files != st.session_state["uploaded_files"]:
            self.upload_files(uploaded_files)

        # Render chat history
        self.render_chat_history()

        # User query input with unique key to prevent re-rendering
        query = st.text_input("Ask a question", key="user_query")

        # Process query only if it's new
        if query and query != st.session_state.get("last_query"):
            response = self.submit_query(query)
            
            if response:
                st.session_state["answer"] = response
                st.session_state["last_query"] = query  # Store the last processed query

        # Display the latest answer
        if st.session_state.get("answer"):
            st.subheader("Response")
            st.write(st.session_state["answer"])

        # End session button
        if st.button("End Session"):
            self.end_session()

def main():
    app = DocumentChatApp()
    app.run()

if __name__ == "__main__":
    main()