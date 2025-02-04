import ollama

def generate_response(query, context, chat_history=None, history_limit=5):
    """
    Generate a response using LLM, incorporating chat history and relevant context from documents.
    
    :param query: User's question
    :param context: Retrieved document chunks (if any)
    :param chat_history: List of previous chat exchanges (optional)
    :param history_limit: Number of recent chat exchanges to keep
    :return: Generated response from LLM
    """

    # Format chat history (keep only last `history_limit` exchanges)
    formatted_history = []
    if chat_history:
        recent_chats = chat_history[-history_limit:]  # Keep only the most recent `history_limit` messages
        for chat in recent_chats:
            chat_entry = eval(chat)  # Convert stored JSON string back to Python dict
            formatted_history.append(f"Q: {chat_entry['question']}\nA: {chat_entry['answer']}")

    # Create prompt with context & chat history
    history_text = "\n".join(formatted_history) if formatted_history else "No prior conversation."
    prompt = (
        f"Previous conversation:\n{history_text}\n\n"
        f"Context:\n{context}\n\n"
        f"Q: {query}\nA:"
    )

    # Generate response using deepseek model
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "system", "content": prompt}])

    return response["message"]["content"]
