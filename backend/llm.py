import ollama

def generate_response(query, context):
    prompt = f"Using the following context, answer the question:\n{context}\n\nQ: {query}\nA:"
    response = ollama.chat(model="deepseek-r1:1.5b", messages=[{"role": "system", "content": prompt}])
    return response["message"]["content"]
