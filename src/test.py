from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    # Test connection to Ollama
    test_llm = ChatOllama(
        model="tinyllama:1.1b-chat-v1-q4_K_M",
        base_url="http://localhost:11434"
    )
    # Simple test query
    test_llm.invoke("test")
except Exception as e:
    print(e)

print("success")