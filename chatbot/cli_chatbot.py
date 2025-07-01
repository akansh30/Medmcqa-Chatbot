from chatbot.flow import build_chatbot_graph

def run_chatbot():
    graph = build_chatbot_graph()
    while True:
        query = input("\n Ask a medical question (type 'exit' to close the session): ")
        if query.lower() == "exit":
            break
        result = graph.invoke({"query": query})
        print(f"\n {result['response']}")
