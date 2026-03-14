def augmented_text(docs: dict, type: str, response: str):
    print("Search Results:")
    for doc in docs.values():
        print(f" - {doc['title']}")
    print(type)
    print(response)
