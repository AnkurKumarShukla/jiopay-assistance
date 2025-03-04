import json
from backend.llm_handler import query_llm
from backend.context_retrival import hybrid_search

def response(query):
    """
    Retrieves relevant context using hybrid search, generates a response using an LLM,
    and returns sources with URL, section, and relevance score.
    """
    # Retrieve relevant context
    results = hybrid_search(query, top_k=5)

    # Extract relevant texts, metadata, and scores
    extracted_contexts = []
    sources = []

    for idx, res in enumerate(results):
        if "metadata" in res:
            source_text = res["metadata"].get("source_text", "No source text available")
            category = res["metadata"].get("category", "Unknown Category")
            score = res.get("score", 0)  # Default to 0 if score is missing

            # Decode 'origin' field
            origin_str = res["metadata"].get("origin", "{}")  # Default to empty dict if missing
            try:
                origin = json.loads(origin_str)  # Convert string back to dictionary
            except json.JSONDecodeError:
                origin = {}  # If decoding fails, default to empty dictionary

            # Extract URL and Section from origin
            url = origin.get("url", "Unknown URL")
            section = origin.get("section", "Unknown Section")

            extracted_contexts.append(f"Context {idx + 1}: {source_text}")
            sources.append({
                "category": category,
                "url": url,
                "section": section,
                "score": score
            })

    # Construct prompt for LLM
    context_str = "\n\n".join(extracted_contexts)
    prompt = f"""give response to the point . No Salutation and greeting . i context is not sufficient then tell user "insufficient context ! kindly contact admin - admin@jio.com" else  Use the following retrieved context to answer the query:

    {context_str}

    Query: {query}
    Answer:"""

    # Generate response from LLM
    llm_response = query_llm(prompt)

    return {
        "response": llm_response,
        "sources": sources
    }



# print(response("How long would it require to become P2M merchant after upgradation request?"))