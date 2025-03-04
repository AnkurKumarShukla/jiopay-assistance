import streamlit as st
from backend.response_manager import response  # Import the response function

# Page configuration
st.set_page_config(page_title="JioPay Business Assistant", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 700px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        max-width: 75%;
        font-size: 15px;
    }
    .chat-message.assistant {
        background-color: #e9f5ff;
        border-left: 4px solid #0050a8;
        align-self: flex-start;
    }
    .chat-message.user {
        background-color: #fddede;
        border-left: 4px solid #d74714;
        align-self: flex-end;
        margin-left: auto;
    }
    .source-box {
        background-color: #f4f4f4;
        padding: 8px;
        border-radius: 6px;
        font-size: 14px;
        margin-top: 5px;
    }
    .source-title {
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom header
st.markdown("""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
    <div style="background-color: #0050a8; border-radius: 50%; width: 40px; height: 40px; display: flex; justify-content: center; align-items: center; color: white; font-weight: bold; font-size: 18px; margin-right: 10px;">
        Jio
    </div>
    <span style="font-size: 18px; font-weight: bold; color: #0050a8;">JioPay Business Assistant</span>
</div>
""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Query input section
question = st.chat_input("Ask something about JioPay...", key="chat_input")

# Handle user input
if question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = response(question)  # Call the backend function
            llm_response = result["response"]
            sources = result["sources"]

            # Display AI response
            st.write(llm_response)

            # Sort sources by score in descending order
            sorted_sources = sorted(sources, key=lambda x: x["score"], reverse=True)

            # Take top 3 sources to calculate the average score
            top_3_scores = [src["score"] for src in sorted_sources[:3]]
            avg_score = round(sum(top_3_scores) / len(top_3_scores), 2) if top_3_scores else 0

            # Remove duplicate sources (same URL and section)
            unique_sources = set((src["url"], src["section"]) for src in sources)

            # Display sources in a collapsible section
            if unique_sources:
                with st.expander("Sources (Click to view)"):
                    for url, section in unique_sources:
                        st.markdown(
                            f"""<div class="source-box">
                                <span class="source-title">Section:</span> {section}  
                                <br><span class="source-title">Score:</span> {avg_score}  
                                <br><span class="source-title">URL:</span> <a href="{url}" target="_blank">{url}</a>
                            </div>""",
                            unsafe_allow_html=True
                        )

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": llm_response})

# run uvicorn main:app --reload


