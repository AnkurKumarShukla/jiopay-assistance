import os
import json
import boto3
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
import urllib3
from http.client import RemoteDisconnected
import time

# Load environment variables
load_dotenv(os.path.join("backend", ".env"))

# === AWS Bedrock Setup ===
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"  # Change if needed

if AWS_ACCESS_KEY and AWS_SECRET_KEY:
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

# === OpenAI Setup ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Google Gemini Setup ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# === Mistral Setup ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def query_bedrock_llm(prompt):
    """Queries Claude 3 via AWS Bedrock"""
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(request_payload),
            contentType="application/json",
            accept="application/json"
        )
        model_response = json.loads(response["body"].read())
        return model_response["content"][0]["text"]
    except (ClientError, Exception) as e:
        print(f"❌ AWS Bedrock Error: {e}")
        return None  # Fallback to OpenAI


def query_openai_llm(prompt):
    """Queries OpenAI GPT-4 / GPT-3.5"""
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return None  # Fallback to Gemini


def query_gemini_llm(prompt):
    """Queries Google Gemini"""
    try:
        llm =ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None  # Fallback to Mistral


def query_mistral_llm(prompt):
    """Queries Mistral AI"""
    try:
        llm = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"❌ Mistral Error: {e}")
        return "Error: No LLM available"


import urllib3
from http.client import RemoteDisconnected
def query_llm(prompt, max_retries=3, delay=2):
    """Decides which LLM to use in sequence: Bedrock → OpenAI → Gemini → Mistral.
    If one fails, the next available LLM is tried in order.
    If any API fails due to network issues, it retries up to `max_retries` times.
    """
    llm_sequence = []

    if AWS_ACCESS_KEY and AWS_SECRET_KEY:
        llm_sequence.append(("AWS Bedrock (Claude 3)", query_bedrock_llm))
    if OPENAI_API_KEY:
        llm_sequence.append(("OpenAI (GPT-4)", query_openai_llm))
    if GEMINI_API_KEY:
        llm_sequence.append(("Google Gemini", query_gemini_llm))
    if MISTRAL_API_KEY:
        llm_sequence.append(("Mistral AI", query_mistral_llm))

    if not llm_sequence:
        return "Error: No valid LLM API keys provided."

    for name, llm_function in llm_sequence:
        for attempt in range(1, max_retries + 1):
            print(f"⚡ Trying {name} (Attempt {attempt})")
            try:
                # Force a fresh connection
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = llm_function(prompt)
                
                if response:  # Ensure valid response
                    print(f"✅ Success with {name}")
                    return response

            except (RemoteDisconnected, urllib3.exceptions.ProtocolError) as net_err:
                print(f"❌ {name} failed due to network error: {net_err}. Retrying in {delay} sec...")
                time.sleep(delay)  # Wait before retrying

            except Exception as e:
                print(f"❌ {name} failed: {str(e)}. Trying next LLM...")
                break  # Don't retry if it's a different issue (e.g., API key issue)

    return "Error: All LLMs failed to generate a response."




# ✅ Test the Unified LLM Handler
# if __name__ == "__main__":
#     user_prompt = "What is the capital of india ?"
#     print(query_llm(user_prompt))
