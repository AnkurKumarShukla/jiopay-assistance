from playwright.sync_api import sync_playwright
import json
import time

# Function to capture network requests for JSON data
def capture_api_calls(page):
    api_data = []

    def log_response(response):
        # Check for JSON responses
        if "json" in response.headers.get("content-type", ""):
            try:
                json_data = response.json()
                print(f"Captured API Response: {json_data}")
                api_data.append(json_data)
            except Exception as e:
                print(f"Error parsing JSON: {e}")

    # Listen to all network responses
    page.on("response", log_response)
    return api_data

# Infinite scrolling to load dynamic content
def infinite_scroll(page, scroll_pause_time=2):
    previous_height = None
    while True:
        # Scroll down to the bottom of the page
        page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        # Get current scroll height
        current_height = page.evaluate("document.body.scrollHeight")
        if previous_height == current_height:
            # Stop if no more content is loaded
            break
        previous_height = current_height
    print("Finished scrolling.")

# Main Playwright logic
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, slow_mo=200)
    page = browser.new_page()
    page.goto("https://jiopay.com/business/help-center")
    page.wait_for_load_state("networkidle")

    # Capture API calls to get JSON data
    api_data = capture_api_calls(page)

    # Perform infinite scrolling to load all dynamic content
    infinite_scroll(page)

    # Extract all visible text and HTML after scrolling
    html = page.content()
    text_content = page.locator("body").inner_text()

    # Save data to JSON file
    collected_data = {
        "url": page.url,
        "text_content": text_content,
        "html": html,
        "api_data": api_data
    }

    with open("helpcentre.json", "w", encoding="utf-8") as f:
        json.dump(collected_data, f, indent=4, ensure_ascii=False)

    browser.close()

print("Data collection complete. Check react_scraped_data.json.")
