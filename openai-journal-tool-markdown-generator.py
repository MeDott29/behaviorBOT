import os
import json
import datetime
import requests
from pathlib import Path
import argparse
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define the journal tool
journal_tool = {
    "type": "function",
    "function": {
        "name": "add_journal_entry",
        "description": "Add a new entry to the personal memory journal",
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "The content of the journal entry"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags to categorize the entry"
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance level from 1-10, with 10 being most important",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["entry"]
        }
    }
}

# Set up the journal directory
def setup_journal_directory(journal_dir="journal"):
    """Create the journal directory if it doesn't exist"""
    path = Path(journal_dir)
    path.mkdir(exist_ok=True)
    
    # Also create images subdirectory
    images_path = path / "images"
    images_path.mkdir(exist_ok=True)
    
    # Create markdown views directory
    markdown_path = path / "markdown_views"
    markdown_path.mkdir(exist_ok=True)
    
    return path

# Function to generate an image search query using GPT-4o
def generate_image_search_query(journal_entry):
    """Generate an image search query based on the journal entry content"""
    messages = [
        {"role": "system", "content": "You are an assistant that helps create effective image search queries based on journal entries. Create a short, specific query that would find images that visually represent the journal entry's content and mood."},
        {"role": "user", "content": f"Create an image search query for this journal entry: {journal_entry}"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=50
    )
    
    return response.choices[0].message.content

# Function to search for images using the generated query
def search_images(query, num_results=3):
    """Search for images using a search API"""
    # This is a placeholder for an actual image search API
    # You would replace this with your preferred image search API
    
    # For example, if using Unsplash API:
    api_key = os.environ.get("UNSPLASH_API_KEY")
    if not api_key:
        # Return placeholder images if no API key
        return [
            {"url": "https://placeholder.com/800x600", "source": "placeholder"},
            {"url": "https://placeholder.com/800x600", "source": "placeholder"},
            {"url": "https://placeholder.com/800x600", "source": "placeholder"}
        ]
    
    headers = {
        "Authorization": f"Client-ID {api_key}"
    }
    
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={num_results}"
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        images = []
        for item in data.get("results", [])[:num_results]:
            images.append({
                "url": item.get("urls", {}).get("regular", ""),
                "source": f"Unsplash - {item.get('user', {}).get('name', 'Unknown')}"
            })
        
        # Fill with placeholders if we didn't get enough results
        while len(images) < num_results:
            images.append({
                "url": "https://placeholder.com/800x600",
                "source": "placeholder"
            })
            
        return images
    
    except Exception as e:
        print(f"Error searching for images: {e}")
        # Return placeholder images on error
        return [
            {"url": "https://placeholder.com/800x600", "source": "placeholder"},
            {"url": "https://placeholder.com/800x600", "source": "placeholder"},
            {"url": "https://placeholder.com/800x600", "source": "placeholder"}
        ]

# Function to select the best image using GPT-4o
def select_best_image(journal_entry, images):
    """Ask GPT-4o to select the best image for the journal entry"""
    # Prepare the message with images
    image_descriptions = []
    for i, img in enumerate(images, 1):
        image_descriptions.append(f"Image {i}: {img['url']} (Source: {img['source']})")
    
    image_list = "\n".join(image_descriptions)
    
    messages = [
        {"role": "system", "content": "You are an assistant that helps select the most appropriate image for a journal entry. Consider the mood, content, and themes of the entry when making your selection."},
        {"role": "user", "content": f"Journal entry: {journal_entry}\n\nPlease select the most appropriate image from these options:\n{image_list}\n\nExplain why this image best represents the journal entry."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Parse the response to determine which image was selected
    selected_image_response = response.choices[0].message.content
    
    # Return both the full response explaining the choice and try to determine which image was selected
    selected_image_number = None
    for i in range(1, len(images) + 1):
        if f"Image {i}" in selected_image_response.split(".")[:3]:  # Look in first few sentences
            selected_image_number = i - 1  # Convert to zero-indexed
            break
    
    if selected_image_number is None:
        # If we couldn't determine the selection, default to the first
        selected_image_number = 0
    
    return {
        "selected_image": images[selected_image_number],
        "selection_rationale": selected_image_response
    }

# Function to add a journal entry
def generate_markdown(journal_entry, timestamp, tags, importance, image_info):
    """Generate a markdown file for the journal entry and selected image"""
    # Format the date for display
    date_obj = datetime.datetime.fromisoformat(timestamp)
    formatted_date = date_obj.strftime("%A, %B %d, %Y at %I:%M %p")
    
    # Format tags for display
    tags_str = ", ".join([f"#{tag}" for tag in tags]) if tags else ""
    
    # Create the markdown content
    markdown = f"""# Journal Entry: {formatted_date}

![Selected Image]({image_info['url']})
*Image source: {image_info['source']}*

## Entry
{journal_entry}

**Importance:** {importance}/10
{f"**Tags:** {tags_str}" if tags_str else ""}

## Why This Image Was Selected
{image_info['selection_rationale']}

---
*Search query used: "{image_info['search_query']}"*
"""
    return markdown

def add_journal_entry(entry, tags=None, importance=5):
    """Add a new entry to the journal with an associated image"""
    journal_dir = setup_journal_directory()
    
    # Create a timestamp and filename
    timestamp = datetime.datetime.now().isoformat()
    base_filename = timestamp.replace(':', '-').split('.')[0]
    json_filename = f"{base_filename}.json"
    md_filename = f"{base_filename}.md"
    
    # Generate an image search query based on the entry
    search_query = generate_image_search_query(entry)
    print(f"Generated image search query: {search_query}")
    
    # Search for images
    images = search_images(search_query)
    print(f"Found {len(images)} images for the journal entry")
    
    # Select the best image
    image_selection = select_best_image(entry, images)
    print(f"Selected image: {image_selection['selected_image']['url']}")
    
    # Image information
    image_info = {
        "url": image_selection['selected_image']['url'],
        "source": image_selection['selected_image']['source'],
        "search_query": search_query,
        "selection_rationale": image_selection['selection_rationale']
    }
    
    # Create the journal entry with the selected image
    journal_entry = {
        "timestamp": timestamp,
        "entry": entry,
        "tags": tags or [],
        "importance": importance,
        "image": image_info
    }
    
    # Save the entry to a JSON file
    with open(journal_dir / json_filename, "w") as f:
        json.dump(journal_entry, f, indent=2)
    
    # Generate and save markdown file
    markdown_content = generate_markdown(entry, timestamp, tags or [], importance, image_info)
    with open(journal_dir / md_filename, "w") as f:
        f.write(markdown_content)
    
    return f"Journal entry saved to {json_filename} with an associated image. Markdown view available at {md_filename}"

# Function to handle the assistant's tool calls
def handle_tool_calls(tool_calls):
    """Process tool calls from the assistant"""
    results = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "add_journal_entry":
            entry = function_args.get("entry")
            tags = function_args.get("tags", [])
            importance = function_args.get("importance", 5)
            
            result = add_journal_entry(entry, tags, importance)
            results.append(result)
    
    return results

# Function to chat with the model
def chat_with_model(prompt, model="gpt-4o"):
    """Chat with the OpenAI model, allowing it to use the journal tool"""
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[journal_tool],
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    
    # Check if the model wants to use tools
    if assistant_message.tool_calls:
        # Handle the tool calls
        tool_results = handle_tool_calls(assistant_message.tool_calls)
        
        # Add the tool results to the conversation
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in assistant_message.tool_calls
            ]
        })
        
        for idx, tc in enumerate(assistant_message.tool_calls):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_results[idx]
            })
        
        # Get a final response from the model
        final_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        assistant_message = final_response.choices[0].message
    
    return assistant_message.content

def main():
    parser = argparse.ArgumentParser(description="Chat with GPT-4o and allow it to add journal entries with images")
    parser.add_argument("prompt", nargs="?", default=None, help="Initial prompt to send to the model")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive chat mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        print("Starting interactive chat (type 'exit' to quit):")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            response = chat_with_model(user_input, args.model)
            print(f"\nAssistant: {response}")
    elif args.prompt:
        response = chat_with_model(args.prompt, args.model)
        print(f"Assistant: {response}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
