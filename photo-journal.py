import os
import json
import datetime
import requests
from pathlib import Path
import argparse
from openai import OpenAI
import subprocess
import platform

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

# Function to search for images using the Pexels API
def search_images(query, num_results=3):
    """Search for images using the Pexels API"""
    # Get API key from environment variables
    api_key = os.environ.get("PEXELS_API_KEY")
    
    # Demo images as fallback
    demo_images = [
        {
            "url": "https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=1200",
            "source": "Unsplash - Sadie Teper"
        },
        {
            "url": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e?w=1200",
            "source": "Unsplash - Robert Lukeman"
        },
        {
            "url": "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=1200",
            "source": "Unsplash - Joshua Newton"
        }
    ]
    
    # If no API key, use demo images
    if not api_key:
        print("No Pexels API key found. Using demo images.")
        import random
        return random.sample(demo_images, min(num_results, len(demo_images)))
    
    # Setup the API request
    headers = {
        "Authorization": api_key
    }
    
    # URL encode the query
    from urllib.parse import quote
    encoded_query = quote(query)
    
    # Make the API request
    url = f"https://api.pexels.com/v1/search?query={encoded_query}&per_page={num_results}"
    
    try:
        response = requests.get(url, headers=headers)
        
        # Check for successful response
        if response.status_code != 200:
            print(f"Error from Pexels API: {response.status_code} - {response.text}")
            return demo_images
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the image data
        images = []
        for photo in data.get("photos", [])[:num_results]:
            images.append({
                "url": photo.get("src", {}).get("large", ""),
                "source": f"Pexels - {photo.get('photographer', 'Unknown')}"
            })
        
        # If we got no results, use demo images
        if not images:
            print("No images found via Pexels API. Using demo images.")
            import random
            return random.sample(demo_images, min(num_results, len(demo_images)))
        
        return images
        
    except Exception as e:
        print(f"Error searching for images with Pexels API: {e}")
        return demo_images

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
# Function to download an image and save it locally
def download_image(url, journal_dir, base_filename):
    """Download an image from URL and save to local directory"""
    if url.startswith("https://placeholder.com"):
        # Don't try to download placeholder images
        return url
    
    try:
        # Create a filename based on the journal entry
        image_ext = url.split("?")[0].split(".")[-1]  # Extract extension
        if image_ext not in ["jpg", "jpeg", "png", "gif", "webp"]:
            image_ext = "jpg"  # Default to jpg if unknown extension
            
        image_filename = f"{base_filename}.{image_ext}"
        image_path = journal_dir / "images" / image_filename
        
        # Download the image
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            # Return relative path to the image
            return f"images/{image_filename}"
        else:
            print(f"Failed to download image, status code: {response.status_code}")
            return url
    except Exception as e:
        print(f"Error downloading image: {e}")
        return url

def generate_markdown(journal_entry, timestamp, tags, importance, image_info, local_image_path=None):
    """Generate a markdown file for the journal entry and selected image"""
    # Format the date for display
    date_obj = datetime.datetime.fromisoformat(timestamp)
    formatted_date = date_obj.strftime("%A, %B %d, %Y at %I:%M %p")
    
    # Format tags for display
    tags_str = ", ".join([f"#{tag}" for tag in tags]) if tags else ""
    
    # Determine image path - use local path if available
    image_path = local_image_path if local_image_path else image_info['url']
    
    # Create the markdown content
    markdown = f"""# Journal Entry: {formatted_date}

![Selected Image]({image_path})
*Image source: {image_info['source']}*

## Entry
{journal_entry}

**Importance:** {importance}/10
{f"**Tags:** {tags_str}" if tags_str else ""}

## Why This Image Was Selected
{image_info['selection_rationale']}

"""

    # Add thumbnails section if available
    if 'thumbnails' in image_info and any(image_info['thumbnails']):
        markdown += "\n## All Considered Images\n\n"
        for i, thumb in enumerate(image_info['thumbnails']):
            if thumb:
                markdown += f"![Image Option {i+1}]({thumb['path']}) *Source: {thumb['source']}*  \n"
    
    markdown += f"\n---\n*Search query used: \"{image_info['search_query']}\"*\n"
    
    return markdown

# Function to optionally save a thumbnail image for search results
def save_thumbnails(images, journal_dir, base_filename):
    """Save thumbnails of all search result images for reference"""
    thumbnails_dir = journal_dir / "images" / "thumbnails"
    thumbnails_dir.mkdir(exist_ok=True)
    
    thumbnails = []
    for i, img in enumerate(images):
        try:
            # Create a thumbnail filename
            thumb_filename = f"{base_filename}_thumb_{i+1}.jpg"
            thumb_path = thumbnails_dir / thumb_filename
            
            # Download the thumbnail
            response = requests.get(img["url"], stream=True)
            if response.status_code == 200:
                with open(thumb_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                # Add to list with relative path
                thumbnails.append({
                    "path": f"images/thumbnails/{thumb_filename}",
                    "source": img["source"]
                })
            else:
                thumbnails.append(None)
        except Exception as e:
            print(f"Error saving thumbnail: {e}")
            thumbnails.append(None)
    
    return thumbnails

# Function to open a file with the default system application
def open_file(file_path):
    """Open a file with the default system application"""
    try:
        file_path = Path(file_path).resolve()
        if platform.system() == 'Windows':
            os.startfile(str(file_path))
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(file_path)], check=True)
        else:  # Linux and other Unix-like
            subprocess.run(['xdg-open', str(file_path)], check=True)
        return True
    except Exception as e:
        print(f"Error opening file: {e}")
        return False

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
    
    # Optionally save thumbnails of all search results
    thumbnails = save_thumbnails(images, journal_dir, base_filename)
    
    # Select the best image
    image_selection = select_best_image(entry, images)
    selected_image_url = image_selection['selected_image']['url']
    print(f"Selected image: {selected_image_url}")
    
    # Download the image locally
    local_image_path = download_image(selected_image_url, journal_dir, base_filename)
    print(f"Image saved locally at: {local_image_path}")
    
    # Image information
    image_info = {
        "url": selected_image_url,
        "local_path": local_image_path,
        "source": image_selection['selected_image']['source'],
        "search_query": search_query,
        "selection_rationale": image_selection['selection_rationale'],
        "all_options": [{"url": img["url"], "source": img["source"]} for img in images],
        "thumbnails": thumbnails
    }
    
    # Create the journal entry with the selected image
    journal_entry_data = {
        "timestamp": timestamp,
        "entry": entry,
        "tags": tags or [],
        "importance": importance,
        "image": image_info
    }
    
    # Save the entry to a JSON file
    with open(journal_dir / json_filename, "w") as f:
        json.dump(journal_entry_data, f, indent=2)
    
    # Generate and save markdown file with image selection info
    markdown_content = generate_markdown(
        entry, 
        timestamp, 
        tags or [], 
        importance, 
        image_info, 
        local_image_path
    )
    
    # Save markdown to markdown_views directory
    markdown_path = journal_dir / "markdown_views" / md_filename
    with open(markdown_path, "w") as f:
        f.write(markdown_content)
    
    # Try to open the markdown file with the default viewer
    try:
        open_file(markdown_path)
    except Exception as e:
        print(f"Note: Could not automatically open the markdown file: {e}")
    
    return f"Journal entry saved to {json_filename} with associated images. Markdown view created at markdown_views/{md_filename}"

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

def generate_markdown_from_json(json_file):
    """Generate a markdown file from an existing journal entry JSON file"""
    try:
        # Get the journal directory
        journal_dir = setup_journal_directory()
        
        # Load the JSON file
        json_path = Path(json_file)
        if not json_path.is_absolute():
            json_path = journal_dir / json_path
            
        if not json_path.exists():
            return f"Error: JSON file {json_file} not found"
            
        with open(json_path, "r") as f:
            journal_data = json.load(f)
            
        # Extract data from the JSON
        entry = journal_data.get("entry", "")
        timestamp = journal_data.get("timestamp", datetime.datetime.now().isoformat())
        tags = journal_data.get("tags", [])
        importance = journal_data.get("importance", 5)
        image_info = journal_data.get("image", {})
        local_image_path = image_info.get("local_path", None)
        
        # Generate markdown content
        markdown_content = generate_markdown(
            entry,
            timestamp,
            tags,
            importance,
            image_info,
            local_image_path
        )
        
        # Create markdown filename based on the JSON filename
        md_filename = json_path.stem + ".md"
        markdown_path = journal_dir / "markdown_views" / md_filename
        
        # Save the markdown file
        with open(markdown_path, "w") as f:
            f.write(markdown_content)
            
        # Try to open the markdown file
        open_file(markdown_path)
        
        return f"Markdown file generated and saved to markdown_views/{md_filename}"
        
    except Exception as e:
        return f"Error generating markdown from JSON: {e}"

def list_journal_entries():
    """List all journal entries and allow the user to select one to view"""
    journal_dir = setup_journal_directory()
    
    # Find all JSON files in the journal directory
    json_files = list(journal_dir.glob("*.json"))
    
    if not json_files:
        print("No journal entries found.")
        return
    
    # Sort files by modification time (newest first)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("\nAvailable journal entries:")
    for i, file in enumerate(json_files, 1):
        # Try to extract date from filename
        try:
            # Load the JSON to get the entry preview
            with open(file, "r") as f:
                data = json.load(f)
            
            # Get timestamp and format it
            timestamp = data.get("timestamp", "")
            if timestamp:
                date_obj = datetime.datetime.fromisoformat(timestamp)
                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = "Unknown date"
            
            # Get a preview of the entry
            entry = data.get("entry", "")
            preview = entry[:50] + "..." if len(entry) > 50 else entry
            
            print(f"{i}. [{date_str}] {preview}")
        except Exception:
            # Fallback to just showing the filename
            print(f"{i}. {file.name}")
    
    # Ask user to select an entry
    try:
        choice = input("\nEnter the number of the entry to view (or 'q' to quit): ")
        if choice.lower() in ['q', 'quit', 'exit']:
            return
        
        index = int(choice) - 1
        if 0 <= index < len(json_files):
            selected_file = json_files[index]
            result = generate_markdown_from_json(selected_file)
            print(result)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Chat with GPT-4o and allow it to add journal entries with images")
    parser.add_argument("prompt", nargs="?", default=None, help="Initial prompt to send to the model")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive chat mode")
    parser.add_argument("--generate-markdown", metavar="JSON_FILE", help="Generate a markdown file from an existing journal entry JSON file")
    parser.add_argument("--list-entries", action="store_true", help="List all journal entries and view one as markdown")
    
    args = parser.parse_args()
    
    if args.list_entries:
        list_journal_entries()
    elif args.generate_markdown:
        result = generate_markdown_from_json(args.generate_markdown)
        print(result)
    elif args.interactive:
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
