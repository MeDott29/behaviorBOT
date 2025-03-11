import os
import json
import datetime
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
    return path

# Function to add a journal entry
def add_journal_entry(entry, tags=None, importance=5):
    """Add a new entry to the journal"""
    journal_dir = setup_journal_directory()
    
    # Create a timestamp and filename
    timestamp = datetime.datetime.now().isoformat()
    filename = f"{timestamp.replace(':', '-').split('.')[0]}.json"
    
    # Create the journal entry
    journal_entry = {
        "timestamp": timestamp,
        "entry": entry,
        "tags": tags or [],
        "importance": importance
    }
    
    # Save the entry to a file
    with open(journal_dir / filename, "w") as f:
        json.dump(journal_entry, f, indent=2)
    
    return f"Journal entry saved to {filename}"

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
    parser = argparse.ArgumentParser(description="Chat with GPT-4o and allow it to add journal entries")
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
