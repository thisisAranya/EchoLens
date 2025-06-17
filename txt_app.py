from fastapi import FastAPI
from pydantic import BaseModel
import psutil
import torch
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# Global variables
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# JSON file for conversation history
HISTORY_FILE = "conversation_history.json"

def load_history():
    """Load conversation history from JSON file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('messages', [])
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    return []

def save_history(messages):
    """Save conversation history to JSON file"""
    try:
        data = {
            "last_updated": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages
        }
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving history: {e}")
        return False

# Load conversation history on startup
conversation_history = load_history()
print(f"Loaded {len(conversation_history)} messages from history")

class TextRequest(BaseModel):
    text: str
    remember: bool = True  # Option to use memory or not

class ClearRequest(BaseModel):
    confirm: bool = True

@app.get("/")
def home():
    return {"message": "AI Engine Running!", "memory": f"{psutil.virtual_memory().percent}%"}

@app.get("/health")
def health():
    model_status = "loaded" if model else "not loaded"
    return {
        "status": "healthy",
        "ram": f"{psutil.virtual_memory().percent}%",
        "model": model_status,
        "conversation_length": len(conversation_history)
    }

@app.post("/load_model")
def load_model():
    global model, tokenizer
    try:
        print("Loading AI model... (this takes 2-3 minutes first time)")
        model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        print("✅ Model loaded!")
        return {"status": "Model loaded successfully!"}
    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}

@app.post("/chat")
def chat(request: TextRequest):
    global conversation_history
    
    if model is None or tokenizer is None:
        return {"error": "Model not loaded. Visit /load_model first"}

    try:
        # Build messages based on whether to remember or not
        if request.remember and conversation_history:
            # Use conversation history
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": request.text})
        else:
            # Fresh conversation
            messages = [{"role": "user", "content": request.text}]

        # Limit conversation length to prevent memory issues
        if len(messages) > 20:  # Keep last 10 exchanges (20 messages)
            messages = messages[-20:]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Encode and move to device
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode only the newly generated part
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Update conversation history if remembering
        if request.remember:
            conversation_history.append({"role": "user", "content": request.text})
            conversation_history.append({"role": "assistant", "content": response.strip()})
            
            # Save to JSON file
            save_success = save_history(conversation_history)

        return {
            "response": response.strip(),
            "conversation_length": len(conversation_history),
            "remembered": request.remember,
            "saved_to_file": save_success if request.remember else False
        }
        
    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

@app.post("/clear_memory")
def clear_memory(request: ClearRequest):
    """Clear conversation history"""
    global conversation_history
    
    if request.confirm:
        conversation_history = []
        save_success = save_history(conversation_history)
        return {
            "status": "Memory cleared!", 
            "conversation_length": 0,
            "file_cleared": save_success
        }
    else:
        return {"error": "Set confirm=true to clear memory"}

@app.get("/conversation")
def get_conversation():
    """View current conversation history"""
    return {
        "conversation": conversation_history,
        "length": len(conversation_history),
        "file_exists": os.path.exists(HISTORY_FILE)
    }

@app.get("/export_history")
def export_history():
    """Export conversation history with metadata"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        except Exception as e:
            return {"error": f"Failed to read history file: {e}"}
    else:
        return {"error": "No history file found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)