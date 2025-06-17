from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import psutil
import torch
import json
import os
import gc
import time
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
import io

app = FastAPI()

# Global variables
model = None
processor = None
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

# RAM Management Settings
RAM_CLEANUP_THRESHOLD = 75.0  # Cleanup when RAM usage exceeds this %
AUTO_CLEANUP_ENABLED = True
CLEANUP_AFTER_REQUESTS = True

def get_ram_usage():
    """Get current RAM usage percentage"""
    return psutil.virtual_memory().percent

def auto_cleanup_ram(force=False):
    """Automatic RAM cleanup with smart triggers"""
    try:
        ram_before = get_ram_usage()
        
        # Check if cleanup is needed
        if not force and ram_before < RAM_CLEANUP_THRESHOLD:
            return {"skipped": True, "ram_usage": ram_before}
        
        # Python garbage collection
        collected = gc.collect()
        
        # PyTorch cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        ram_after = get_ram_usage()
        freed = ram_before - ram_after
        
        if freed > 0:
            print(f"🧹 Auto cleanup: {freed:.1f}% RAM freed (was {ram_before:.1f}%, now {ram_after:.1f}%)")
        
        return {
            "cleaned": True,
            "objects_collected": collected,
            "ram_before": round(ram_before, 1),
            "ram_after": round(ram_after, 1),
            "ram_freed": round(freed, 1)
        }
        
    except Exception as e:
        print(f"⚠️ Auto cleanup failed: {e}")
        return {"error": str(e)}

def cleanup_decorator(func):
    """Decorator to automatically cleanup RAM after function execution"""
    def wrapper(*args, **kwargs):
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Auto cleanup if enabled
            if CLEANUP_AFTER_REQUESTS:
                cleanup_result = auto_cleanup_ram()
                if isinstance(result, dict) and "cleanup_info" not in result:
                    result["cleanup_info"] = cleanup_result
            
            return result
        except Exception as e:
            # Still try to cleanup even if function failed
            if CLEANUP_AFTER_REQUESTS:
                auto_cleanup_ram()
            raise e
    
    return wrapper

class TextRequest(BaseModel):
    text: str
    remember: bool = True

class ClearRequest(BaseModel):
    confirm: bool = True

@app.get("/")
def home():
    return {
        "message": "AI Engine with SmolVLM - Text & Vision!", 
        "memory": f"{psutil.virtual_memory().percent}%",
        "features": ["text_chat", "image_analysis", "vision_qa", "conversation_memory"],
        "model": "SmolVLM-256M-Instruct"
    }

@app.get("/health")
def health():
    model_status = "loaded" if model else "not loaded"
    return {
        "status": "healthy",
        "ram": f"{psutil.virtual_memory().percent}%",
        "model": model_status,
        "model_type": "SmolVLM (Vision + Language)",
        "conversation_length": len(conversation_history),
        "device": str(device)
    }

@app.post("/load_model")
def load_model():
    global model, processor
    try:
        print("🧠 Loading SmolVLM model...")
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        
        # Initialize processor
        print("📝 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Initialize model with device-specific settings
        print("🖼️ Loading vision-language model...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        }
        
        # Add attention implementation for CUDA
        if device.type == "cuda":
            model_kwargs["_attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["_attn_implementation"] = "eager"
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
        
        print("✅ SmolVLM loaded successfully!")
        return {
            "status": "SmolVLM loaded successfully!",
            "model": model_name,
            "device": str(device),
            "capabilities": ["text_generation", "image_understanding", "visual_qa"]
        }
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {"error": f"Failed to load model: {str(e)}"}

@app.post("/chat")
@cleanup_decorator
def chat(request: TextRequest):
    global conversation_history
    
    if model is None or processor is None:
        return {"error": "Model not loaded. Visit /load_model first"}

    try:
        # Create messages for text-only conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.text}
                ]
            }
        ]
        
        # Add conversation history if remembering (text only for now)
        if request.remember and conversation_history:
            # Add recent text conversations for context
            recent_text_messages = [msg for msg in conversation_history[-10:] if msg.get("type") == "text"]
            if recent_text_messages:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_text_messages[-5:]])
                messages[0]["content"][0]["text"] = f"Previous conversation:\n{context}\n\nCurrent question: {request.text}"

        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=None, return_tensors="pt")
        inputs = inputs.to(device)

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("assistant\n")[-1].strip()

        # Update conversation history if remembering
        save_success = False
        if request.remember:
            conversation_history.append({"role": "user", "content": request.text, "type": "text"})
            conversation_history.append({"role": "assistant", "content": response, "type": "text"})
            save_success = save_history(conversation_history)

        return {
            "response": response,
            "conversation_length": len(conversation_history),
            "remembered": request.remember,
            "saved_to_file": save_success,
            "model": "SmolVLM"
        }
        
    except Exception as e:
        return {"error": f"Chat failed: {str(e)}"}

@app.post("/analyze_image")
@cleanup_decorator
async def analyze_image(file: UploadFile = File(...), question: str = "Describe this image in detail.", remember: bool = True):
    global conversation_history
    
    if model is None or processor is None:
        return {"error": "Model not loaded. Visit /load_model first"}
    
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"🖼️ Analyzing image: {file.filename}")
        
        # Create messages with image and text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Prepare inputs with image
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode response
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("assistant\n")[-1].strip()
        
        # Save to history if remembering
        save_success = False
        if remember:
            conversation_history.append({
                "role": "user", 
                "content": f"[Image: {file.filename}] {question}",
                "type": "image",
                "filename": file.filename,
                "question": question
            })
            conversation_history.append({
                "role": "assistant", 
                "content": response,
                "type": "image_response"
            })
            save_success = save_history(conversation_history)
        
        return {
            "filename": file.filename,
            "question": question,
            "response": response,
            "image_size": f"{image.width}x{image.height}",
            "conversation_length": len(conversation_history),
            "remembered": remember,
            "saved_to_file": save_success,
            "model": "SmolVLM"
        }
        
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}

@app.post("/vision_chat")
@cleanup_decorator
async def vision_chat(file: UploadFile = File(...), text: str = "What do you see?", remember: bool = True):
    """Enhanced vision chat with conversation context"""
    global conversation_history
    
    if model is None or processor is None:
        return {"error": "Model not loaded. Visit /load_model first"}
    
    try:
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Build conversation context
        context_text = text
        if remember and conversation_history:
            # Add recent conversation context
            recent_messages = conversation_history[-6:]
            if recent_messages:
                context_parts = []
                for msg in recent_messages:
                    if msg.get("type") in ["text", "image_response"]:
                        context_parts.append(f"{msg['role']}: {msg['content']}")
                
                if context_parts:
                    context_text = f"Previous conversation:\n" + "\n".join(context_parts[-4:]) + f"\n\nNow looking at this image: {text}"
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": context_text}
                ]
            }
        ]
        
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("assistant\n")[-1].strip()
        
        # Save to history
        save_success = False
        if remember:
            conversation_history.append({
                "role": "user", 
                "content": f"[Image: {file.filename}] {text}",
                "type": "vision_chat",
                "filename": file.filename
            })
            conversation_history.append({
                "role": "assistant", 
                "content": response,
                "type": "vision_response"
            })
            save_success = save_history(conversation_history)
        
        return {
            "filename": file.filename,
            "user_message": text,
            "response": response,
            "conversation_length": len(conversation_history),
            "remembered": remember,
            "saved_to_file": save_success,
            "model": "SmolVLM"
        }
        
    except Exception as e:
        return {"error": f"Vision chat failed: {str(e)}"}

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

@app.post("/force_cleanup")
def force_cleanup():
    """Manually trigger RAM cleanup"""
    cleanup_result = auto_cleanup_ram(force=True)
    ram_usage = get_ram_usage()
    
    return {
        "status": "Cleanup completed",
        "current_ram_usage": f"{ram_usage:.1f}%",
        "cleanup_details": cleanup_result,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/ram_settings")
def get_ram_settings():
    """Get current RAM management settings"""
    return {
        "auto_cleanup_enabled": AUTO_CLEANUP_ENABLED,
        "cleanup_after_requests": CLEANUP_AFTER_REQUESTS,
        "cleanup_threshold": f"{RAM_CLEANUP_THRESHOLD}%",
        "current_ram_usage": f"{get_ram_usage():.1f}%"
    }

@app.post("/ram_settings")
def update_ram_settings(
    auto_cleanup: bool = None,
    cleanup_after_requests: bool = None, 
    threshold: float = None
):
    """Update RAM management settings"""
    global AUTO_CLEANUP_ENABLED, CLEANUP_AFTER_REQUESTS, RAM_CLEANUP_THRESHOLD
    
    if auto_cleanup is not None:
        AUTO_CLEANUP_ENABLED = auto_cleanup
    
    if cleanup_after_requests is not None:
        CLEANUP_AFTER_REQUESTS = cleanup_after_requests
    
    if threshold is not None and 50 <= threshold <= 95:
        RAM_CLEANUP_THRESHOLD = threshold
    
    return {
        "status": "Settings updated",
        "auto_cleanup_enabled": AUTO_CLEANUP_ENABLED,
        "cleanup_after_requests": CLEANUP_AFTER_REQUESTS,
        "cleanup_threshold": f"{RAM_CLEANUP_THRESHOLD}%"
    }

@app.get("/model_info")
def model_info():
    """Get detailed model information"""
    return {
        "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "model_type": "Vision-Language Model",
        "parameters": "256M",
        "capabilities": [
            "Text generation",
            "Image understanding", 
            "Visual question answering",
            "Image description",
            "Multi-modal conversation"
        ],
        "device": str(device),
        "memory_efficient": True,
        "loaded": model is not None,
        "auto_ram_cleanup": AUTO_CLEANUP_ENABLED
    }

if __name__ == "__main__":
    import uvicorn
    print("🧠 Starting SmolVLM with Auto RAM Cleanup...")
    print(f"⚙️ Auto cleanup: {'ON' if AUTO_CLEANUP_ENABLED else 'OFF'}")
    print(f"🎯 Cleanup threshold: {RAM_CLEANUP_THRESHOLD}%")
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=["./"]  # Watch current directory for changes
    )