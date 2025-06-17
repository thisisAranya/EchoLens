from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psutil
import torch
import json
import os
import gc
import time
import tempfile
from datetime import datetime
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
import io
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Audio processing imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

app = FastAPI()

# Global variables
model = None
processor = None
whisper_model = None
tts_engine = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# JSON file for conversation history
HISTORY_FILE = "conversation_history.json"

# Audio settings
AUDIO_TEMP_DIR = "temp_audio"
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

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

# Create temp audio directory
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

# Load conversation history on startup
conversation_history = load_history()
print(f"Loaded {len(conversation_history)} messages from history")

# RAM Management Settings
RAM_CLEANUP_THRESHOLD = 75.0
AUTO_CLEANUP_ENABLED = True
CLEANUP_AFTER_REQUESTS = True

def get_ram_usage():
    """Get current RAM usage percentage"""
    return psutil.virtual_memory().percent

def auto_cleanup_ram(force=False):
    """Automatic RAM cleanup with smart triggers"""
    try:
        ram_before = get_ram_usage()
        
        if not force and ram_before < RAM_CLEANUP_THRESHOLD:
            return {"skipped": True, "ram_usage": ram_before}
        
        collected = gc.collect()
        
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
            result = func(*args, **kwargs)
            
            if CLEANUP_AFTER_REQUESTS:
                cleanup_result = auto_cleanup_ram()
                if isinstance(result, dict) and "cleanup_info" not in result:
                    result["cleanup_info"] = cleanup_result
            
            return result
        except Exception as e:
            if CLEANUP_AFTER_REQUESTS:
                auto_cleanup_ram()
            raise e
    
    return wrapper

class TextRequest(BaseModel):
    text: str
    remember: bool = True

class TTSRequest(BaseModel):
    text: str
    voice_speed: int = 150  # Words per minute
    save_file: bool = False

class ClearRequest(BaseModel):
    confirm: bool = True

@app.get("/")
def home():
    return {
        "message": "🎵 Complete AI Engine - Text, Vision & Audio!", 
        "memory": f"{get_ram_usage():.1f}%",
        "features": [
            "text_chat", 
            "image_analysis", 
            "speech_to_text", 
            "text_to_speech",
            "voice_conversations",
            "conversation_memory"
        ],
        "models": {
            "vision_language": "SmolVLM-256M-Instruct",
            "speech_recognition": "Whisper" if WHISPER_AVAILABLE else "Not Available",
            "text_to_speech": "pyttsx3" if TTS_AVAILABLE else "Not Available"
        }
    }

@app.get("/health")
def health():
    model_status = "loaded" if model else "not loaded"
    whisper_status = "loaded" if whisper_model else "not loaded"
    tts_status = "available" if tts_engine else "not available"
    ram_usage = get_ram_usage()
    
    cleanup_triggered = False
    if AUTO_CLEANUP_ENABLED and ram_usage > RAM_CLEANUP_THRESHOLD:
        auto_cleanup_ram()
        cleanup_triggered = True
        ram_usage = get_ram_usage()
    
    return {
        "status": "healthy",
        "ram": f"{ram_usage:.1f}%",
        "ram_status": "high" if ram_usage > 80 else "normal" if ram_usage > 60 else "low",
        "auto_cleanup_triggered": cleanup_triggered,
        "models": {
            "vision_language": model_status,
            "speech_recognition": whisper_status,
            "text_to_speech": tts_status
        },
        "conversation_length": len(conversation_history),
        "device": str(device),
        "audio_features": {
            "whisper_available": WHISPER_AVAILABLE,
            "tts_available": TTS_AVAILABLE,
            "audio_processing": PYDUB_AVAILABLE
        }
    }

@app.post("/load_models")
def load_models():
    global model, processor, whisper_model, tts_engine
    try:
        print("🧠 Loading all AI models...")
        
        # Load vision-language model
        print("🖼️ Loading SmolVLM...")
        model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"
        processor = AutoProcessor.from_pretrained(model_name)
        
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        }
        
        if device.type == "cuda":
            model_kwargs["_attn_implementation"] = "flash_attention_2"
        else:
            model_kwargs["_attn_implementation"] = "eager"
        
        model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs).to(device)
        
        # Load Whisper model
        if WHISPER_AVAILABLE:
            print("🎤 Loading Whisper model...")
            whisper_model = whisper.load_model("base")  # You can use "tiny", "base", "small", "medium", "large"
            print("✅ Whisper loaded!")
        else:
            print("⚠️ Whisper not available. Install with: pip install openai-whisper")
        
        # Initialize TTS engine
        if TTS_AVAILABLE:
            print("🔊 Initializing Text-to-Speech...")
            tts_engine = pyttsx3.init()
            
            # Configure TTS settings
            voices = tts_engine.getProperty('voices')
            if voices:
                tts_engine.setProperty('voice', voices[0].id)  # Use first available voice
            tts_engine.setProperty('rate', 150)  # Speaking rate
            tts_engine.setProperty('volume', 0.9)  # Volume level
            print("✅ TTS initialized!")
        else:
            print("⚠️ TTS not available. Install with: pip install pyttsx3")
        
        print("✅ All models loaded successfully!")
        return {
            "status": "All models loaded successfully!",
            "vision_language_model": model_name,
            "speech_recognition": "Whisper-base" if WHISPER_AVAILABLE else "Not available",
            "text_to_speech": "pyttsx3" if TTS_AVAILABLE else "Not available",
            "device": str(device)
        }
    except Exception as e:
        return {"error": f"Failed to load models: {str(e)}"}

@app.post("/chat")
@cleanup_decorator
def chat(request: TextRequest):
    global conversation_history
    
    if model is None or processor is None:
        return {"error": "Model not loaded. Visit /load_models first"}

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.text}
                ]
            }
        ]
        
        if request.remember and conversation_history:
            recent_text_messages = [msg for msg in conversation_history[-10:] if msg.get("type") == "text"]
            if recent_text_messages:
                context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_text_messages[-5:]])
                messages[0]["content"][0]["text"] = f"Previous conversation:\n{context}\n\nCurrent question: {request.text}"

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=None, return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("assistant\n")[-1].strip()

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

@app.post("/transcribe_audio")
@cleanup_decorator
async def transcribe_audio(file: UploadFile = File(...), auto_chat: bool = False, remember: bool = True):
    """Transcribe audio to text, optionally auto-chat with AI"""
    global conversation_history
    
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=501, detail="Whisper not available. Install with: pip install openai-whisper")
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded. Visit /load_models first")
    
    try:
        # Check file format
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format. Supported: {SUPPORTED_AUDIO_FORMATS}")
        
        # Save uploaded file temporarily
        temp_file_path = os.path.join(AUDIO_TEMP_DIR, f"temp_audio_{int(time.time())}{file_ext}")
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        print(f"🎤 Transcribing audio: {file.filename}")
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_file_path)
        transcribed_text = result["text"].strip()
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        response_data = {
            "filename": file.filename,
            "transcribed_text": transcribed_text,
            "language": result.get("language", "unknown"),
            "duration": len(content) / (16000 * 2),  # Approximate duration
        }
        
        # Auto-chat if requested
        if auto_chat and model is not None:
            print("🤖 Auto-chatting with transcribed text...")
            
            # Create chat request
            chat_request = TextRequest(text=transcribed_text, remember=remember)
            chat_response = chat(chat_request)
            
            response_data["auto_chat"] = True
            response_data["ai_response"] = chat_response.get("response", "")
            response_data["conversation_length"] = len(conversation_history)
        else:
            # Just save transcription to history if remembering
            if remember:
                conversation_history.append({
                    "role": "user", 
                    "content": f"[Audio transcription: {file.filename}] {transcribed_text}",
                    "type": "audio_transcription",
                    "filename": file.filename
                })
                save_history(conversation_history)
        
        return response_data
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {"error": f"Audio transcription failed: {str(e)}"}

@app.post("/text_to_speech")
@cleanup_decorator
def text_to_speech(request: TTSRequest):
    """Convert text to speech audio file"""
    
    if not TTS_AVAILABLE:
        raise HTTPException(status_code=501, detail="TTS not available. Install with: pip install pyttsx3")
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized. Visit /load_models first")
    
    try:
        # Configure TTS settings
        tts_engine.setProperty('rate', request.voice_speed)
        
        # Generate unique filename
        timestamp = int(time.time())
        audio_filename = f"tts_output_{timestamp}.wav"
        audio_path = os.path.join(AUDIO_TEMP_DIR, audio_filename)
        
        print(f"🔊 Converting text to speech: {request.text[:50]}...")
        
        # Save to file
        tts_engine.save_to_file(request.text, audio_path)
        tts_engine.runAndWait()
        
        response_data = {
            "text": request.text,
            "audio_filename": audio_filename,
            "voice_speed": request.voice_speed,
            "file_path": audio_path,
            "status": "success"
        }
        
        if request.save_file:
            response_data["download_url"] = f"/download_audio/{audio_filename}"
        
        return response_data
        
    except Exception as e:
        return {"error": f"Text-to-speech failed: {str(e)}"}

@app.get("/download_audio/{filename}")
def download_audio(filename: str):
    """Download generated audio file"""
    file_path = os.path.join(AUDIO_TEMP_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.post("/voice_conversation")
@cleanup_decorator
async def voice_conversation(file: UploadFile = File(...), respond_with_voice: bool = True, remember: bool = True):
    """Complete voice conversation: Speech-to-Text → AI Chat → Text-to-Speech"""
    
    if not WHISPER_AVAILABLE or not TTS_AVAILABLE:
        missing = []
        if not WHISPER_AVAILABLE:
            missing.append("Whisper (speech recognition)")
        if not TTS_AVAILABLE:
            missing.append("pyttsx3 (text-to-speech)")
        
        raise HTTPException(
            status_code=501, 
            detail=f"Voice conversation requires: {', '.join(missing)}"
        )
    
    try:
        print("🎙️ Starting voice conversation...")
        
        # Step 1: Transcribe audio
        transcribe_result = await transcribe_audio(file, auto_chat=True, remember=remember)
        
        if "error" in transcribe_result:
            return transcribe_result
        
        user_text = transcribe_result["transcribed_text"]
        ai_response = transcribe_result.get("ai_response", "")
        
        response_data = {
            "user_speech": user_text,
            "ai_response": ai_response,
            "transcription_language": transcribe_result.get("language", "unknown")
        }
        
        # Step 2: Convert AI response to speech
        if respond_with_voice and ai_response:
            tts_request = TTSRequest(text=ai_response, save_file=True)
            tts_result = text_to_speech(tts_request)
            
            if "error" not in tts_result:
                response_data["ai_voice_file"] = tts_result["audio_filename"]
                response_data["download_url"] = tts_result.get("download_url")
            else:
                response_data["tts_error"] = tts_result["error"]
        
        response_data["conversation_length"] = len(conversation_history)
        return response_data
        
    except Exception as e:
        return {"error": f"Voice conversation failed: {str(e)}"}

@app.post("/analyze_image")
@cleanup_decorator
async def analyze_image(file: UploadFile = File(...), question: str = "Describe this image in detail.", remember: bool = True):
    global conversation_history
    
    if model is None or processor is None:
        return {"error": "Model not loaded. Visit /load_models first"}
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        print(f"🖼️ Analyzing image: {file.filename}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        response = generated_texts[0].split("assistant\n")[-1].strip()
        
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

@app.delete("/cleanup_audio_files")
def cleanup_audio_files():
    """Clean up temporary audio files"""
    try:
        deleted_files = []
        for filename in os.listdir(AUDIO_TEMP_DIR):
            if filename.startswith(("temp_audio_", "tts_output_")):
                file_path = os.path.join(AUDIO_TEMP_DIR, filename)
                os.remove(file_path)
                deleted_files.append(filename)
        
        return {
            "status": "Audio cleanup completed",
            "deleted_files": deleted_files,
            "count": len(deleted_files)
        }
    except Exception as e:
        return {"error": f"Audio cleanup failed: {str(e)}"}


# Add this after your existing endpoints, before if __name__ == "__main__":
@app.get("/web")
def web_interface():
    """Serve the web interface"""
    return FileResponse("ai_web_interface.html")

# Optional: Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    import uvicorn
    print("🎵 Starting Complete AI Engine with Audio Processing...")
    print(f"⚙️ Auto cleanup: {'ON' if AUTO_CLEANUP_ENABLED else 'OFF'}")
    print(f"🎯 Cleanup threshold: {RAM_CLEANUP_THRESHOLD}%")
    print(f"🎤 Speech recognition: {'Available' if WHISPER_AVAILABLE else 'Not available'}")
    print(f"🔊 Text-to-speech: {'Available' if TTS_AVAILABLE else 'Not available'}")

    uvicorn.run(
        "img_txt_audio_app:app",  # Use import string instead of app
        host="127.0.0.1", 
        port=8000,
        reload=True,
        reload_dirs=["./"]
    )