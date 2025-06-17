from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import psutil
import os
import time
import tempfile

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

app = FastAPI()

# Global variables
whisper_model = None
tts_engine = None

# Audio settings
AUDIO_TEMP_DIR = "temp_audio"
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

# Create temp audio directory
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    voice_speed: int = 150  # Words per minute

@app.get("/")
def home():
    return {
        "message": "🎵 Simple Audio AI Engine!", 
        "memory": f"{psutil.virtual_memory().percent:.1f}%",
        "features": [
            "speech_to_text", 
            "text_to_speech",
            "voice_echo"
        ],
        "status": {
            "whisper": "Available" if WHISPER_AVAILABLE else "Not Available",
            "tts": "Available" if TTS_AVAILABLE else "Not Available"
        }
    }

@app.get("/health")
def health():
    whisper_status = "loaded" if whisper_model else "not loaded"
    tts_status = "available" if tts_engine else "not available"
    
    return {
        "status": "healthy",
        "ram": f"{psutil.virtual_memory().percent:.1f}%",
        "audio_models": {
            "speech_recognition": whisper_status,
            "text_to_speech": tts_status
        },
        "dependencies": {
            "whisper_available": WHISPER_AVAILABLE,
            "tts_available": TTS_AVAILABLE
        }
    }

@app.post("/load_audio_models")
def load_audio_models():
    global whisper_model, tts_engine
    try:
        print("🎵 Loading audio models...")
        
        # Load Whisper model
        if WHISPER_AVAILABLE:
            print("🎤 Loading Whisper model...")
            whisper_model = whisper.load_model("base")  # Options: "tiny", "base", "small", "medium", "large"
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
                tts_engine.setProperty('voice', voices[0].id)
            tts_engine.setProperty('rate', 150)
            tts_engine.setProperty('volume', 0.9)
            print("✅ TTS initialized!")
        else:
            print("⚠️ TTS not available. Install with: pip install pyttsx3")
        
        return {
            "status": "Audio models loaded successfully!",
            "speech_recognition": "Whisper-base" if WHISPER_AVAILABLE else "Not available",
            "text_to_speech": "pyttsx3" if TTS_AVAILABLE else "Not available"
        }
    except Exception as e:
        return {"error": f"Failed to load audio models: {str(e)}"}

@app.post("/speech_to_text")
async def speech_to_text(file: UploadFile = File(...)):
    """Convert speech to text using Whisper"""
    
    if not WHISPER_AVAILABLE:
        raise HTTPException(status_code=501, detail="Whisper not available. Install with: pip install openai-whisper")
    
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Whisper model not loaded. Visit /load_audio_models first")
    
    try:
        # Check file format
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_AUDIO_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported audio format. Supported: {SUPPORTED_AUDIO_FORMATS}")
        
        # Save uploaded file temporarily
        temp_file_path = os.path.join(AUDIO_TEMP_DIR, f"speech_{int(time.time())}{file_ext}")
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        print(f"🎤 Transcribing: {file.filename}")
        
        # Transcribe with Whisper
        result = whisper_model.transcribe(temp_file_path)
        transcribed_text = result["text"].strip()
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        return {
            "filename": file.filename,
            "transcribed_text": transcribed_text,
            "language": result.get("language", "unknown"),
            "confidence": "high" if len(transcribed_text) > 10 else "medium"
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return {"error": f"Speech transcription failed: {str(e)}"}

@app.post("/text_to_speech")
def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio file"""
    
    if not TTS_AVAILABLE:
        raise HTTPException(status_code=501, detail="TTS not available. Install with: pip install pyttsx3")
    
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized. Visit /load_audio_models first")
    
    try:
        # Configure speech rate
        tts_engine.setProperty('rate', request.voice_speed)
        
        # Generate unique filename
        timestamp = int(time.time())
        audio_filename = f"speech_{timestamp}.wav"
        audio_path = os.path.join(AUDIO_TEMP_DIR, audio_filename)
        
        print(f"🔊 Converting to speech: {request.text[:50]}...")
        
        # Generate speech
        tts_engine.save_to_file(request.text, audio_path)
        tts_engine.runAndWait()
        
        return {
            "text": request.text,
            "audio_filename": audio_filename,
            "voice_speed": request.voice_speed,
            "download_url": f"/download/{audio_filename}",
            "status": "success"
        }
        
    except Exception as e:
        return {"error": f"Text-to-speech failed: {str(e)}"}

@app.post("/voice_echo")
async def voice_echo(file: UploadFile = File(...), voice_speed: int = 150):
    """Voice echo: Speech → Text → Speech (audio processing demo)"""
    
    if not WHISPER_AVAILABLE or not TTS_AVAILABLE:
        missing = []
        if not WHISPER_AVAILABLE:
            missing.append("Whisper")
        if not TTS_AVAILABLE:
            missing.append("pyttsx3")
        
        raise HTTPException(
            status_code=501, 
            detail=f"Voice echo requires: {', '.join(missing)}"
        )
    
    try:
        print("🎙️ Starting voice echo...")
        
        # Step 1: Speech to Text
        speech_result = await speech_to_text(file)
        
        if "error" in speech_result:
            return speech_result
        
        transcribed_text = speech_result["transcribed_text"]
        
        # Step 2: Text to Speech (echo back)
        tts_request = TTSRequest(text=f"You said: {transcribed_text}", voice_speed=voice_speed)
        tts_result = text_to_speech(tts_request)
        
        if "error" in tts_result:
            return {
                "transcribed_text": transcribed_text,
                "tts_error": tts_result["error"]
            }
        
        return {
            "original_filename": file.filename,
            "transcribed_text": transcribed_text,
            "language": speech_result.get("language", "unknown"),
            "echo_text": f"You said: {transcribed_text}",
            "echo_audio_filename": tts_result["audio_filename"],
            "download_url": tts_result["download_url"]
        }
        
    except Exception as e:
        return {"error": f"Voice echo failed: {str(e)}"}

@app.get("/download/{filename}")
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

@app.get("/list_audio_files")
def list_audio_files():
    """List all generated audio files"""
    try:
        files = []
        for filename in os.listdir(AUDIO_TEMP_DIR):
            if filename.endswith('.wav'):
                file_path = os.path.join(AUDIO_TEMP_DIR, filename)
                file_size = os.path.getsize(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "download_url": f"/download/{filename}"
                })
        
        return {
            "audio_files": files,
            "count": len(files),
            "total_size_mb": round(sum(f["size_bytes"] for f in files) / (1024*1024), 2)
        }
    except Exception as e:
        return {"error": f"Failed to list files: {str(e)}"}

@app.delete("/cleanup_audio")
def cleanup_audio_files():
    """Clean up all temporary audio files"""
    try:
        deleted_files = []
        for filename in os.listdir(AUDIO_TEMP_DIR):
            if filename.endswith('.wav'):
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

@app.get("/audio_info")
def audio_info():
    """Get audio system information"""
    return {
        "audio_engine": "Simple Audio AI",
        "speech_recognition": {
            "engine": "OpenAI Whisper",
            "model": "base",
            "available": WHISPER_AVAILABLE,
            "loaded": whisper_model is not None
        },
        "text_to_speech": {
            "engine": "pyttsx3",
            "available": TTS_AVAILABLE,
            "initialized": tts_engine is not None
        },
        "supported_formats": SUPPORTED_AUDIO_FORMATS,
        "temp_directory": AUDIO_TEMP_DIR
    }

if __name__ == "__main__":
    import uvicorn
    print("🎵 Starting Simple Audio AI Engine...")
    print(f"🎤 Speech recognition: {'Available' if WHISPER_AVAILABLE else 'Install: pip install openai-whisper'}")
    print(f"🔊 Text-to-speech: {'Available' if TTS_AVAILABLE else 'Install: pip install pyttsx3'}")
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=["./"]  # Watch current directory for changes
    )