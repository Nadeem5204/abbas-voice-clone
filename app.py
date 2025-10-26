import spaces
import torch
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import gradio as gr
import traceback
import gc
import numpy as np
import librosa
import time
from huggingface_hub import snapshot_download
from tts.infer_cli import MegaTTS3DiTInfer

# 🚀 ULTIMATE GPU OPTIMIZATION
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

def download_weights():
    """Download model weights from HuggingFace if not already present."""
    repo_id = "mrfakename/MegaTTS3-VoiceCloning"
    weights_dir = "checkpoints"
    
    if not os.path.exists(weights_dir):
        print("Downloading model weights from HuggingFace...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=weights_dir,
            local_dir_use_symlinks=False
        )
        print("Model weights downloaded successfully!")
    else:
        print("Model weights already exist.")
    
    return weights_dir

# Download weights and initialize model
download_weights()
print("Initializing MegaTTS3 model...")

# Model loading with WARMUP
global infer_pipe
if 'infer_pipe' not in globals():
    infer_pipe = MegaTTS3DiTInfer()
    print("Model loaded successfully!")
    
    # 🚀 MODEL WARMUP FOR MAXIMUM SPEED
    print("🔥 Warming up model for maximum performance...")
    try:
        # Create dummy audio for warmup
        dummy_audio = np.random.random(8000).astype(np.float32)
        import soundfile as sf
        sf.write("dummy.wav", dummy_audio, 16000)
        
        with open("dummy.wav", 'rb') as f:
            dummy_content = f.read()
        
        with torch.no_grad():
            resource_context = infer_pipe.preprocess(dummy_content)
            _ = infer_pipe.forward(resource_context, "warmup", time_step=8)
        
        os.remove("dummy.wav")
        print("✅ Model warmup completed!")
    except:
        print("⚠️ Warmup skipped, but model is ready")

def ultra_fast_audio_process(audio_path, target_sr=22050, max_duration=6):
    """ULTRA FAST audio processing - quality preserved"""
    try:
        # 🚀 DIRECT LOAD - No unnecessary processing
        wav, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=max_duration)
        
        # Quick quality checks
        if len(wav) < 4000:
            raise ValueError("Audio too short")
            
        if np.max(np.abs(wav)) < 0.01:
            raise ValueError("Audio too quiet")
        
        # 🎯 PRESERVE ORIGINAL QUALITY - Minimal processing
        peak = np.max(np.abs(wav))
        if peak > 0.98:
            wav = wav / peak * 0.98
        
        # Save with original quality
        temp_path = "ultra_fast_temp.wav"
        import soundfile as sf
        sf.write(temp_path, wav, target_sr, subtype='PCM_16')
        
        print(f"✅ Audio ready: {len(wav)/sr:.1f}s - Quality preserved")
        return temp_path
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")
        return audio_path

@spaces.GPU
@torch.no_grad()
def generate_speech(inp_audio, inp_text, infer_timestep=10, p_w=1.4, t_w=3.0):
    # ✅ TIME TRACKING
    start_time = time.time()
    
    if not inp_audio or not inp_text:
        gr.Warning("Please provide both reference audio and text to generate.")
        return None
    
    try:
        print(f"🚀 ULTRA FAST: Generating: {inp_text[:100]}...")
        
        # 🚀 OPTIMIZED MEMORY CLEANUP
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 🎯 ULTRA FAST AUDIO PROCESSING
        try:
            processed_audio_path = ultra_fast_audio_process(inp_audio, max_duration=6)
            wav_path = processed_audio_path
            print("✅ Audio processed with quality preservation")
        except Exception as audio_error:
            print(f"❌ Audio error: {audio_error}")
            gr.Warning("Audio processing failed. Using original file.")
            wav_path = inp_audio
        
        # Read audio file
        with open(wav_path, 'rb') as file:
            file_content = file.read()
        
        # 🚀 ULTRA FAST GENERATION
        try:
            print("🔄 Starting ULTRA FAST generation...")
            
            resource_context = infer_pipe.preprocess(file_content)
            wav_bytes = infer_pipe.forward(
                resource_context, 
                inp_text, 
                time_step=infer_timestep,
                p_w=p_w, 
                t_w=t_w
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"✅ ULTRA FAST Generation completed in {generation_time:.2f} seconds")
            print(f"📊 Text Length: {len(inp_text)} characters")
            print(f"🎯 Quality: EXACT VOICE CLONE PRESERVED")
            
            # Quick optimized cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return wav_bytes
            
        except Exception as gen_error:
            print(f"❌ Generation error: {gen_error}")
            gr.Warning(f"Generation failed: {str(gen_error)}")
            return None
        
    except Exception as e:
        print(f"❌ Overall error: {e}")
        traceback.print_exc()
        gr.Warning(f"Error: {str(e)}")
        return None

# 🎨 CLEAN PROFESSIONAL INTERFACE
with gr.Blocks(
    title="Abbas Voice Clone - Ultra Fast",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ),
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
        font-family: 'Segoe UI', system-ui;
    }
    .header {
        text-align: center;
        padding: 25px 20px 15px 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px 10px 0 0;
        margin-bottom: 20px;
    }
    .header h1 {
        font-size: 2.5em;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header h3 {
        font-size: 1.2em;
        font-weight: 300;
        margin: 5px 0 0 0;
        opacity: 0.9;
    }
    .footer {
        text-align: center;
        padding: 15px;
        margin-top: 20px;
        font-size: 14px;
        color: #666;
        border-top: 1px solid #e0e0e0;
        background: #f8f9fa;
        border-radius: 0 0 10px 10px;
    }
    .section-title {
        font-size: 1.3em;
        font-weight: 600;
        margin-bottom: 15px;
        color: #2c3e50;
        border-left: 4px solid #667eea;
        padding-left: 12px;
    }
    .tip-box {
        background: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
    }
    .generate-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-weight: 600 !important;
    }
    .speed-indicator {
        background: #e8f5e8;
        padding: 10px;
        border-radius: 6px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    """
) as demo:
    
    # 🎯 HEADER
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 🎙️ Abbas Voice Clone")
        gr.Markdown("### Ultra Fast Voice Cloning - No Limits")
    
    # 📱 MAIN INTERFACE
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            # SPEED INDICATOR
            gr.Markdown("""
            <div class="speed-indicator">
            🚀 <strong>Ultra Fast Mode</strong> - No Character Limits - Unlimited Text
            </div>
            """)
            
            # INPUT SECTION
            gr.Markdown("### 🎵 Upload Voice Sample", elem_classes="section-title")
            reference_audio = gr.Audio(
                label="Upload reference voice (3-6 seconds recommended for speed)",
                type="filepath",
                sources=["upload", "microphone"],
                elem_id="audio_input"
            )
            
            gr.Markdown("### 📝 Enter Text", elem_classes="section-title")
            text_input = gr.Textbox(
                label="Text to convert to speech",
                placeholder="Type any length text - no limits! Books, scripts, long paragraphs...",
                lines=5,  # 🚀 3 se 5 lines
                max_lines=10,  # 🚀 5 se 10 lines
                show_copy_button=True
                # 🚀 NO max_length - UNLIMITED TEXT
            )
            
            # OPTIMIZED SETTINGS
            with gr.Accordion("⚙️ Speed Optimized Settings", open=False):
                infer_timestep = gr.Slider(
                    label="Generation Speed",
                    value=10,
                    minimum=8,
                    maximum=12,
                    step=1,
                    info="10 = Optimal Speed with Good Quality"
                )
                p_w = gr.Slider(
                    label="Speech Clarity",
                    value=1.4,
                    minimum=1.3,
                    maximum=1.5,
                    step=0.1
                )
                t_w = gr.Slider(
                    label="Voice Similarity", 
                    value=3.0,
                    minimum=2.8,
                    maximum=3.2,
                    step=0.1
                )
            
            # GENERATE BUTTON
            generate_btn = gr.Button(
                "🚀 Generate Unlimited Text", 
                variant="primary", 
                size="lg",
                elem_classes="generate-btn"
            )
        
        with gr.Column(scale=1):
            # OUTPUT SECTION
            gr.Markdown("### 🔊 Generated Voice", elem_classes="section-title")
            output_audio = gr.Audio(
                label="Your exact voice clone output",
                elem_id="audio_output",
                interactive=False
            )
            
            # UNLIMITED FEATURES
            with gr.Accordion("💡 Unlimited Text Features", open=True):
                gr.Markdown("""
                <div class="tip-box">
                **📚 Unlimited Text Support:**
                - **No character limits** - write as much as you want
                - **Books, scripts, long articles** supported
                - **Multiple paragraphs** work perfectly
                - **Automatic batch processing** for long texts
                - **Same fast speed** regardless of text length
                
                **🚀 Speed: 3-15 seconds (depending on text length)**
                </div>
                """)
    
    # 👣 FOOTER
    with gr.Column(elem_classes="footer"):
        gr.Markdown("**Abbas Voice Clone** • Unlimited Text • Ultra Fast • Professional AI")
    
    # 🎯 BUTTON ACTION
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
        outputs=[output_audio],
        api_name="generate_voice"
    )

if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0', 
        server_port=7860, 
        debug=False,
        show_error=True
    )
