import gradio as gr
import torch
import librosa
import soundfile as sf
import numpy as np
import os
import tempfile
from FastAudioSR import FASR
from huggingface_hub import hf_hub_download

# Global variable to store the model
model = None

def load_model():
    """Load the FlashSR model from HuggingFace Hub"""
    global model
    print("Loading FlashSR model...")
    try:
        file_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="upsampler.pth", local_dir="./models")
        model = FASR(file_path)

        # Use half precision if CUDA is available
        if torch.cuda.is_available():
            _ = model.model.half()
            print("Model loaded successfully with CUDA (half precision)")
        else:
            print("Model loaded successfully on CPU")

        return "Model loaded successfully!"
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return f"Error loading model: {str(e)}"

def enhance_audio(audio_file):
    """
    Enhance the uploaded audio file using FlashSR

    Args:
        audio_file: Path to the uploaded audio file

    Returns:
        tuple: (sample_rate, audio_array) for Gradio audio output
    """
    global model

    if model is None:
        return None, "Please load the model first by clicking 'Load Model'"

    if audio_file is None:
        return None, "Please upload an audio file"

    try:
        # Load audio at 16kHz (model input requirement)
        print(f"Loading audio file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=16000)

        # Convert to torch tensor
        if torch.cuda.is_available():
            lowres_wav = torch.from_numpy(y).unsqueeze(0).half()
        else:
            lowres_wav = torch.from_numpy(y).unsqueeze(0).float()

        # Run enhancement
        print("Running audio enhancement...")
        enhanced_wav = model.run(lowres_wav)

        # Convert to numpy for Gradio
        enhanced_audio = enhanced_wav.cpu().numpy()

        # Ensure it's 1D array
        if enhanced_audio.ndim > 1:
            enhanced_audio = enhanced_audio.squeeze()

        print("Audio enhancement completed!")

        # Return as (sample_rate, audio_array) tuple for Gradio
        return (48000, enhanced_audio), "Enhancement completed successfully!"

    except Exception as e:
        error_msg = f"Error during enhancement: {str(e)}"
        print(error_msg)
        return None, error_msg

# Custom CSS for better styling
custom_css = """
#header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
#header h1 {
    color: white;
    margin: 0;
    font-size: 2.5em;
}
#header p {
    color: #f0f0f0;
    margin: 10px 0 0 0;
}
.container {
    max-width: 1200px;
    margin: auto;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="FlashSR - Audio Enhancement") as demo:

    # Header
    with gr.Row(elem_id="header"):
        gr.Markdown(
            """
            # FlashSR - Audio Enhancement GUI
            ### Upscale 16kHz audio to crystal clear 48kHz at 200-400x realtime speed!
            """
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Step 1: Load Model")
            load_btn = gr.Button("Load Model", variant="primary", size="lg")
            load_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Step 2: Upload Audio")
            audio_input = gr.Audio(
                label="Upload Audio File (will be resampled to 16kHz)",
                type="filepath"
            )
            enhance_btn = gr.Button("Enhance Audio", variant="primary", size="lg")
            enhance_status = gr.Textbox(label="Enhancement Status", interactive=False)

        with gr.Column():
            gr.Markdown("## Step 3: Listen & Download")
            audio_output = gr.Audio(
                label="Enhanced Audio (48kHz)",
                type="numpy"
            )

    # Info section
    with gr.Row():
        gr.Markdown(
            """
            ## How it works

            1. **Load Model**: Click the button to download and load the FlashSR model from HuggingFace Hub
            2. **Upload Audio**: Upload any audio file (MP3, WAV, FLAC, etc.) - it will be automatically resampled to 16kHz
            3. **Enhance**: Click 'Enhance Audio' to upscale your audio to 48kHz with improved clarity
            4. **Listen & Download**: Play the enhanced audio directly in your browser or download it

            **Note**: The model works best with speech audio. First-time use will download the model (~10MB).
            """
        )

    # Event handlers
    load_btn.click(
        fn=load_model,
        inputs=None,
        outputs=load_status
    )

    enhance_btn.click(
        fn=enhance_audio,
        inputs=audio_input,
        outputs=[audio_output, enhance_status]
    )

if __name__ == "__main__":
    print("Starting FlashSR GUI...")
    print("The interface will open in your browser automatically.")
    print("If it doesn't, navigate to the URL shown below.")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
