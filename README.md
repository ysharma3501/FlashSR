# FlashSR

This is a tiny audio super-resolution model based on [hierspeech++](https://github.com/sh-lee-prml/HierSpeechpp) that upscales 16khz audio into much clearer 48khz audio at speed over 200x realtime to 400x realtime!

FlashSR is released under an apache-2.0 license.

Model link: https://huggingface.co/YatharthS/FlashSR

## GUI Usage (Easy!)

For a simple web-based GUI to enhance your audio files:

1. Install dependencies:
```bash
pip install -r requirements-gui.txt
pip install torch torchaudio  # or your preferred PyTorch installation
```

2. Run the GUI:
```bash
python app.py
```

3. The GUI will automatically open in your browser at `http://127.0.0.1:7860`

4. Follow the simple steps in the GUI:
   - Click "Load Model" to download and initialize the model
   - Upload your audio file (any format - MP3, WAV, FLAC, etc.)
   - Click "Enhance Audio"
   - Listen to and download your enhanced 48kHz audio!

## Programmatic Usage
Simple 1 line installation:

```
pip install git+https://github.com/ysharma3501/FlashSR.git
```

Load model:
```python
from FastAudioSR import FASR
from huggingface_hub import hf_hub_download

file_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="upsampler.pth", local_dir=".")
upsampler = FASR(file_path)

_ = upsampler.model.half()
```

Run the model:
```python
import librosa
import torch
from IPython.display import Audio

y, sr = librosa.load("path/to/audio.wav", sr=16000) ## resamples to 16khz sampling_rate
lowres_wav = torch.from_numpy(y).unsqueeze(0).half()

new_wav = upsampler.run(lowres_wav)
Audio(new_wav, rate=48000)
```


## Onnx usage
Big thanks to [Xenova](https://github.com/xenova/) for converting FlashSR to onnx and decreasing model size to just **500kb** making it perfect for edge devices!

Installation:
```
pip install onnxruntime librosa soundfile huggingface-hub
```

Running the model:
```python
import librosa
import onnxruntime as ort
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

# Download the ONNX model from HF Hub
model_path = hf_hub_download(repo_id="YatharthS/FlashSR", filename="model.onnx", subfolder="onnx")

# Load audio file at 16kHz
y, sr = librosa.load("path/to/audio.wav", sr=16000)  # Resamples to 16kHz
lowres_wav = y[np.newaxis, :]  # Add batch dimension

# Create ONNX session and run inference
ort_session = ort.InferenceSession(model_path)
onnx_output = ort_session.run(["reconstruction"], {"audio_values": lowres_wav})[0]

# Save output audio at 48kHz
sf.write('output.wav', onnx_output.squeeze(0), samplerate=48000)
```

## Final notes
Thanks very much to the authors of hierspeech++. Thanks for checking out this repository as well.

Stars would be well appreciated, thank you.

Email: yatharthsharma3501@gmail.com
