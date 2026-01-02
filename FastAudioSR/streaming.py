import onnxruntime as ort
import numpy as np
from collections import deque

class StreamingFASRONNX:
    def __init__(self, onnx_model_path, onnx_execution_provider = 'CPUExecutionProvider', n_cpu = 1):
        

        # Initialize ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = n_cpu
        session_options.inter_op_num_threads = n_cpu
        providers = [onnx_execution_provider]
        self.model = ort.InferenceSession(onnx_model_path, session_options, providers=providers)

        # Initialize input buffer for streaming
        self.input_streaming_buffer = deque(maxlen=16000*30)  # 30 seconds max input buffer
        self.output_streaming_buffer = []
        self.overlap_buffer = []

    def reset(self):
        """Reset the input and output streaming buffers."""
        self.input_streaming_buffer.clear()
        self.output_streaming_buffer = []
        self.overlap_buffer = []

    @staticmethod
    def crossfade(part1, part2, overlap):
        """
        Crossfade between two audio segments with linear ramping.
        
        Args:
            part1: First audio segment (list or array)
            part2: Second audio segment (list or array)
            overlap: Number of samples to crossfade
        
        Returns:
            Combined audio as numpy array
        """
        part1 = np.array(part1)
        part2 = np.array(part2)
        
        # Create linear ramp from 0 to 1
        ramp = np.linspace(0, 1, overlap)
        
        # Extract overlap regions
        part1_overlap = part1[-overlap:]
        part2_overlap = part2[:overlap]
        
        # Apply crossfade: fade out part1, fade in part2
        crossfaded = part1_overlap * (1 - ramp) + part2_overlap * ramp
        
        # Combine: part1 (without overlap) + crossfaded region + part2 (without overlap)
        result = np.concatenate([part1[:-overlap], crossfaded, part2[overlap:]])
        
        return result

    def run(self, input_speech):
        """Upsample the 16khz input speech to 48 khz"""
        if input_speech.shape != (1,):
            raise ValueError("Input speech must be a 1D array with shape (N,)")
    
        if input_speech.dtype != 'float32':
            raise ValueError("Input speech must be of type float32")
        
        if input_speech.abs().max() > 1.0:
            raise ValueError("Input speech values must be in the range [-1.0, 1.0]")

        upsampled = self.model.run(None, {'x': input_speech[None, None, :]})[0].flatten()
        upsampled_normalized = upsampled / (abs(upsampled).max() + 1e-7) * 0.999
        return upsampled_normalized
    
    def process_input(self, input_speech, chunk_size = 4000):
        """Upsample 16khz input speech to 48 khz in streaming mode. Some minimal degradation in quality is possible
        compared to non-streaming mode, especially for very small chunk sizes.
        
        Returns None if there is not enough input yet to produce a chunk of output.

        Args:
            input_speech (np.ndarray): 1D array of input speech samples at 16kHz.
            chunk_size (int): Number of input samples to process per chunk (default is 4000 at 16khz, or 250 ms).
                              Recommended minimum is 1000 samples for lowest latency and maximum is 16000 samples
                              for highest quality.
        """

        # Add audio to buffer
        self.input_streaming_buffer.extend(input_speech.tolist())

        # Upsample chunk of input audio
        if len(self.input_streaming_buffer) >= chunk_size:
            # Handle case where streaming is ongoing and buffer has at least one chunk
            input_chunk = np.array(
                self.overlap_buffer + [self.input_streaming_buffer.popleft() for _ in range(chunk_size)],
                dtype=np.float32
            )
            self.overlap_buffer = input_chunk[-500:].tolist()  # Save last 500 samples for overlap
            upsampled_chunk = self.model.run(None, {'x': input_chunk[None, None, :]})[0].flatten().tolist()

            # Cross fade overlap region between total current output buffer and current chunk
            if len(self.output_streaming_buffer) > 0:
                # prev_chunk_overlap = np.array(
                #     [self.output_streaming_buffer.popleft() for _ in range(500)],
                #     dtype=np.float32
                # )

                self.output_streaming_buffer = self.crossfade(
                    self.output_streaming_buffer,
                    upsampled_chunk[1500 - 500:],
                    overlap=500
                )
            else:
                self.output_streaming_buffer = upsampled_chunk

        else:
            # Not enough input yet to produce a chunk
            return None
        
    def get_output(self, n_samples = 12000):
        """Generator that yields available upsampled output chunks.
        
        Args:
            num_samples (int): Number of output samples to yield per chunk (should correspond to the same total time,
                               at 48khz, as the input chunk size at 16khz).

        Yields:
            np.ndarray: 1D np.ndarray chunks of upsampled speech at 48khz.
        """

        cnt = 0
        orig_n_samples = n_samples
        while len(self.output_streaming_buffer) >= n_samples:
            if cnt == 0:
                n_samples = n_samples - 2000
            else:
                n_samples = orig_n_samples
            
            # Get chunk to return
            chunk = self.output_streaming_buffer[:n_samples]

            # Remove yielded samples from output buffer
            self.output_streaming_buffer = self.output_streaming_buffer[n_samples:]

            # Increment counter
            cnt += 1

            yield chunk

        # Yield any remaining samples in the buffer
        if len(self.output_streaming_buffer) > 0:
            yield self.output_streaming_buffer
            self.output_streaming_buffer = []