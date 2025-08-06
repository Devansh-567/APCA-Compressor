import numpy as np
from typing import List, Tuple, Dict
import pickle
import zlib
import time
import matplotlib.pyplot as plt


class APCACompressor:
    # Initialize the compressor with a specified window size
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.reset()

    # Reset the model weights, bias, and context window
    def reset(self):
        self.context_window = []
        self.model_weights = np.random.rand(self.window_size)
        self.bias = 0.0

    # Check if all values in input arrays are finite and valid
    def _is_valid(self, *arrays) -> bool:
        return all(np.all(np.isfinite(arr)) for arr in arrays)

    # Predict the next value based on the current context window
    def predict(self) -> float:
        if len(self.context_window) < self.window_size:
            return self.context_window[-1] if self.context_window else 0.0
        x = np.array(self.context_window[-self.window_size:])
        if not self._is_valid(x):
            return 0.0
        pred = float(np.dot(x, self.model_weights) + self.bias)
        if not np.isfinite(pred):
            self.reset()
            return 0.0
        return pred

    # Update the model weights and bias using gradient descent
    def update_model(self, actual: float, prediction: float, lr: float = 0.001):
        if len(self.context_window) < self.window_size:
            return
        x = np.array(self.context_window[-self.window_size:])
        if not self._is_valid(x, np.array([actual, prediction])):
            return
        error = actual - prediction
        if not np.isfinite(error):
            return
        grad = lr * error * x
        if not self._is_valid(grad, self.model_weights + grad):
            self.reset()  # Reset on instability
            return
        self.model_weights += grad
        self.bias += lr * error
        self.model_weights = np.clip(self.model_weights, -10, 10)  # Prevent weights from exploding
        self.bias = np.clip(self.bias, -10, 10)

    # Encode the input data by computing prediction deltas and compressing them
    def encode(self, data: List[float]) -> Tuple[bytes, Dict]:
        self.reset()
        encoded_deltas = []
        predictions = []
        start_time = time.time()

        for value in data:
            prediction = self.predict()
            delta = value - prediction
            encoded_deltas.append(delta)
            predictions.append(prediction)
            self.context_window.append(value)
            if len(self.context_window) > self.window_size * 2:
                self.context_window.pop(0)
            self.update_model(value, prediction)

        payload = {
            "deltas": encoded_deltas,
            "model_weights": self.model_weights.copy(),
            "bias": self.bias
        }

        raw_bytes = pickle.dumps(payload)
        compressed = zlib.compress(raw_bytes)
        end_time = time.time()

        metrics = {
            "original_size": len(pickle.dumps(data)),
            "compressed_size": len(compressed),
            "compression_ratio": len(pickle.dumps(data)) / len(compressed),
            "encoding_time": end_time - start_time,
            "predictions": predictions
        }

        return compressed, metrics

    # Decode the compressed data and reconstruct the original sequence
    def decode(self, compressed_data: bytes, original_data: List[float]) -> Tuple[List[float], Dict]:
        self.reset()
        start_time = time.time()
        raw_bytes = zlib.decompress(compressed_data)
        payload = pickle.loads(raw_bytes)

        encoded_deltas = payload["deltas"]
        self.model_weights = payload["model_weights"].copy()
        self.bias = payload["bias"]

        reconstructed = []
        predictions = []

        for delta in encoded_deltas:
            prediction = self.predict()
            if not np.isfinite(prediction):
                self.reset()
                prediction = 0.0
            value = prediction + delta
            value = np.clip(value, -1e6, 1e6)  # Avoid overflow
            predictions.append(prediction)
            reconstructed.append(float(value))
            self.context_window.append(float(value))
            if len(self.context_window) > self.window_size * 2:
                self.context_window.pop(0)
            self.update_model(value, prediction)

        end_time = time.time()

        N = min(len(original_data), len(reconstructed))
        original_np = np.array(original_data[:N])
        reconstructed_np = np.array(reconstructed[:N])

        if not self._is_valid(original_np, reconstructed_np):
            mae = float('nan')
            rmse = float('nan')
        else:
            abs_error = np.abs(reconstructed_np - original_np)
            if not self._is_valid(abs_error):
                mae = float('nan')
                rmse = float('nan')
            else:
                mae = np.mean(abs_error)
                rmse = np.sqrt(np.mean((reconstructed_np - original_np) ** 2))

        metrics = {
            "decoding_time": end_time - start_time,
            "reconstruction_error_mae": mae,
            "reconstruction_error_rmse": rmse,
            "predictions": predictions
        }

        return reconstructed, metrics


# Baseline compression using zlib and gzip (level 9)
def baseline_compress(data: List[float]) -> Dict[str, float]:
    raw_bytes = pickle.dumps(data)
    return {
        "gzip": len(zlib.compress(raw_bytes, level=9)),
        "zlib": len(zlib.compress(raw_bytes))
    }


# Visualize the original vs decompressed data and compression performance

def visualize_results(data: List[float], decompressed: List[float], predictions: List[float],
                      compression_ratio: float, mae: float, rmse: float,
                      encoding_time: float, decoding_time: float,
                      compressed: bytes):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot original vs decompressed signal
    axes[0].plot(data, label="Original")
    axes[0].plot(decompressed, '--', label="Decompressed")
    axes[0].set_title("Original vs Reconstructed Signal")
    axes[0].legend()
    axes[0].grid(True)

    # Plot prediction vs actual values during encoding
    axes[1].plot(data, label="Actual Value")
    axes[1].plot(predictions, '--', label="Model Prediction")
    axes[1].set_title("Prediction vs Actual (During Encoding)")
    axes[1].legend()
    axes[1].grid(True)

    # Compression ratio comparison with other methods
    baseline_sizes = baseline_compress(data)
    labels = ['APCA', 'zlib', 'gzip']
    sizes = [
        len(compressed),
        baseline_sizes['zlib'],
        baseline_sizes['gzip']
    ]
    ratios = [len(pickle.dumps(data)) / s for s in sizes]
    axes[2].bar(labels, ratios)
    axes[2].set_title("Compression Ratio Comparison")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("apca_performance_report.png", dpi=200)
    plt.show()

    # Print results to console
    print("\n📊 Final Metrics:")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Encoding Time: {encoding_time:.4f}s, Decoding Time: {decoding_time:.4f}s")

    print("\n📉 Reconstruction Errors:")
    if np.isnan(mae) or np.isinf(mae):
        print("MAE: N/A (invalid reconstruction)")
    else:
        print(f"MAE: {mae:.6f}")

    if np.isnan(rmse) or np.isinf(rmse):
        print("RMSE: N/A (invalid reconstruction)")
    else:
        print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    # Example data: sine wave with linear drift
    data = [np.sin(i * 0.1) + 0.02 * i for i in range(200)]

    compressor = APCACompressor(window_size=3)
    compressed, enc_metrics = compressor.encode(data)
    decompressed, dec_metrics = compressor.decode(compressed, original_data=data)

    print(f"Original size: {enc_metrics['original_size']} bytes")
    print(f"Compressed size: {enc_metrics['compressed_size']} bytes")
    print(f"Compression Ratio: {enc_metrics['compression_ratio']:.2f}")

    print("\n⏱️ Execution Times:")
    print(f"Encoding Time: {enc_metrics['encoding_time']:.4f}s")
    print(f"Decoding Time: {dec_metrics['decoding_time']:.4f}s")

    visualize_results(
        data=data,
        decompressed=decompressed,
        predictions=enc_metrics["predictions"],
        compression_ratio=enc_metrics['compression_ratio'],
        mae=dec_metrics['reconstruction_error_mae'],
        rmse=dec_metrics['reconstruction_error_rmse'],
        encoding_time=enc_metrics['encoding_time'],
        decoding_time=dec_metrics['decoding_time'],
        compressed=compressed
    )
