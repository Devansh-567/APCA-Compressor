# 🔄 APCA-Compressor

A Python project that implements an **Adaptive Predictive Compression Algorithm (APCA)** for compressing time-series data using autoregressive modeling and delta encoding. It visualizes how accurately the data can be reconstructed and compares its performance with standard compression techniques.

---

## 📌 What It Is

This is a **lossy compressor for time-series or numeric sequences**, built with:

- Sliding window autoregressive prediction
- Delta encoding between predicted and actual values
- Compression using `zlib`
- Reconstruction and error measurement (MAE, RMSE)
- Visualization of the compression performance

---

## 📊 What It Does

- Compresses numeric sequences using a predictive model
- Decompresses them while maintaining high fidelity
- Compares performance against `zlib` and `gzip`
- Generates a visual report

---

### 📈 Sample Result Output (from `apca_compressor.py`)

```bash
Original size: 3902 bytes
Compressed size: 1909 bytes
Compression Ratio: 2.04×

⏱️ Execution Times:
Encoding Time: 0.0175s
Decoding Time: 0.0171s

📉 Reconstruction Errors:
MAE: 504072.291840
RMSE: 708119.914650

## 💻 How to Clone and Run

git clone https://github.com/Devansh-567/APCA-Compressor.git

cd APCA-Compressor

pip install -r requirements.txt

python apca_compressor.py
```
