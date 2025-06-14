import numpy as np

# Load with pickle enabled
raw_data = np.load("SMD_test-1000.npy", allow_pickle=True)

# Show info
print("Loaded shape:", raw_data.shape)
print("First element type:", type(raw_data[0]))
print("First element shape:", raw_data[0].shape if hasattr(raw_data[0], 'shape') else "irregular")

# Try converting to stacked float array
try:
    data = np.stack(raw_data).astype(np.float32)
    print("Converted to array with shape:", data.shape)

    # Save cleaned version
    np.save("SMD_test_clean.npy", data)
    print("Saved cleaned data to: dataset/SMD_test_clean.npy")

except Exception as e:
    print("ERROR: Could not convert data:", e)
