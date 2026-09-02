from src.authorized_rf_pipeline import run_controlled_pipeline

result = run_controlled_pipeline("HELLO FROM SPECTRA-X")

print("=" * 68)
print("SPECTRA-X CONTROLLED RF COMMUNICATION PIPELINE")
print("=" * 68)
print("1. TRANSMIT   :", result["transmit"]["message"])
print("2. RF CAPTURE :", result["capture"]["sample_count"], "samples")
print("3. DETECT     :", result["detection"]["detected"])
print("4. CLASSIFY   :", result["classification"]["class"])
print("5. ANALYZE    :", result["analysis"]["modulation"])
print("6. DECODE     :", result["decoding"]["reason"])
print("7. MESSAGE    :", result["decoding"]["message"])
print("=" * 68)
