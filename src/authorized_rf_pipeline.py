"""
SPECTRA-X controlled RF communication pipeline.

Demo-only signal chain:
TRANSMIT -> RF CAPTURE -> DETECT -> CLASSIFY -> ANALYZE -> DECODE

The transmitter is generated locally by this simulator. No external RF
receiver, third-party communication, or arbitrary intercepted payload is used.
"""

from __future__ import annotations

import binascii
import numpy as np


PREAMBLE = np.array(
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    dtype=np.uint8,
)


def _bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bytes_from_bits(bits: np.ndarray) -> bytes:
    usable = len(bits) - (len(bits) % 8)
    if usable <= 0:
        return b""
    return np.packbits(bits[:usable]).tobytes()


def build_test_packet(message: str) -> tuple[np.ndarray, bytes]:
    """
    Build a controlled test packet.

    Layout:
        PREAMBLE | LENGTH(1 byte) | PAYLOAD | CRC32(4 bytes)
    """
    payload = message.encode("utf-8")
    if len(payload) > 200:
        raise ValueError("Demo payload must be <= 200 bytes.")

    length = bytes([len(payload)])
    body = length + payload
    crc = binascii.crc32(body).to_bytes(4, "big")
    bits = np.concatenate(
        [PREAMBLE, _bits_from_bytes(body + crc)]
    ).astype(np.uint8)
    return bits, payload


def bpsk_modulate(
    bits: np.ndarray,
    samples_per_symbol: int = 16,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Controlled BPSK baseband waveform.
    """
    symbols = (2.0 * bits.astype(float)) - 1.0
    return amplitude * np.repeat(symbols, samples_per_symbol).astype(np.float32)


def simulate_transmission(
    message: str = "HELLO FROM SPECTRA-X",
    samples_per_symbol: int = 16,
    snr_db: float = 18.0,
    seed: int = 42,
) -> dict:
    """
    Generate a reproducible controlled RF/baseband test transmission.
    """
    rng = np.random.default_rng(seed)
    bits, payload = build_test_packet(message)

    tx = bpsk_modulate(bits, samples_per_symbol)

    signal_power = float(np.mean(tx**2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=tx.shape)

    rx = tx + noise.astype(np.float32)

    return {
        "message": message,
        "payload": payload,
        "bits": bits,
        "tx_waveform": tx,
        "rx_waveform": rx.astype(np.float32),
        "snr_db": float(snr_db),
        "samples_per_symbol": int(samples_per_symbol),
        "sample_rate": int(samples_per_symbol),
    }


def detect_signal(
    waveform: np.ndarray,
    threshold_ratio: float = 0.18,
) -> dict:
    """
    Energy-based detection for the simulated receiver.
    """
    x = np.asarray(waveform, dtype=float)
    power = float(np.mean(x**2))
    peak = float(np.max(np.abs(x)))
    threshold = max(0.02, peak * threshold_ratio)
    detected = bool(power > threshold**2 * 0.08)

    return {
        "detected": detected,
        "power": power,
        "peak": peak,
        "threshold": threshold,
    }


def extract_features(
    waveform: np.ndarray,
    sample_rate: float,
) -> np.ndarray:
    """
    Five deterministic signal features for the prototype pipeline.

    These are simulator features, not a claim that they are the exact
    five training features of the saved Random Forest.
    """
    x = np.asarray(waveform, dtype=float)

    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)

    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    zero_crossings = float(np.mean(np.diff(np.signbit(x)).astype(float)))

    if len(spectrum) > 1:
        dominant_frequency = float(freqs[1:][np.argmax(spectrum[1:])])
    else:
        dominant_frequency = 0.0

    return np.array(
        [rms, peak, mean_abs, zero_crossings, dominant_frequency],
        dtype=np.float32,
    )


def classify_signal(
    waveform: np.ndarray,
    sample_rate: float,
    model=None,
    scaler=None,
) -> dict:
    """
    Classify the controlled signal.

    If the project's saved model/scaler are supplied and compatible, they
    are used. Otherwise the simulator reports the known controlled class.
    """
    features = extract_features(waveform, sample_rate)

    if model is not None:
        model_features = features.reshape(1, -1)
        if scaler is not None:
            model_features = scaler.transform(model_features)

        try:
            prediction = model.predict(model_features)[0]
            confidence = None

            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(model_features)[0]))

            return {
                "class": str(prediction),
                "confidence": confidence,
                "features": features,
                "source": "project-model",
            }
        except Exception:
            pass

    return {
        "class": "CONTROLLED_TEST_BPSK",
        "confidence": 1.0,
        "features": features,
        "source": "simulator-known-class",
    }


def analyze_signal(
    waveform: np.ndarray,
    sample_rate: float,
    snr_db: float,
) -> dict:
    """
    Communication analysis for the controlled test signal.
    """
    x = np.asarray(waveform, dtype=float)

    return {
        "modulation": "BPSK",
        "estimated_bandwidth_hz": float(sample_rate),
        "snr_db": float(snr_db),
        "sample_count": int(len(x)),
        "analysis_status": "READY FOR CONTROLLED DECODE",
    }


def decode_test_signal(
    waveform: np.ndarray,
    samples_per_symbol: int = 16,
) -> dict:
    """
    Recover the payload from the locally generated controlled test packet.
    """
    x = np.asarray(waveform, dtype=float)

    usable = (len(x) // samples_per_symbol) * samples_per_symbol
    symbols = x[:usable].reshape(-1, samples_per_symbol).mean(axis=1)
    bits = (symbols >= 0).astype(np.uint8)

    preamble_len = len(PREAMBLE)
    if len(bits) < preamble_len + 8 + 32:
        return {
            "success": False,
            "message": None,
            "reason": "FRAME TOO SHORT",
        }

    # Search for the known controlled-test preamble.
    match_index = None
    for i in range(len(bits) - preamble_len + 1):
        if np.array_equal(bits[i:i + preamble_len], PREAMBLE):
            match_index = i
            break

    if match_index is None:
        return {
            "success": False,
            "message": None,
            "reason": "PREAMBLE NOT FOUND",
        }

    payload_start = match_index + preamble_len
    length_bits = bits[payload_start:payload_start + 8]
    if len(length_bits) < 8:
        return {"success": False, "message": None, "reason": "NO LENGTH"}

    payload_len = int.from_bytes(
        _bytes_from_bits(length_bits),
        "big",
    )

    frame_bits = 8 + payload_len * 8 + 32
    frame = bits[payload_start:payload_start + frame_bits]

    if len(frame) < frame_bits:
        return {"success": False, "message": None, "reason": "INCOMPLETE FRAME"}

    body = _bytes_from_bits(frame[:8 + payload_len * 8])
    received_crc = _bytes_from_bits(frame[8 + payload_len * 8:])

    calculated_crc = binascii.crc32(body).to_bytes(4, "big")

    if received_crc != calculated_crc:
        return {
            "success": False,
            "message": None,
            "reason": "CRC FAILED",
        }

    payload = body[1:]
    try:
        message = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "success": False,
            "message": None,
            "reason": "PAYLOAD IS NOT UTF-8",
        }

    return {
        "success": True,
        "message": message,
        "reason": "CRC PASS",
    }


def run_controlled_pipeline(
    message: str = "HELLO FROM SPECTRA-X",
    model=None,
    scaler=None,
) -> dict:
    """
    Execute the complete controlled pipeline in one call.
    """
    sim = simulate_transmission(message=message)

    detection = detect_signal(sim["rx_waveform"])
    classification = classify_signal(
        sim["rx_waveform"],
        sim["sample_rate"],
        model=model,
        scaler=scaler,
    )
    analysis = analyze_signal(
        sim["rx_waveform"],
        sim["sample_rate"],
        sim["snr_db"],
    )
    decoding = decode_test_signal(
        sim["rx_waveform"],
        sim["samples_per_symbol"],
    )

    return {
        "transmit": {
            "message": message,
            "sample_count": len(sim["tx_waveform"]),
        },
        "capture": {
            "sample_count": len(sim["rx_waveform"]),
            "snr_db": sim["snr_db"],
            "waveform": sim["rx_waveform"],
        },
        "detection": detection,
        "classification": classification,
        "analysis": analysis,
        "decoding": decoding,
    }
