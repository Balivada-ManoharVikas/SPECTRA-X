from pathlib import Path
from datetime import datetime
import math

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# SPECTRA-X
# AI-POWERED RF RECONNAISSANCE • DEMO PROTOTYPE
# ============================================================

st.set_page_config(
    page_title="SPECTRA-X | RF Reconnaissance",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "rf_environment.csv"
MODEL_FILE = ROOT / "models" / "spectrax_rf_balanced_safe.joblib"


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #050b13 !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 100% !important;
    padding: 8px 12px 8px 12px !important;
}

section[data-testid="stSidebar"] {
    display: none !important;
}

.sx-header {
    background: linear-gradient(90deg,#071522,#091c2b);
    border: 1px solid #17405a;
    border-radius: 10px;
    padding: 9px 16px;
    margin-bottom: 8px;
}

.sx-brand {
    font-size: 24px;
    font-weight: 900;
    letter-spacing: 1px;
    color: #f4f8fb;
}

.sx-subtitle {
    font-size: 10px;
    color: #61d7ff;
    letter-spacing: 1px;
    margin-top: 2px;
}

.sx-live {
    color: #3ff08a;
    font-size: 10px;
    font-weight: 900;
}

.panel {
    background: #07111d;
    border: 1px solid #17384f;
    border-radius: 9px;
    padding: 9px;
    margin-bottom: 7px;
}

.panel-title {
    color: #62d8ff;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}

.status-good {
    color: #46ef8b;
    font-weight: 800;
    font-size: 10px;
    line-height: 1.6;
}

.status-info {
    color: #5eafff;
    font-weight: 800;
    font-size: 10px;
    line-height: 1.6;
}

.metric-big {
    color: #f5f8fa;
    font-size: 23px;
    font-weight: 900;
}

.metric-label {
    color: #7892a4;
    font-size: 9px;
}

.event-row {
    color: #d6e3eb;
    font-size: 10px;
    padding: 5px 2px;
    border-bottom: 1px solid #142b3d;
}

.communication-message {
    background: #02070c;
    border: 1px solid #1f506b;
    border-radius: 7px;
    padding: 9px;
    color: #e7f7ff;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.5;
    min-height: 46px;
}

.decode-ok {
    color: #46ef8b;
    font-weight: 900;
}

.decode-wait {
    color: #7e96a5;
    font-weight: 800;
}

.log-box {
    height: 105px;
    overflow: hidden;
    background: #02070c;
    border: 1px solid #17384f;
    border-radius: 7px;
    padding: 7px;
    color: #c7d7e0;
    font-family: monospace;
    font-size: 8px;
    line-height: 1.45;
}

.footer {
    text-align: center;
    color: #506b7b;
    font-size: 8px;
    padding-top: 3px;
}

div.stButton > button {
    border-radius: 7px !important;
    min-height: 34px !important;
    font-weight: 800 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data
def load_rf_data():
    df = pd.read_csv(DATA_FILE)

    required = {"time_slot", "band", "transmission"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {', '.join(sorted(missing))}"
        )

    df = df.copy()

    for col in ["time_slot", "band", "transmission"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["time_slot"] = df["time_slot"].astype(int)
    df["band"] = df["band"].astype(int)
    df["transmission"] = df["transmission"].clip(0, 1).astype(int)

    return df


@st.cache_resource
def load_rf_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


try:
    rf_df = load_rf_data()
    DATA_OK = True
    DATA_ERROR = ""
except Exception as exc:
    rf_df = pd.DataFrame()
    DATA_OK = False
    DATA_ERROR = str(exc)

try:
    rf_model = load_rf_model()
    MODEL_OK = rf_model is not None
except Exception as exc:
    rf_model = None
    MODEL_OK = False


# ============================================================
# SESSION STATE
# ============================================================

if "running" not in st.session_state:
    st.session_state.running = False

if "x" not in st.session_state:
    st.session_state.x = -35.0

if "y" not in st.session_state:
    st.session_state.y = -35.0

if "direction" not in st.session_state:
    st.session_state.direction = 0

if "slot" not in st.session_state:
    st.session_state.slot = 0

if "flight_path" not in st.session_state:
    st.session_state.flight_path = [(-35.0, -35.0)]

if "logs" not in st.session_state:
    st.session_state.logs = [
        "[SYSTEM] SPECTRA-X initialized",
        "[SYSTEM] 3D RF environment loaded",
        "[SYSTEM] RF sensor online",
        "[SYSTEM] ML pipeline ready"
        if MODEL_OK else
        "[SYSTEM] ML pipeline unavailable",
    ]

if "waterfall" not in st.session_state:
    st.session_state.waterfall = None

if "decoded_message" not in st.session_state:
    st.session_state.decoded_message = None

if "decode_status" not in st.session_state:
    st.session_state.decode_status = "WAITING"


def add_log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(
        f"[{timestamp}] {message}"
    )
    st.session_state.logs = st.session_state.logs[-10:]


# ============================================================
# FIXED MISSION PARAMETERS
# ============================================================

SOURCES = [
    {
        "id": "A",
        "x": -30,
        "y": -20,
        "freq": 88.5,
        "type": "FM Radio",
        "class": "RF Source A",
        "priority": "LOW",
    },
    {
        "id": "B",
        "x": 25,
        "y": 20,
        "freq": 2400.0,
        "type": "WiFi",
        "class": "RF Source B",
        "priority": "HIGH",
    },
    {
        "id": "C",
        "x": 30,
        "y": -25,
        "freq": 1300.0,
        "type": "RF Link",
        "class": "RF Source C",
        "priority": "MEDIUM",
    },
    {
        "id": "D",
        "x": -20,
        "y": 30,
        "freq": 900.0,
        "type": "Cellular",
        "class": "RF Source D",
        "priority": "HIGH",
    },
]


# ============================================================
# TERRAIN
# ============================================================

terrain_x = np.linspace(-50, 50, 55)
terrain_y = np.linspace(-50, 50, 55)
TX, TY = np.meshgrid(terrain_x, terrain_y)

TZ = (
    20 * np.exp(
        -(((TX + 28) ** 2) + ((TY - 25) ** 2)) / 400
    )
    + 28 * np.exp(
        -(((TX - 25) ** 2) + ((TY - 22) ** 2)) / 450
    )
    + 19 * np.exp(
        -(((TX + 28) ** 2) + ((TY + 22) ** 2)) / 430
    )
    + 24 * np.exp(
        -(((TX - 28) ** 2) + ((TY + 25) ** 2)) / 420
    )
    + 3 * np.sin(TX / 8) * np.cos(TY / 9)
)

TZ -= TZ.min()


def terrain_height(x, y):
    ix = np.abs(terrain_x - x).argmin()
    iy = np.abs(terrain_y - y).argmin()
    return float(TZ[iy, ix])


# ============================================================
# RF OBSERVATION FROM YOUR CSV
# ============================================================

def current_observation():
    if rf_df.empty:
        return {
            "time_slot": 0,
            "band": 0,
            "transmission": 0,
            "frequency": 0.0,
            "power": -95.0,
            "snr": 0.0,
            "distance": 999.0,
        }

    row = rf_df.iloc[
        st.session_state.slot % len(rf_df)
    ]

    max_band = max(
        int(rf_df["band"].max()),
        1
    )

    # Maps the actual CSV band index into the simulator's
    # 30 MHz – 6 GHz operating range.
    frequency = (
        30.0
        + (int(row["band"]) / max_band) * 5970.0
    )

    nearest = min(
        SOURCES,
        key=lambda src:
        (src["x"] - st.session_state.x) ** 2
        + (src["y"] - st.session_state.y) ** 2
    )

    distance = math.sqrt(
        (nearest["x"] - st.session_state.x) ** 2
        + (nearest["y"] - st.session_state.y) ** 2
    )

    transmission = int(row["transmission"])

    # Simulation sensor model driven by the real CSV
    # transmission state and drone/source geometry.
    power = (
        -96
        + 60 * transmission
        - min(distance * 0.7, 30)
    )

    snr = max(
        0.0,
        8 + 35 * transmission - distance * 0.2
    )

    return {
        "time_slot": int(row["time_slot"]),
        "band": int(row["band"]),
        "transmission": transmission,
        "frequency": frequency,
        "power": power,
        "snr": snr,
        "distance": distance,
    }


# ============================================================
# MODEL FEATURE ADAPTER
# ============================================================

def build_model_input(obs):
    """
    Uses feature names when the saved estimator exposes them.
    Otherwise uses the five-feature prototype mapping:
        frequency, power, bandwidth-proxy, snr, transmission
    """

    bandwidth_proxy = float(
        abs(obs["frequency"] - 1000.0) / 1000.0
    )

    feature_values = {
        "frequency": obs["frequency"],
        "freq": obs["frequency"],
        "power": obs["power"],
        "power_dbm": obs["power"],
        "snr": obs["snr"],
        "signal_power": obs["power"],
        "transmission": obs["transmission"],
        "tx": obs["transmission"],
        "band": obs["band"],
        "band_index": obs["band"],
        "bandwidth": bandwidth_proxy,
        "bandwidth_proxy": bandwidth_proxy,
        "time_slot": obs["time_slot"],
    }

    names = getattr(
        rf_model,
        "feature_names_in_",
        None
    )

    if names is not None:
        values = []

        for name in names:
            key = str(name).strip().lower()

            if key in feature_values:
                values.append(feature_values[key])
            else:
                # Unknown feature: retain deterministic simulation
                # value instead of silently failing the dashboard.
                values.append(0.0)

        return np.asarray(
            values,
            dtype=float
        ).reshape(1, -1)

    expected = int(
        getattr(
            rf_model,
            "n_features_in_",
            5
        )
    )

    prototype = [
        obs["frequency"],
        obs["power"],
        bandwidth_proxy,
        obs["snr"],
        obs["transmission"],
    ]

    if expected <= len(prototype):
        prototype = prototype[:expected]
    else:
        prototype.extend(
            [0.0] * (expected - len(prototype))
        )

    return np.asarray(
        prototype,
        dtype=float
    ).reshape(1, -1)


def classify(obs):
    if not MODEL_OK:
        return "MODEL OFFLINE", 0.0, []

    try:
        X = build_model_input(obs)

        prediction = rf_model.predict(X)[0]

        top = []
        confidence = 0.0

        if hasattr(rf_model, "predict_proba"):
            probabilities = rf_model.predict_proba(X)[0]
            classes = rf_model.classes_

            order = np.argsort(
                probabilities
            )[::-1]

            confidence = float(
                probabilities[order[0]]
            )

            for index in order[:5]:
                top.append(
                    (
                        str(classes[index]),
                        float(probabilities[index])
                    )
                )

        return (
            str(prediction),
            confidence,
            top,
        )

    except Exception as exc:
        return (
            "FEATURE MAPPING ERROR",
            0.0,
            [(str(exc), 0.0)],
        )


# ============================================================
# AUTHORIZED TEST-SIGNAL COMMUNICATION ANALYSIS
# ============================================================

TEST_PAYLOADS = {
    "A": "SPECTRA-X TEST SIGNAL A — FM IDENTIFICATION",
    "B": "SPECTRA-X TEST MESSAGE — WIFI TEST PAYLOAD",
    "C": "SPECTRA-X TEST SIGNAL C — RF LINK IDENTIFICATION",
    "D": "SPECTRA-X TEST SIGNAL D — CELLULAR TEST PAYLOAD",
}


def analyze_authorized_test_signal(obs):
    """
    Demo-only communication analysis.

    The decoded payload is a predefined message associated with the
    simulated RF source. It is intentionally not arbitrary-message
    interception or third-party communications decoding.
    """
    nearest = min(
        SOURCES,
        key=lambda src:
        (src["x"] - st.session_state.x) ** 2
        + (src["y"] - st.session_state.y) ** 2
    )

    message = TEST_PAYLOADS.get(
        nearest["id"],
        "SPECTRA-X AUTHORIZED TEST PAYLOAD"
    )

    # A successful demo decode requires either an active CSV
    # transmission or an explicit operator-triggered test frame.
    return {
        "source_id": nearest["id"],
        "source_type": nearest["type"],
        "message": message,
        "frame_status": "VALID TEST FRAME",
        "decode_status": "DECODED",
    }


# ============================================================
# DRONE CLOSED LOOP
# ============================================================

def update_drone(speed):
    if not st.session_state.running:
        return

    step = max(
        0.45,
        min(2.2, speed * 0.10)
    )

    x = st.session_state.x
    y = st.session_state.y
    direction = st.session_state.direction

    # 0 = EAST
    # 1 = NORTH
    # 2 = WEST
    # 3 = SOUTH

    if direction == 0:
        x += step
        if x >= 35:
            x = 35
            direction = 1

    elif direction == 1:
        y += step
        if y >= 35:
            y = 35
            direction = 2

    elif direction == 2:
        x -= step
        if x <= -35:
            x = -35
            direction = 3

    else:
        y -= step
        if y <= -35:
            y = -35
            direction = 0

    st.session_state.x = x
    st.session_state.y = y
    st.session_state.direction = direction

    st.session_state.flight_path.append((x, y))

    # Keep the visual path compact.
    st.session_state.flight_path = (
        st.session_state.flight_path[-120:]
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="sx-header">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div class="sx-brand">📡 SPECTRA-X</div>
            <div class="sx-subtitle">
                AI POWERED RF RECONNAISSANCE • ADAPTIVE SCAN SIMULATION
            </div>
        </div>
        <div class="sx-live">● LIVE &nbsp; RF SENSOR | ML ENGINE | SCAN ENGINE</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(run_every=0.35)
def dashboard():

    altitude = st.session_state.get(
        "altitude", 120
    )

    speed = st.session_state.get(
        "speed", 15
    )

    update_drone(speed)

    if st.session_state.running and DATA_OK:
        st.session_state.slot = (
            st.session_state.slot + 1
        ) % len(rf_df)

    obs = current_observation()
    predicted, confidence, top_predictions = classify(obs)

    drone_ground_z = terrain_height(
        st.session_state.x,
        st.session_state.y
    )

    drone_z = drone_ground_z + altitude

    left, center, right = st.columns(
        [1.25, 6.2, 1.8],
        gap="small"
    )

    # ========================================================
    # LEFT — CONTROLS
    # ========================================================

    with left:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">SIMULATOR CONTROLS</div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:
            if st.button(
                "▶ START",
                use_container_width=True
            ):
                st.session_state.running = True
                add_log(
                    "MISSION STARTED — DRONE LOOP ACTIVE"
                )

        with c2:
            if st.button(
                "⏸ PAUSE",
                use_container_width=True
            ):
                st.session_state.running = False
                add_log("MISSION PAUSED")

        if st.button(
            "↻ RESET MISSION",
            use_container_width=True
        ):
            st.session_state.running = False
            st.session_state.x = -35.0
            st.session_state.y = -35.0
            st.session_state.direction = 0
            st.session_state.slot = 0
            st.session_state.flight_path = [
                (-35.0, -35.0)
            ]
            st.session_state.waterfall = None
            st.session_state.decoded_message = None
            st.session_state.decode_status = "WAITING"
            add_log("MISSION RESET")

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        altitude_value = st.slider(
            "Drone Altitude (m)",
            30,
            300,
            altitude,
            10
        )

        speed_value = st.slider(
            "Drone Speed (m/s)",
            1,
            40,
            speed,
        )

        st.session_state.altitude = altitude_value
        st.session_state.speed = speed_value

        scan_mode = st.selectbox(
            "Scan Strategy",
            [
                "Smart Scan",
                "Priority Scan",
                "Wideband Sweep",
            ],
            key="scan_mode",
        )

        state = (
            "ACTIVE"
            if st.session_state.running
            else "STANDBY"
        )

        st.markdown(
            f"""
<div class="panel">
<div class="panel-title">DRONE TELEMETRY</div>

<div class="metric-big" style="color:#45ef8a">
{state}
</div>
<div class="metric-label">Mission state</div>

<br>

<div class="metric-big">{altitude_value} m</div>
<div class="metric-label">Altitude</div>

<br>

<div class="metric-big">{speed_value} m/s</div>
<div class="metric-label">Speed</div>

<br>

<div class="metric-big">{st.session_state.x:.1f} m</div>
<div class="metric-label">East Position</div>

<br>

<div class="metric-big">{st.session_state.y:.1f} m</div>
<div class="metric-label">North Position</div>
</div>
""",
            unsafe_allow_html=True
        )

    # ========================================================
    # CENTER — ONE SINGLE 3D MAP
    # ========================================================

    with center:

        st.markdown(
            '<div class="panel-title">'
            '🛰 3D RF MISSION ENVIRONMENT • SATELLITE / TACTICAL VIEW'
            '</div>',
            unsafe_allow_html=True
        )

        fig = go.Figure()

        # Terrain
        fig.add_trace(
            go.Surface(
                x=TX,
                y=TY,
                z=TZ,
                colorscale="Earth",
                showscale=False,
                hovertemplate=(
                    "EAST %{x:.1f} m"
                    "<br>NORTH %{y:.1f} m"
                    "<br>TERRAIN %{z:.1f} m"
                    "<extra></extra>"
                ),
                name="Terrain",
            )
        )

        # RF sources + coverage
        for source in SOURCES:

            theta = np.linspace(
                0,
                2 * np.pi,
                70
            )

            radius = 10

            cx = (
                source["x"]
                + radius * np.cos(theta)
            )

            cy = (
                source["y"]
                + radius * np.sin(theta)
            )

            cz = np.array([
                terrain_height(a, b) + 0.8
                for a, b in zip(cx, cy)
            ])

            fig.add_trace(
                go.Scatter3d(
                    x=cx,
                    y=cy,
                    z=cz,
                    mode="lines",
                    line=dict(
                        width=4,
                        dash="dot"
                    ),
                    name=f"Source {source['id']} Coverage",
                    hoverinfo="skip",
                )
            )

            source_z = terrain_height(
                source["x"],
                source["y"]
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[source["x"]],
                    y=[source["y"]],
                    z=[source_z + 5],
                    mode="markers+text",
                    marker=dict(
                        size=9,
                        symbol="diamond"
                    ),
                    text=[
                        f"📡 SOURCE {source['id']}<br>"
                        f"{source['freq']:g} MHz"
                    ],
                    textposition="top center",
                    name=f"Source {source['id']}",
                    hovertemplate=(
                        f"<b>{source['type']}</b>"
                        f"<br>Frequency: {source['freq']:g} MHz"
                        f"<br>Priority: {source['priority']}"
                        "<extra></extra>"
                    ),
                )
            )

        # Drone altitude line
        fig.add_trace(
            go.Scatter3d(
                x=[
                    st.session_state.x,
                    st.session_state.x
                ],
                y=[
                    st.session_state.y,
                    st.session_state.y
                ],
                z=[
                    drone_ground_z,
                    drone_z
                ],
                mode="lines",
                line=dict(width=5),
                name="Sensor altitude",
                hoverinfo="skip",
            )
        )

        # Drone
        fig.add_trace(
            go.Scatter3d(
                x=[st.session_state.x],
                y=[st.session_state.y],
                z=[drone_z],
                mode="markers+text",
                marker=dict(
                    size=13,
                    symbol="diamond"
                ),
                text=["🚁 SPECTRA-X DRONE"],
                textposition="top center",
                name="Drone",
                hovertemplate=(
                    "<b>SPECTRA-X DRONE</b>"
                    f"<br>Altitude: {altitude_value} m"
                    f"<br>East: {st.session_state.x:.1f} m"
                    f"<br>North: {st.session_state.y:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

        # RF sensor footprint
        theta = np.linspace(
            0,
            2 * np.pi,
            80
        )

        sensor_radius = 13

        sx = (
            st.session_state.x
            + sensor_radius * np.cos(theta)
        )

        sy = (
            st.session_state.y
            + sensor_radius * np.sin(theta)
        )

        sz = np.array([
            terrain_height(a, b) + 0.7
            for a, b in zip(sx, sy)
        ])

        fig.add_trace(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line=dict(
                    width=5,
                    dash="dot"
                ),
                name="RF Sensor Coverage",
                hoverinfo="skip",
            )
        )

        # Flight path
        if len(st.session_state.flight_path) > 1:

            path_x = [
                p[0]
                for p in st.session_state.flight_path
            ]

            path_y = [
                p[1]
                for p in st.session_state.flight_path
            ]

            path_z = [
                terrain_height(a, b) + 0.8
                for a, b in st.session_state.flight_path
            ]

            fig.add_trace(
                go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode="lines",
                    line=dict(width=4),
                    name="Drone flight path",
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            height=560,
            margin=dict(
                l=0,
                r=0,
                t=0,
                b=0
            ),
            paper_bgcolor="#050b13",
            plot_bgcolor="#050b13",
            scene=dict(
                bgcolor="#050b13",
                xaxis=dict(
                    title="EAST",
                    gridcolor="#24485e",
                    zerolinecolor="#24485e",
                ),
                yaxis=dict(
                    title="NORTH",
                    gridcolor="#24485e",
                    zerolinecolor="#24485e",
                ),
                zaxis=dict(
                    title="ALTITUDE",
                    gridcolor="#24485e",
                    zerolinecolor="#24485e",
                ),
                camera=dict(
                    eye=dict(
                        x=1.45,
                        y=1.45,
                        z=1.0
                    )
                ),
                aspectmode="manual",
                aspectratio=dict(
                    x=1.25,
                    y=1.25,
                    z=0.55
                ),
            ),
            legend=dict(
                bgcolor="rgba(3,8,14,.90)",
                font=dict(
                    color="white",
                    size=8
                ),
                x=0.72,
                y=0.98,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                "displaylogo": False,
                "scrollZoom": True,
                "responsive": True,
            },
            key="SPECTRA_X_SINGLE_3D_MAP",
        )

    # ========================================================
    # RIGHT — LIVE SYSTEM + ML
    # ========================================================

    with right:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">SYSTEM STATUS</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="status-good">✓ RF SENSOR ONLINE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="{"status-good" if MODEL_OK else "status-info"}">'
            f'{"✓ ML MODEL LOADED" if MODEL_OK else "⚠ ML MODEL OFFLINE"}'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="status-good">✓ SCAN ENGINE READY</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="status-info">● 4 RF SOURCES ACTIVE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">CURRENT RF OBSERVATION</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="event-row"><b>Band:</b> {obs["band"]}</div>
<div class="event-row"><b>Frequency:</b> {obs["frequency"]:.1f} MHz</div>
<div class="event-row"><b>Transmission:</b> {"ACTIVE" if obs["transmission"] else "IDLE"}</div>
<div class="event-row"><b>Power:</b> {obs["power"]:.1f} dBm</div>
<div class="event-row"><b>SNR:</b> {obs["snr"]:.1f} dB</div>
<div class="event-row"><b>Scan:</b> {scan_mode}</div>
""",
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">ML INFERENCE</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
<div class="metric-label">PREDICTED SIGNAL CLASS</div>
<div class="metric-big">{predicted}</div>

<br>

<div class="metric-label">CONFIDENCE</div>
<div class="metric-big">{confidence * 100:.1f}%</div>
""",
            unsafe_allow_html=True
        )

        if confidence > 0:
            st.progress(
                min(
                    1.0,
                    max(0.0, confidence)
                )
            )

        if top_predictions:

            st.markdown(
                '<div class="metric-label">TOP PREDICTIONS</div>',
                unsafe_allow_html=True
            )

            for cls, probability in top_predictions[:3]:

                st.markdown(
                    f"""
<div class="event-row">
    {cls} — {probability * 100:.1f}%
</div>
""",
                    unsafe_allow_html=True
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ========================================================
    # COMMUNICATION ANALYSIS
    # ========================================================

    comm_left, comm_right = st.columns(
        [1.0, 1.35],
        gap="small"
    )

    with comm_left:
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">📡 COMMUNICATION ANALYSIS</div>',
            unsafe_allow_html=True
        )

        nearest_source = min(
            SOURCES,
            key=lambda src:
            (src["x"] - st.session_state.x) ** 2
            + (src["y"] - st.session_state.y) ** 2
        )

        modulation_map = {
            "FM Radio": "FM",
            "WiFi": "QPSK / OFDM",
            "RF Link": "QPSK",
            "Cellular": "QAM / OFDM",
        }

        modulation = modulation_map.get(
            nearest_source["type"],
            "SIMULATED"
        )

        st.markdown(
            f"""
<div class="event-row"><b>Signal:</b> {nearest_source["type"]}</div>
<div class="event-row"><b>Frequency:</b> {obs["frequency"]:.1f} MHz</div>
<div class="event-row"><b>Power:</b> {obs["power"]:.1f} dBm</div>
<div class="event-row"><b>SNR:</b> {obs["snr"]:.1f} dB</div>
<div class="event-row"><b>Modulation:</b> {modulation}</div>
<div class="event-row"><b>RF Frame:</b> {"ACTIVE" if obs["transmission"] else "IDLE"}</div><div class="event-row"><b>Demo Decode:</b> AUTHORIZED TEST PAYLOAD</div>
""",
            unsafe_allow_html=True
        )

        if st.button(
            "🔓 ANALYZE & DECODE TEST SIGNAL",
            use_container_width=True,
            key="decode_authorized_signal",
        ):
            result = analyze_authorized_test_signal(obs)

            st.session_state.decoded_message = result["message"]
            st.session_state.decode_status = result["decode_status"]

            add_log(
                f"COMMUNICATION ANALYSIS COMPLETE — "
                f"AUTHORIZED TEST PAYLOAD FROM SOURCE {result['source_id']}"
            )

    with comm_right:
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">💬 DECODED MESSAGE</div>',
            unsafe_allow_html=True
        )

        if st.session_state.decoded_message:
            st.markdown(
                '<div class="decode-ok">✓ DECODE COMPLETE • TEST PAYLOAD</div>'
                '<div class="communication-message">'
                + st.session_state.decoded_message
                + '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="decode-wait">READY FOR OPERATOR ANALYSIS</div>'
                '<div class="communication-message">'
                'Press ANALYZE & DECODE TEST SIGNAL'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    # ========================================================
    # LOWER LIVE ANALYTICS
    # ========================================================

    spectrum_col, waterfall_col, log_col = st.columns(
        [1, 1, 1.25],
        gap="small"
    )

    rng = np.random.default_rng(
        1000 + st.session_state.slot
    )

    freq_axis = np.linspace(
        30,
        6000,
        420
    )

    spectrum = (
        -87
        + rng.normal(
            0,
            1.5,
            len(freq_axis)
        )
    )

    # Original prototype RF pattern
    for peak, strength, width in [
        (88.5, 38, 18),
        (900, 34, 23),
        (1300, 40, 25),
        (2400, 48, 30),
    ]:

        spectrum += (
            strength
            * np.exp(
                -(
                    (freq_axis - peak)
                    / width
                ) ** 2
            )
        )

    if obs["transmission"]:

        spectrum += (
            35
            * np.exp(
                -(
                    (freq_axis - obs["frequency"])
                    / 24
                ) ** 2
            )
        )

    with spectrum_col:

        st.markdown(
            '<div class="panel-title">📈 LIVE RF SPECTRUM</div>',
            unsafe_allow_html=True
        )

        spectrum_fig = go.Figure(
            go.Scatter(
                x=freq_axis,
                y=spectrum,
                mode="lines",
                hovertemplate=(
                    "%{x:.1f} MHz"
                    "<br>%{y:.1f} dB"
                    "<extra></extra>"
                ),
            )
        )

        spectrum_fig.update_layout(
            height=160,
            margin=dict(
                l=35,
                r=5,
                t=2,
                b=25
            ),
            paper_bgcolor="#07111d",
            plot_bgcolor="#07111d",
            font=dict(
                color="#b8cbd6",
                size=8
            ),
            xaxis_title="MHz",
            yaxis_title="dB",
        )

        st.plotly_chart(
            spectrum_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key="SPECTRA_X_LIVE_SPECTRUM",
        )

    with waterfall_col:

        st.markdown(
            '<div class="panel-title">🌊 RF WATERFALL</div>',
            unsafe_allow_html=True
        )

        if st.session_state.waterfall is None:

            waterfall = rng.normal(
                0,
                0.7,
                (38, 150)
            )

        else:

            waterfall = np.vstack([
                st.session_state.waterfall[1:],
                rng.normal(
                    0,
                    0.7,
                    (1, 150)
                )
            ])

        for peak in [
            88.5,
            900,
            1300,
            2400,
        ]:

            index = int(
                (peak / 6000) * 149
            )

            waterfall[-8:,
                       max(0, index - 2):index + 3] += 4

        if obs["transmission"]:

            index = int(
                (obs["frequency"] / 6000) * 149
            )

            waterfall[-8:,
                       max(0, index - 2):index + 3] += 5

        st.session_state.waterfall = waterfall

        waterfall_fig = go.Figure(
            go.Heatmap(
                z=waterfall,
                colorscale="Turbo",
                showscale=False,
            )
        )

        waterfall_fig.update_layout(
            height=160,
            margin=dict(
                l=5,
                r=5,
                t=2,
                b=25
            ),
            paper_bgcolor="#07111d",
            plot_bgcolor="#07111d",
        )

        st.plotly_chart(
            waterfall_fig,
            use_container_width=True,
            config={"displaylogo": False},
            key="SPECTRA_X_RF_WATERFALL",
        )

    with log_col:

        st.markdown(
            '<div class="panel-title">📋 MISSION LOG</div>',
            unsafe_allow_html=True
        )

        log_html = '<div class="log-box">'

        for item in st.session_state.logs[-8:]:
            log_html += (
                item.replace("<", "&lt;")
                .replace(">", "&gt;")
                + "<br>"
            )

        log_html += "</div>"

        st.markdown(
            log_html,
            unsafe_allow_html=True
        )


# ============================================================
# RUN ONLY ONE DASHBOARD INSTANCE
# ============================================================

dashboard()

st.markdown(
    '<div class="footer">'
    'SPECTRA-X • AI RF RECONNAISSANCE • DEMONSTRATION PROTOTYPE'
    '</div>',
    unsafe_allow_html=True
)
