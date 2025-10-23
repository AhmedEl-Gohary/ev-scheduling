#!/usr/bin/env python3

"""
Flask Web Application for EV Charging Schedule Visualization
Run with: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import secrets
from src.algorithms.sa import simulated_annealing
from src.algorithms.greedy import greedy_schedule
from eval import *

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

algorithm_states = {}

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/start', methods=['POST'])
def start_algorithm():
    """Start the SA algorithm with given parameters"""
    data = request.json

    # Normalize parameters
    params = data.copy()
    for ev in params["evs"]:
        if "arrival_slot" not in ev:
            ev["arrival_slot"] = ev.get("arrival", 0)
        if "departure_slot" not in ev:
            ev["departure_slot"] = ev.get("departure", params["time_slots"] - 1)

    if "spots_per_line" not in params:
        if "lines" in params:
            if isinstance(params["lines"], dict):
                params["spots_per_line"] = [int(v) for v in params["lines"].values()]
            else:
                params["spots_per_line"] = params["lines"]

    if "level_powers" not in params:
        if "charging_levels" in params:
            params["level_powers"] = [float(v) for k, v in sorted(params["charging_levels"].items())]

    if "delta_t" not in params:
        params["delta_t"] = 1.0

    # Generate initial schedule
    X0, B0 = greedy_schedule(params)

    # Run SA with state capture
    X_best, B_best, states = simulated_annealing(
        X0, B0, params,
        T0=params.get('T0', 10.0),
        Tf=params.get('Tf', 0.001),
        imax=params.get('imax', 50),
        nT=params.get('nT', 30),
        rng_seed=params.get('seed', 42)
    )

    # Store in session
    session_id = secrets.token_hex(8)
    algorithm_states[session_id] = {
        'params': params,
        'states': states,
        'current_iteration': 0
    }

    return jsonify({
        'session_id': session_id,
        'total_iterations': len(states),
        'initial_state': states[0],
        'params': params  # --- MODIFICATION: Send params back to client
    })


@app.route('/api/state/<session_id>/<int:iteration>')
def get_state(session_id, iteration):
    """Get state at specific iteration"""
    if session_id not in algorithm_states:
        return jsonify({'error': 'Session not found'}), 404

    data = algorithm_states[session_id]
    states = data['states']

    if iteration < 0 or iteration >= len(states):
        return jsonify({'error': 'Invalid iteration'}), 400

    state = states[iteration]
    params = data['params']

    # Convert state to visualization format
    viz_data = convert_state_to_viz(state, params)

    return jsonify({
        'iteration': iteration,
        'total_iterations': len(states),
        'state': viz_data,
        'params': params
    })


def convert_state_to_viz(state, params):
    """Convert numpy arrays to visualization-friendly format"""
    X = np.array(state['X'])
    B = np.array(state['B'])

    J, L, Smax, T = X.shape
    S_i = params["spots_per_line"]

    # Extract vehicle assignments
    vehicles = []
    for j in range(J):
        ev = params["evs"][j]
        # Find where vehicle is assigned
        assignments = []
        for t in range(T):
            for i in range(L):
                for s in range(S_i[i]):
                    if X[j, i, s, t] == 1:
                        level_idx = np.where(B[j, :, t] == 1)[0]
                        level = int(level_idx[0]) if level_idx.size > 0 else 0
                        assignments.append({
                            'slot': t,
                            'line': i,
                            'spot': s,
                            'level': level,
                            'power': params["level_powers"][level]
                        })

        vehicles.append({
            'id': j,
            'name': ev.get('id', f'EV{j + 1}'),
            'arrival': ev['arrival_slot'],
            'departure': ev['departure_slot'],
            'energy_required': ev['energy_required'],
            'assignments': assignments
        })

    # Compute power profile
    power_profile = compute_total_power_profile(B, params["level_powers"]).tolist()

    return {
        'vehicles': vehicles,
        'power_profile': power_profile,
        'metrics': {
            'objective': state['objective'],
            'best_objective': state['best_objective'],
            'tardiness': state['tardiness'],
            'peak_power': state['peak_power'],
            'temperature': state['temperature']
        }
    }


# ============================================================================
# HTML TEMPLATE
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV Charging Station Optimizer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            padding: 15px;
            color: #2c3e50;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        .header {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }

        .header h1 {
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 1.8em;
        }

        .header p {
            color: #7f8c8d;
            font-size: 0.95em;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 15px;
            height: calc(100vh - 140px);
        }

        .left-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-y: auto;
        }

        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .viz-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            overflow: auto;
        }

        .metrics-panel {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }

        .form-group {
            margin-bottom: 15px;
        }

        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #34495e;
            font-size: 0.9em;
        }

        .form-group input, .form-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #d divde;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #3498db;
        }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .ev-list {
            border: 1px solid #ecf0f1;
            border-radius: 4px;
            padding: 10px;
            max-height: 200px;
            overflow-y: auto;
        }

        .ev-item {
            background: #ecf0f1;
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 0.85em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .ev-item button {
            background: #e74c3c;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8em;
        }

        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            transition: background 0.2s;
        }

        .btn:hover:not(:disabled) {
            background: #2980b9;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-small {
            padding: 6px 12px;
            font-size: 0.85em;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 15px;
        }

        .metric-box {
            background: #ecf0f1;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }

        .metric-box .label {
            font-size: 0.75em;
            color: #7f8c8d;
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .metric-box .value {
            font-size: 1.4em;
            font-weight: bold;
            color: #2c3e50;
        }

        .final-results {
            background: #d5f4e6;
            border: 2px solid #27ae60;
            border-radius: 6px;
            padding: 15px;
            margin-top: 15px;
        }

        .final-results h3 {
            color: #27ae60;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .final-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .final-metric {
            background: white;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }

        .controls {
            display: flex;
            gap: 8px;
            margin-bottom: 12px;
        }

        .controls button {
            flex: 1;
            padding: 8px;
            background: #34495e;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }

        .controls button:hover:not(:disabled) {
            background: #2c3e50;
        }

        .controls button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .slider-container {
            margin-bottom: 12px;
        }

        .slider-container input[type="range"] {
            width: 100%;
            height: 4px;
            border-radius: 2px;
            background: #d divde;
            outline: none;
        }

        .slider-label {
            display: flex;
            justify-space-between;
            margin-top: 4px;
            font-size: 0.75em;
            color: #7f8c8d;
        }

        .timeline-view {
            margin-top: 20px;
        }

        .station-grid {
            display: grid;
            gap: 15px;
        }

        .charging-line {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 12px;
        }

        .line-header {
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
            font-size: 0.95em;
        }

        .timeline-container {
            display: grid;
            grid-template-columns: 60px 1fr;
            gap: 8px;
        }

        .spot-labels {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .spot-label {
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
            font-weight: 600;
            color: #7f8c8d;
            background: #ecf0f1;
            border-radius: 4px;
        }

        .timeline-grid {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .spot-timeline {
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: 1fr;
            gap: 2px;
            height: 40px;
        }

        .time-cell {
            background: #fff;
            border: 1px solid #e9ecef;
            border-radius: 3px;
            position: relative;
            overflow: hidden;
            /* Note: transition: all is removed to allow animation to work */
        }

        /* --- MODIFICATION START: New Fade Animations --- */

        @keyframes fadeInAnim {
            from { opacity: 0; }
            to   { opacity: 1; }
        }

        @keyframes fadeOutAnim {
            from { opacity: 1; }
            to   { opacity: 0; }
        }

        .cell-fade-in {
            animation: fadeInAnim 0.4s ease-out;
        }

        .cell-fade-out {
            animation: fadeOutAnim 0.4s ease-in;
        }

        /* --- MODIFICATION END --- */


        .vehicle-in-cell {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            color: white;
            font-weight: bold;
            position: relative;
        }

        .vehicle-in-cell::before {
            content: '🚗';
            font-size: 0.9em;
            margin-right: 2px;
        }

        .level-badge {
            position: absolute;
            top: 2px;
            right: 2px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 0.65em;
            padding: 1px 4px;
            border-radius: 2px;
        }

        .time-axis {
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: 1fr;
            gap: 2px;
            margin-top: 5px;
            margin-left: 68px;
        }

        .time-label {
            text-align: center;
            font-size: 0.7em;
            color: #7f8c8d;
        }

        .power-chart-container {
            margin-top: 20px;
            background: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
        }

        .chart-header {
            font-weight: 600;
            margin-bottom: 10px;
            color: #2c3e50;
            font-size: 0.95em;
        }

        canvas {
            max-width: 100%;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }

        .vehicle-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }

        .legend-color {
            width: 30px;
            height: 20px;
            border-radius: 3px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }

        .section-toggle {
            background: #ecf0f1;
            padding: 8px 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 600;
            color: #2c3e50;
            font-size: 0.9em;
        }

        .section-toggle:hover {
            background: #d divde;
        }

        .section-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .section-content.open {
            max-height: 1000px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ EV Charging Station Optimizer</h1>
            <p>Simulated Annealing Algorithm Visualization</p>
        </div>

        <div class="main-grid">
            <div class="left-panel">
                <h2>Configuration</h2>

                <div class="section-toggle" onclick="toggleSection('stationConfig')">
                    🏢 Station Setup ▼
                </div>
                <div class="section-content open" id="stationConfig">
                    <div class="form-group">
                        <label>Time Slots</label>
                        <input type="number" id="timeSlots" value="24" min="1">
                    </div>
                    <div class="form-group">
                        <label>Power Limit (kW)</label>
                        <input type="number" id="powerMax" value="50" step="1">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Charging Lines</label>
                            <input type="number" id="numLines" value="2" min="1" max="5">
                        </div>
                        <div class="form-group">
                            <label>Spots per Line</label>
                            <input type="number" id="spotsPerLine" value="3" min="1">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Charging Levels (kW, comma-separated)</label>
                        <input type="text" id="chargingLevels" value="3, 7, 11">
                    </div>
                </div>

                <div class="section-toggle" onclick="toggleSection('evConfig')">
                    🚗 Electric Vehicles ▼
                </div>
                <div class="section-content open" id="evConfig">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Arrival Slot</label>
                            <input type="number" id="evArrival" value="0" min="0">
                        </div>
                        <div class="form-group">
                            <label>Departure Slot</label>
                            <input type="number" id="evDeparture" value="8" min="0">
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Energy Required (kWh)</label>
                        <input type="number" id="evEnergy" value="18" step="0.1">
                    </div>
                    <button class="btn btn-small" onclick="addEV()">+ Add Vehicle</button>
                    <div class="ev-list" id="evList"></div>
                </div>

                <div class="section-toggle" onclick="toggleSection('algoConfig')">
                    ⚙️ Algorithm Parameters ▼
                </div>
                <div class="section-content" id="algoConfig">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Initial Temp (T0)</label>
                            <input type="number" id="t0" value="10" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Final Temp (Tf)</label>
                            <input type="number" id="tf" value="0.001" step="0.001">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Iterations</label>
                            <input type="number" id="imax" value="30" min="1">
                        </div>
                        <div class="form-group">
                            <label>Steps per Temp</label>
                            <input type="number" id="nT" value="20" min="1">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Weight Tardiness</label>
                            <input type="number" id="wTardiness" value="1.0" step="0.1">
                        </div>
                        <div class="form-group">
                            <label>Weight Peak</label>
                            <input type="number" id="wPeak" value="0.5" step="0.1">
                        </div>
                    </div>
                </div>

                <button class="btn" onclick="startOptimization()" style="margin-top: 15px;">
                    🚀 Start Optimization
                </button>
            </div>

            <div class="right-panel">
                <div class="metrics-panel" id="metricsPanel" style="display: none;">
                    <div class="metrics-grid">
                        <div class="metric-box">
                            <div class="label">Iteration</div>
                            <div class="value" id="metricIteration">0</div>
                        </div>
                        <div class="metric-box">
                            <div class="label">Temperature</div>
                            <div class="value" id="metricTemp">0</div>
                        </div>
                        <div class="metric-box">
                            <div class="label">Objective</div>
                            <div class="value" id="metricObj">0</div>
                        </div>
                        <div class="metric-box">
                            <div class="label">Tardiness</div>
                            <div class="value" id="metricTard">0</div>
                        </div>
                        <div class="metric-box">
                            <div class="label">Peak Power</div>
                            <div class="value" id="metricPeak">0</div>
                        </div>
                        <div class="metric-box">
                            <div class="label">Best Obj</div>
                            <div class="value" id="metricBest">0</div>
                        </div>
                    </div>

                    <div class="controls">
                        <button onclick="previousIteration()">⏮ Prev</button>
                        <button onclick="playPause()" id="playBtn">▶ Play</button>
                        <button onclick="nextIteration()">Next ⏭</button>
                    </div>
                    <div class="slider-container">
                        <input type="range" id="iterationSlider" min="0" max="0" value="0" 
                               oninput="seekIteration(this.value)">
                        <div class="slider-label">
                            <span>Iteration 0</span>
                            <span id="sliderMax">0</span>
                        </div>
                    </div>

                    <div id="finalResults" class="final-results" style="display: none;">
                        <h3>🎯 Final Results</h3>
                        <div class="final-grid">
                            <div class="final-metric">
                                <div class="label">Final Tardiness</div>
                                <div class="value" id="finalTardiness">0</div>
                            </div>
                            <div class="final-metric">
                                <div class="label">Final Peak Power</div>
                                <div class="value" id="finalPeak">0 kW</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="viz-panel">
                    <div id="visualization">
                        <div class="loading">Configure parameters and click "Start Optimization"</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let totalIterations = 0;
        let currentIteration = 0;
        let isPlaying = false;
        let playInterval = null;
        let currentParams = null;
        let evs = [];
        let previousVizState = null; 

        const VEHICLE_COLORS = [
            '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
            '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b'
        ];

        function toggleSection(id) {
            const section = document.getElementById(id);
            section.classList.toggle('open');
        }

        function addEV() {
            const arrival = parseInt(document.getElementById('evArrival').value);
            const departure = parseInt(document.getElementById('evDeparture').value);
            const energy = parseFloat(document.getElementById('evEnergy').value);

            evs.push({
                id: `EV${evs.length + 1}`,
                arrival_slot: arrival,
                departure_slot: departure,
                energy_required: energy
            });

            updateEVList();
        }

        function removeEV(index) {
            evs.splice(index, 1);
            updateEVList();
        }

        function updateEVList() {
            const list = document.getElementById('evList');
            if (evs.length === 0) {
                list.innerHTML = '<div style="text-align: center; color: #7f8c8d; padding: 20px;">No vehicles added yet</div>';
                return;
            }

            list.innerHTML = evs.map((ev, i) => `
                <div class="ev-item">
                    <span>${ev.id}: ${ev.arrival_slot}→${ev.departure_slot} (${ev.energy_required}kWh)</span>
                    <button onclick="removeEV(${i})">×</button>
                </div>
            `).join('');
        }

        function loadDefaultEVs() {
            evs = [
                {id: "EV1", arrival_slot: 0, departure_slot: 8, energy_required: 18.0},
                {id: "EV2", arrival_slot: 1, departure_slot: 10, energy_required: 24.0},
                {id: "EV3", arrival_slot: 2, departure_slot: 14, energy_required: 30.0},
                {id: "EV4", arrival_slot: 5, departure_slot: 20, energy_required: 60.0}
            ];
            updateEVList();
        }

        async function startOptimization() {
            if (evs.length === 0) {
                alert('Please add at least one vehicle');
                return;
            }

            try {
                const timeSlots = parseInt(document.getElementById('timeSlots').value);
                const numLines = parseInt(document.getElementById('numLines').value);
                const spotsPerLine = parseInt(document.getElementById('spotsPerLine').value);
                const chargingLevels = document.getElementById('chargingLevels').value
                    .split(',').map(x => parseFloat(x.trim()));

                const input = {
                    time_slots: timeSlots,
                    delta_t: 1.0,
                    spots_per_line: Array(numLines).fill(spotsPerLine),
                    level_powers: chargingLevels,
                    P_max: parseFloat(document.getElementById('powerMax').value),
                    T0: parseFloat(document.getElementById('t0').value),
                    Tf: parseFloat(document.getElementById('tf').value),
                    imax: parseInt(document.getElementById('imax').value),
                    nT: parseInt(document.getElementById('nT').value),
                    w_tardiness: parseFloat(document.getElementById('wTardiness').value),
                    w_peak: parseFloat(document.getElementById('wPeak').value),
                    evs: evs
                };

                document.getElementById('visualization').innerHTML = '<div class="loading">Running optimization...</div>';

                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(input)
                });

                const data = await response.json();
                sessionId = data.session_id;
                totalIterations = data.total_iterations;
                currentIteration = 0;
                previousVizState = null; 
                currentParams = data.params; // --- MODIFICATION: Store params

                document.getElementById('iterationSlider').max = totalIterations - 1;
                document.getElementById('sliderMax').textContent = `Iteration ${totalIterations - 1}`;
                document.getElementById('metricsPanel').style.display = 'block';
                document.getElementById('finalResults').style.display = 'none';

                // --- MODIFICATION: Render skeleton, then load data
                renderVisualizationSkeleton(currentParams);
                await loadIteration(0);
                // ---

            } catch (error) {
                alert('Error: ' + error.message);
            }
        }

        async function loadIteration(iteration) {
            if (!sessionId) return;

            try {
                const response = await fetch(`/api/state/${sessionId}/${iteration}`);
                const data = await response.json();

                currentIteration = iteration;
                // currentParams is already set
                document.getElementById('iterationSlider').value = iteration;

                updateMetrics(data.state.metrics, iteration, data.total_iterations);

                // --- MODIFICATION: Call new update function
                updateVisualization(data.state, data.params, iteration === 0);

                previousVizState = data.state; // Store the just-rendered state

                // Show final results if at last iteration
                if (iteration === data.total_iterations - 1) {
                    showFinalResults(data.state.metrics);
                } else {
                    document.getElementById('finalResults').style.display = 'none';
                }
            } catch (error) {
                console.error('Error loading iteration:', error);
            }
        }

        function updateMetrics(metrics, iteration, total) {
            document.getElementById('metricIteration').textContent = `${iteration}/${total-1}`;
            document.getElementById('metricTemp').textContent = metrics.temperature.toFixed(3);
            document.getElementById('metricObj').textContent = metrics.objective.toFixed(1);
            document.getElementById('metricBest').textContent = metrics.best_objective.toFixed(1);
            document.getElementById('metricTard').textContent = metrics.tardiness.toFixed(1);
            document.getElementById('metricPeak').textContent = metrics.peak_power.toFixed(1);
        }

        function showFinalResults(metrics) {
            document.getElementById('finalResults').style.display = 'block';
            document.getElementById('finalTardiness').textContent = metrics.tardiness.toFixed(1);
            document.getElementById('finalPeak').textContent = metrics.peak_power.toFixed(1) + ' kW';
        }

        // --- MODIFICATION: New helper function ---
        function getVehicleCellHTML(vehicleInfo) {
            return `
                <div class="vehicle-in-cell">
                    ${vehicleInfo.vehicleIdx + 1}
                    <div class="level-badge">L${vehicleInfo.level + 1}</div>
                </div>
            `;
        }

        // --- MODIFICATION: This function now *only* builds the static grid skeleton ---
        function renderVisualizationSkeleton(params) {
            const viz = document.getElementById('visualization');
            const numLines = params.spots_per_line.length;
            const timeSlots = params.time_slots;

            // Create vehicle legend container
            let html = '<div class="vehicle-legend" id="vehicleLegend"></div>';

            html += '<div class="timeline-view"><div class="station-grid">';

            // Create charging lines with timeline
            for (let lineIdx = 0; lineIdx < numLines; lineIdx++) {
                const numSpots = params.spots_per_line[lineIdx];
                html += `
                    <div class="charging-line">
                        <div class="line-header">Charging Line ${lineIdx + 1}</div>
                        <div class="timeline-container">
                            <div class="spot-labels">
                `;

                for (let spotIdx = 0; spotIdx < numSpots; spotIdx++) {
                    html += `<div class="spot-label">Spot ${spotIdx + 1}</div>`;
                }

                html += `
                            </div>
                            <div class="timeline-grid">
                `;

                for (let spotIdx = 0; spotIdx < numSpots; spotIdx++) {
                    html += '<div class="spot-timeline">';
                    for (let t = 0; t < timeSlots; t++) {
                        // --- MODIFICATION: Render empty cell with unique ID ---
                        html += `<div class="time-cell" id="cell-${lineIdx}-${spotIdx}-${t}"></div>`;
                    }
                    html += '</div>';
                }

                html += `
                            </div>
                        </div>
                    </div>
                `;
            }

            // Add time axis
            html += '<div class="time-axis">';
            for (let t = 0; t < timeSlots; t++) {
                html += `<div class="time-label">${t}</div>`;
            }
            html += '</div>';

            html += '</div></div>';

            // Add power chart
            html += `
                <div class="power-chart-container">
                    <div class="chart-header">Power Profile Over Time</div>
                    <canvas id="powerChart" width="800" height="200"></canvas>
                </div>
            `;

            viz.innerHTML = html;
        }

        // --- MODIFICATION: New function to update the grid based on state ---
        function updateVisualization(state, params, isInitialRender = false) {
            const numLines = params.spots_per_line.length;
            const timeSlots = params.time_slots;

            // 1. Update Vehicle Legend
            const legend = document.getElementById('vehicleLegend');
            let legendHtml = '';
            state.vehicles.forEach((vehicle, idx) => {
                const color = VEHICLE_COLORS[idx % VEHICLE_COLORS.length];
                legendHtml += `
                    <div class="legend-item">
                        <div class="legend-color" style="background: ${color}">${idx + 1}</div>
                        <span>${vehicle.name} (${vehicle.arrival}→${vehicle.departure}, ${vehicle.energy_required}kWh)</span>
                    </div>
                `;
            });
            legend.innerHTML = legendHtml;

            // 2. Update Grid Cells
            for (let lineIdx = 0; lineIdx < numLines; lineIdx++) {
                const numSpots = params.spots_per_line[lineIdx];
                for (let spotIdx = 0; spotIdx < numSpots; spotIdx++) {
                    for (let t = 0; t < timeSlots; t++) {
                        const cell = document.getElementById(`cell-${lineIdx}-${spotIdx}-${t}`);

                        const vehicleInfo = findVehicleAtSlot(state.vehicles, lineIdx, spotIdx, t);
                        const prevVehicleInfo = previousVizState ? findVehicleAtSlot(previousVizState.vehicles, lineIdx, spotIdx, t) : null;

                        const newContent = vehicleInfo ? getVehicleCellHTML(vehicleInfo) : '';
                        const oldContent = prevVehicleInfo ? getVehicleCellHTML(prevVehicleInfo) : '';

                        if (newContent === oldContent) continue; // No change

                        const newBg = vehicleInfo ? VEHICLE_COLORS[vehicleInfo.vehicleIdx % VEHICLE_COLORS.length] : '#fff';
                        const oldBg = prevVehicleInfo ? VEHICLE_COLORS[prevVehicleInfo.vehicleIdx % VEHICLE_COLORS.length] : '#fff';

                        cell.classList.remove('cell-fade-in', 'cell-fade-out');

                        if (newContent && !oldContent) { // Car Added (Fade In)
                            cell.innerHTML = newContent;
                            cell.style.background = newBg;
                            if (!isInitialRender) {
                                cell.classList.add('cell-fade-in');
                                setTimeout(() => cell.classList.remove('cell-fade-in'), 400);
                            }
                        } else if (!newContent && oldContent) { // Car Removed (Fade Out)
                            cell.innerHTML = oldContent; // Keep old content for animation
                            cell.style.background = oldBg;
                            if (!isInitialRender) {
                                cell.classList.add('cell-fade-out');
                                setTimeout(() => {
                                    cell.innerHTML = '';
                                    cell.style.background = '#fff';
                                    cell.classList.remove('cell-fade-out');
                                }, 400);
                            } else {
                                cell.innerHTML = '';
                                cell.style.background = '#fff';
                            }
                        } else if (newContent && oldContent) { // Car Changed (Fade In New)
                            cell.innerHTML = newContent;
                            cell.style.background = newBg;
                            if (!isInitialRender) {
                                cell.classList.add('cell-fade-in');
                                setTimeout(() => cell.classList.remove('cell-fade-in'), 400);
                            }
                        } else { // Both empty
                            cell.innerHTML = '';
                            cell.style.background = '#fff';
                        }
                    }
                }
            }

            // 3. Update Power Chart
            drawPowerChart(state.power_profile, params.P_max);
        }

        function findVehicleAtSlot(vehicles, lineIdx, spotIdx, timeSlot) {
            for (let vIdx = 0; vIdx < vehicles.length; vIdx++) {
                const vehicle = vehicles[vIdx];
                for (const assignment of vehicle.assignments) {
                    if (assignment.line === lineIdx && 
                        assignment.spot === spotIdx &&
                        assignment.slot === timeSlot) {
                        return {
                            vehicleIdx: vIdx,
                            level: assignment.level,
                            power: assignment.power
                        };
                    }
                }
            }
            return null;
        }

        function drawPowerChart(powerProfile, pMax) {
            const canvas = document.getElementById('powerChart');
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            if (powerProfile.length === 0) return;

            const maxPower = Math.max(...powerProfile, pMax || 0) * 1.1;
            const padding = 40;
            const chartWidth = width - 2 * padding;
            const chartHeight = height - 2 * padding;

            ctx.clearRect(0, 0, width, height);

            // Draw axes
            ctx.strokeStyle = '#d divde';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();

            // Draw P_max line
            if (pMax) {
                const yMax = height - padding - (pMax / maxPower) * chartHeight;
                ctx.strokeStyle = '#e74c3c';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(padding, yMax);
                ctx.lineTo(width - padding, yMax);
                ctx.stroke();
                ctx.setLineDash([]);

                ctx.fillStyle = '#e74c3c';
                ctx.font = '12px sans-serif';
                ctx.fillText(`Max: ${pMax.toFixed(1)} kW`, width - padding + 5, yMax + 4);
            }

            // Draw power profile
            ctx.strokeStyle = '#3498db';
            ctx.fillStyle = 'rgba(52, 152, 219, 0.2)';
            ctx.lineWidth = 2;

            ctx.beginPath();
            ctx.moveTo(padding, height - padding);

            for (let i = 0; i < powerProfile.length; i++) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                const y = height - padding - (powerProfile[i] / maxPower) * chartHeight;
                ctx.lineTo(x, y);
            }

            ctx.lineTo(width - padding, height - padding);
            ctx.closePath();
            ctx.fill();

            ctx.beginPath();
            for (let i = 0; i < powerProfile.length; i++) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                const y = height - padding - (powerProfile[i] / maxPower) * chartHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw Y-axis labels
            ctx.fillStyle = '#7f8c8d';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 4; i++) {
                const value = (maxPower / 4) * i;
                const y = height - padding - (i / 4) * chartHeight;
                ctx.fillText(value.toFixed(1), padding - 8, y + 4);
            }

            // Draw X-axis labels
            ctx.textAlign = 'center';
            const step = Math.ceil(powerProfile.length / 12);
            for (let i = 0; i < powerProfile.length; i += step) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                ctx.fillText(i.toString(), x, height - padding + 20);
            }

            // Labels
            ctx.textAlign = 'center';
            ctx.font = 'bold 12px sans-serif';
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Time Slot', width / 2, height - 5);

            ctx.save();
            ctx.translate(15, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Power (kW)', 0, 0);
            ctx.restore();
        }

        function nextIteration() {
            if (currentIteration < totalIterations - 1) {
                loadIteration(currentIteration + 1);
            }
        }

        function previousIteration() {
            if (currentIteration > 0) {
                loadIteration(currentIteration - 1);
            }
        }

        function seekIteration(value) {
            loadIteration(parseInt(value));
        }

        function playPause() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('playBtn');

            if (isPlaying) {
                btn.textContent = '⏸ Pause';
                playInterval = setInterval(() => {
                    if (currentIteration < totalIterations - 1) {
                        nextIteration();
                    } else {
                        playPause();
                    }
                }, 1000); 
            } else {
                btn.textContent = '▶ Play';
                if (playInterval) {
                    clearInterval(playInterval);
                    playInterval = null;
                }
            }
        }

        // Initialize with default EVs
        loadDefaultEVs();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting EV Charging Visualization Server...")
    print("Open your browser to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)