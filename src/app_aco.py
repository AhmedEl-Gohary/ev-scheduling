#!/usr/bin/env python3

"""
Flask Web Application for EV Charging Schedule Visualization - Ant Colony Optimization
Run with: python app_aco.py
Then open: http://localhost:5002
"""

from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS
import secrets
import json
import threading
import queue
import numpy as np
from src.algorithms.aco import ant_colony_optimization
from src.eval import compute_total_power_profile

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

algorithm_states = {}
progress_queues = {}


# ============================================================================
# PROGRESS CALLBACK FOR ACO
# ============================================================================

def create_progress_callback(session_id):
    """Create a callback function that sends progress updates"""

    def callback(iteration, current_fitness, best_fitness, avg_fitness, full_state=None):
        if session_id in progress_queues:
            progress_data = {
                'type': 'progress',
                'iteration': iteration,
                'current_fitness': float(current_fitness),
                'best_fitness': float(best_fitness),
                'avg_fitness': float(avg_fitness),
                'message': f"Iteration {iteration}: Best={best_fitness:.2f}"
            }

            # Update state storage for the API to fetch visuals
            if session_id in algorithm_states and full_state:
                algorithm_states[session_id]['states'].append(full_state)

            progress_queues[session_id].put(progress_data)

    return callback


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/start', methods=['POST'])
def start_algorithm():
    """Start the ACO algorithm with given parameters"""
    data = request.json

    # Normalize parameters (Same logic as app2.py)
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

    # Create session
    session_id = secrets.token_hex(8)
    progress_queues[session_id] = queue.Queue()

    # Initialize state storage immediately
    algorithm_states[session_id] = {
        'params': params,
        'states': [],
        'current_iteration': 0
    }

    # Run ACO in background thread
    def run_aco():
        try:
            progress_queues[session_id].put({
                'type': 'status',
                'message': 'Initializing Pheromones...'
            })

            # Call ACO Wrapper
            X_best, B_best, states = ant_colony_optimization(
                params,
                n_ants=params.get('n_ants', 20),
                n_iterations=params.get('n_iterations', 50),
                alpha=params.get('alpha', 1.0),
                beta=params.get('beta', 2.0),
                rho=params.get('rho', 0.1),
                Q=params.get('Q', 100.0),
                rng_seed=params.get('seed', 42),
                progress_callback=create_progress_callback(session_id)
            )

            # Note: states are already appended via callback, but we ensure consistency
            algorithm_states[session_id]['states'] = states

            progress_queues[session_id].put({
                'type': 'complete',
                'session_id': session_id,
                'total_iterations': len(states)
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            progress_queues[session_id].put({
                'type': 'error',
                'message': str(e)
            })

    thread = threading.Thread(target=run_aco, daemon=True)
    thread.start()

    return jsonify({
        'session_id': session_id,
        'status': 'started'
    })


@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """SSE endpoint for progress updates"""

    def generate():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return

        q = progress_queues[session_id]

        while True:
            try:
                data = q.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                if data['type'] in ['complete', 'error']:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/state/<session_id>/<int:iteration>')
def get_state(session_id, iteration):
    """Get state at specific iteration (generation)"""
    if session_id not in algorithm_states:
        return jsonify({'error': 'Session not found'}), 404

    data = algorithm_states[session_id]
    states = data['states']

    # Adjust 1-based iteration to 0-based index if needed, or handle direct indexing
    # Our history list is 0-indexed.
    idx = iteration - 1 if iteration > 0 else 0

    if idx < 0 or idx >= len(states):
        return jsonify({'error': 'Invalid iteration index'}), 400

    state = states[idx]
    params = data['params']

    viz_data = convert_state_to_viz(state, params)

    return jsonify({
        'iteration': iteration,
        'total_iterations': len(states),
        'state': viz_data,
        'params': params
    })


def convert_state_to_viz(state, params):
    """Convert numpy arrays to visualization-friendly format"""
    # ACO state stores lists, convert to numpy
    X = np.array(state['X'])
    B = np.array(state['B'])

    J, L, Smax, T = X.shape
    S_i = params["spots_per_line"]

    vehicles = []
    for j in range(J):
        ev = params["evs"][j]
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

    power_profile = compute_total_power_profile(B, params["level_powers"]).tolist()

    return {
        'vehicles': vehicles,
        'power_profile': power_profile,
        'metrics': {
            'objective': state.get('objective', 0),
            'best_objective': state.get('best_fitness', 0),
            'avg_fitness': state.get('avg_fitness', 0),
            'tardiness': state.get('tardiness', 0),
            'peak_power': state.get('peak_power', 0),
            'iteration': state.get('iteration', 0)
        }
    }


# ============================================================================
# HTML TEMPLATE (Updated with File Upload)
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV Charging - Ant Colony Optimization</title>
    <style>
        /* Shared Styles */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f7fa; padding: 15px; color: #2c3e50; }
        .container { max-width: 1800px; margin: 0 auto; }
        .header { background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #8e44ad; }
        .header h1 { margin-bottom: 5px; font-size: 1.8em; }
        .aco-badge { background: #8e44ad; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; margin-left: 10px; }
        .main-grid { display: grid; grid-template-columns: 350px 1fr; gap: 15px; height: calc(100vh - 140px); }
        .left-panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow-y: auto; }
        .right-panel { display: flex; flex-direction: column; gap: 15px; }
        .viz-panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; overflow: auto; }
        .metrics-panel { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none; }
        h2 { border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; margin-bottom: 15px; font-size: 1.2em; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 600; font-size: 0.9em; }
        .form-group input, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .btn { background: #8e44ad; color: white; border: none; padding: 10px; border-radius: 4px; width: 100%; cursor: pointer; font-weight: 600; }
        .btn:hover:not(:disabled) { background: #732d91; }
        .btn:disabled { opacity: 0.5; }

        /* Upload Area */
        .upload-area {
            border: 2px dashed #bdc3c7;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .upload-area:hover { border-color: #8e44ad; background: #fdf2ff; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 1.5em; color: #7f8c8d; margin-bottom: 5px; }
        .file-name { font-size: 0.8em; color: #2c3e50; font-weight: bold; margin-top: 5px; display: none; }

        /* Progress Bar */
        .progress-container { background: #ecf0f1; border-radius: 8px; padding: 20px; margin-top: 20px; display: none; }
        .progress-container.active { display: block; }
        .progress-bar-container { background: #ddd; border-radius: 10px; height: 24px; overflow: hidden; margin: 10px 0; }
        .progress-bar { background: linear-gradient(90deg, #8e44ad, #9b59b6); height: 100%; width: 0%; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.8em; transition: width 0.3s; }

        /* Metrics */
        .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .metric-box { background: #ecf0f1; padding: 10px; border-radius: 6px; text-align: center; }
        .metric-box .label { font-size: 0.7em; color: #7f8c8d; text-transform: uppercase; }
        .metric-box .value { font-size: 1.2em; font-weight: bold; color: #2c3e50; }

        /* Visualization Grid */
        .charging-line { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
        .timeline-container { display: grid; grid-template-columns: 60px 1fr; gap: 5px; }
        .spot-timeline { display: grid; grid-auto-flow: column; gap: 2px; height: 35px; margin-bottom: 2px; }
        .time-cell { background: white; border: 1px solid #e9ecef; border-radius: 2px; display: flex; align-items: center; justify-content: center; font-size: 0.9em; color: white; font-weight: bold; transition: background 0.3s; }
        .time-axis { display: grid; grid-auto-flow: column; margin-left: 65px; gap: 2px; }
        .time-label { font-size: 0.7em; text-align: center; color: #999; }

        /* Vehicle Legend */
        .vehicle-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        .legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.85em; background: white; padding: 4px 8px; border-radius: 12px; border: 1px solid #eee; }
        .legend-color { width: 12px; height: 12px; border-radius: 50%; }

        .ev-item { background: white; padding: 8px; margin-bottom: 5px; border-radius: 4px; display: flex; justify-content: space-between; border-left: 3px solid #8e44ad; }
        .ev-list { max-height: 200px; overflow-y: auto; margin-top: 10px; }

        .section-toggle { background: #e8e8e8; padding: 8px; cursor: pointer; border-radius: 4px; margin-bottom: 5px; font-weight: 600; font-size: 0.9em; user-select: none; }
        .section-content { display: none; padding: 10px 0; }
        .section-content.open { display: block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐜 EV Charging Optimizer <span class="aco-badge">ACO</span></h1>
            <p>Ant Colony Optimization - Layer-by-Layer Construction Visualization</p>
        </div>

        <div class="main-grid">
            <div class="left-panel">

                <!-- File Upload Section -->
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📂</div>
                    <div>Click to Upload JSON Config</div>
                    <div class="file-name" id="fileName"></div>
                    <input type="file" id="fileInput" accept=".json" onchange="handleFileUpload(this)">
                </div>

                <div class="section-toggle" onclick="this.nextElementSibling.classList.toggle('open')">🏢 Station & EVs ▼</div>
                <div class="section-content open">
                    <div class="form-row">
                        <div class="form-group"><label>Time Slots</label><input type="number" id="timeSlots" value="24"></div>
                        <div class="form-group"><label>Power Max</label><input type="number" id="powerMax" value="50"></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>Lines</label><input type="number" id="numLines" value="2"></div>
                        <div class="form-group"><label>Spots/Line</label><input type="number" id="spotsPerLine" value="3"></div>
                    </div>
                    <div class="form-group"><label>Levels (kW)</label><input type="text" id="chargingLevels" value="3, 7, 11"></div>

                    <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">

                    <div class="form-row">
                        <div class="form-group"><label>Arr</label><input type="number" id="evArrival" value="0"></div>
                        <div class="form-group"><label>Dep</label><input type="number" id="evDeparture" value="10"></div>
                    </div>
                    <div class="form-group"><label>Energy (kWh)</label><input type="number" id="evEnergy" value="20"></div>
                    <button class="btn" style="background:#34495e" onclick="addEV()">+ Add EV</button>
                    <div class="ev-list" id="evList"></div>
                </div>

                <div class="section-toggle" onclick="this.nextElementSibling.classList.toggle('open')">⚙️ ACO Parameters ▼</div>
                <div class="section-content open">
                    <div class="form-row">
                        <div class="form-group"><label>Ants</label><input type="number" id="nAnts" value="20"></div>
                        <div class="form-group"><label>Iterations</label><input type="number" id="nIter" value="50"></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>Alpha (Pheromone)</label><input type="number" id="alpha" value="1.0" step="0.1"></div>
                        <div class="form-group"><label>Beta (Heuristic)</label><input type="number" id="beta" value="2.0" step="0.1"></div>
                    </div>
                    <div class="form-row">
                        <div class="form-group"><label>Rho (Evap)</label><input type="number" id="rho" value="0.1" step="0.05"></div>
                        <div class="form-group"><label>Q (Deposit)</label><input type="number" id="Q" value="100"></div>
                    </div>
                </div>

                <button class="btn" onclick="startACO()" id="startBtn">🚀 Start Colony</button>

                <div class="progress-container" id="progressContainer">
                    <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                        <strong>Optimization Progress</strong>
                        <span id="progText">0/0</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="progressBar"></div>
                    </div>
                    <div class="metrics-grid">
                        <div class="metric-box"><div class="label">Best Fit</div><div class="value" id="progBest">-</div></div>
                        <div class="metric-box"><div class="label">Avg Fit</div><div class="value" id="progAvg">-</div></div>
                        <div class="metric-box"><div class="label">Iter Best</div><div class="value" id="progIterBest">-</div></div>
                    </div>
                </div>
            </div>

            <div class="right-panel">
                <div class="metrics-panel" id="metricsPanel">
                    <div class="metrics-grid" style="grid-template-columns: repeat(4, 1fr);">
                        <div class="metric-box"><div class="label">Iteration</div><div class="value" id="dispIter">-</div></div>
                        <div class="metric-box"><div class="label">Objective</div><div class="value" id="dispObj">-</div></div>
                        <div class="metric-box"><div class="label">Tardiness</div><div class="value" id="dispTard">-</div></div>
                        <div class="metric-box"><div class="label">Peak Power</div><div class="value" id="dispPeak">-</div></div>
                    </div>
                    <div style="margin-top:10px; display:flex; gap:10px;">
                        <input type="range" id="iterSlider" style="flex:1" min="1" value="1" oninput="loadIteration(this.value)">
                    </div>
                </div>

                <div class="viz-panel">
                    <div class="vehicle-legend" id="vehicleLegend"></div>
                    <div id="visualization"></div>
                    <div style="margin-top:20px;">
                        <h4>Power Profile</h4>
                        <canvas id="powerChart" height="80"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        let evs = [];
        let sessionId = null;
        let totalIter = 0;
        let chartInstance = null;
        const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c', '#34495e'];

        // File Upload Handler
        function handleFileUpload(input) {
            const file = input.files[0];
            if (!file) return;

            document.getElementById('fileName').innerText = file.name;
            document.getElementById('fileName').style.display = 'block';

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const data = JSON.parse(e.target.result);
                    populateForm(data);
                } catch (err) {
                    alert("Invalid JSON file!");
                    console.error(err);
                }
            };
            reader.readAsText(file);
        }

        function populateForm(data) {
            // Update simple fields
            if(data.time_slots) document.getElementById('timeSlots').value = data.time_slots;
            if(data.P_max) document.getElementById('powerMax').value = data.P_max;

            // Handle Charging Levels
            if(data.charging_levels) {
                // If dict {"1":3.0, ...} convert to list sorted by keys
                const levels = Object.entries(data.charging_levels)
                    .sort((a,b) => parseInt(a[0]) - parseInt(b[0]))
                    .map(pair => pair[1])
                    .join(', ');
                document.getElementById('chargingLevels').value = levels;
            } else if(data.level_powers) {
                document.getElementById('chargingLevels').value = data.level_powers.join(', ');
            }

            // Handle Lines/Spots
            if(data.lines) {
                // If dict {"L1": 3, "L2": 3}
                if(typeof data.lines === 'object' && !Array.isArray(data.lines)) {
                    const counts = Object.values(data.lines);
                    document.getElementById('numLines').value = counts.length;
                    // Assume uniform spots if possible, or warn user. 
                    // For the UI input simplified to "Spots/Line", we take the max or first.
                    // Ideally UI would support variable lines, but simple UI takes uniform.
                    document.getElementById('spotsPerLine').value = Math.max(...counts);
                } else if (Array.isArray(data.lines)) {
                    document.getElementById('numLines').value = data.lines.length;
                    document.getElementById('spotsPerLine').value = Math.max(...data.lines);
                }
            } else if (data.spots_per_line) {
                document.getElementById('numLines').value = data.spots_per_line.length;
                document.getElementById('spotsPerLine').value = Math.max(...data.spots_per_line);
            }

            // Handle EVs
            if(data.evs && Array.isArray(data.evs)) {
                evs = []; // Clear existing
                data.evs.forEach((ev, i) => {
                    evs.push({
                        id: ev.id || `EV${i+1}`,
                        arrival_slot: ev.arrival_slot !== undefined ? ev.arrival_slot : ev.arrival,
                        departure_slot: ev.departure_slot !== undefined ? ev.departure_slot : ev.departure,
                        energy_required: ev.energy_required
                    });
                });
                renderEVList();
            }
            alert("Configuration loaded from file!");
        }

        function addEV() {
            evs.push({
                id: `EV${evs.length+1}`,
                arrival_slot: parseInt(document.getElementById('evArrival').value),
                departure_slot: parseInt(document.getElementById('evDeparture').value),
                energy_required: parseFloat(document.getElementById('evEnergy').value)
            });
            renderEVList();
        }

        function renderEVList() {
            const el = document.getElementById('evList');
            el.innerHTML = evs.map((ev, i) => `
                <div class="ev-item">
                    <span>${ev.id}: ${ev.arrival_slot}→${ev.departure_slot}, ${ev.energy_required}kWh</span>
                    <span style="cursor:pointer; color:red" onclick="evs.splice(${i},1);renderEVList()">✖</span>
                </div>
            `).join('');
        }

        async function startACO() {
            if(!evs.length) return alert("Add EVs first!");

            const payload = {
                evs: evs,
                time_slots: parseInt(document.getElementById('timeSlots').value),
                P_max: parseFloat(document.getElementById('powerMax').value),
                lines: Array(parseInt(document.getElementById('numLines').value)).fill(parseInt(document.getElementById('spotsPerLine').value)),
                charging_levels: document.getElementById('chargingLevels').value.split(',').reduce((acc, v, i) => ({...acc, [i+1]: parseFloat(v)}), {}),
                n_ants: parseInt(document.getElementById('nAnts').value),
                n_iterations: parseInt(document.getElementById('nIter').value),
                alpha: parseFloat(document.getElementById('alpha').value),
                beta: parseFloat(document.getElementById('beta').value),
                rho: parseFloat(document.getElementById('rho').value),
                Q: parseFloat(document.getElementById('Q').value)
            };

            document.getElementById('startBtn').disabled = true;
            document.getElementById('progressContainer').classList.add('active');

            const res = await fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            sessionId = data.session_id;

            const evtSource = new EventSource(`/api/progress/${sessionId}`);
            evtSource.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if(msg.type === 'progress') {
                    const pct = (msg.iteration / payload.n_iterations) * 100;
                    document.getElementById('progressBar').style.width = pct + '%';
                    document.getElementById('progText').innerText = `${msg.iteration}/${payload.n_iterations}`;
                    document.getElementById('progBest').innerText = msg.best_fitness.toFixed(1);
                    document.getElementById('progAvg').innerText = msg.avg_fitness.toFixed(1);
                    document.getElementById('progIterBest').innerText = msg.current_fitness.toFixed(1);

                    // Auto load latest
                    totalIter = msg.iteration;
                    document.getElementById('iterSlider').max = totalIter;
                    document.getElementById('iterSlider').value = totalIter;
                    loadIteration(totalIter);
                } else if(msg.type === 'complete') {
                    evtSource.close();
                    document.getElementById('startBtn').disabled = false;
                    alert("Optimization Complete!");
                }
            };
        }

        async function loadIteration(it) {
            if(!sessionId) return;
            const res = await fetch(`/api/state/${sessionId}/${it}`);
            const data = await res.json();
            renderViz(data);
        }

        function renderViz(data) {
            const s = data.state;
            const p = data.params;

            // Metrics
            document.getElementById('metricsPanel').style.display = 'block';
            document.getElementById('dispIter').innerText = s.metrics.iteration;
            document.getElementById('dispObj').innerText = s.metrics.objective.toFixed(2);
            document.getElementById('dispTard').innerText = s.metrics.tardiness.toFixed(2);
            document.getElementById('dispPeak').innerText = s.metrics.peak_power.toFixed(2);

            // Legend
            const leg = document.getElementById('vehicleLegend');
            leg.innerHTML = s.vehicles.map((v, i) => `
                <div class="legend-item">
                    <div class="legend-color" style="background:${COLORS[i % COLORS.length]}"></div>
                    ${v.name}
                </div>
            `).join('');

            // Grid
            const viz = document.getElementById('visualization');
            let html = '';

            p.spots_per_line.forEach((spots, lIdx) => {
                html += `<div class="charging-line"><div style="margin-bottom:5px; font-weight:bold; color:#7f8c8d">Line ${lIdx+1}</div>
                <div class="timeline-container">
                    <div style="display:flex; flex-direction:column; gap:2px; justify-content:space-around;">
                        ${Array(spots).fill(0).map((_, i) => `<div style="font-size:0.8em; color:#999">Spot ${i+1}</div>`).join('')}
                    </div>
                    <div>`;

                for(let sp = 0; sp < spots; sp++) {
                    html += `<div class="spot-timeline" style="grid-template-columns: repeat(${p.time_slots}, 1fr)">`;
                    for(let t = 0; t < p.time_slots; t++) {
                        let content = '';
                        let bg = '#eee';

                        // Find vehicle
                        s.vehicles.forEach((v, vIdx) => {
                            const assign = v.assignments.find(a => a.line === lIdx && a.spot === sp && a.slot === t);
                            if(assign) {
                                bg = COLORS[vIdx % COLORS.length];
                                content = `<span style="font-size:0.8em">L${assign.level+1}</span>`;
                            }
                        });
                        html += `<div class="time-cell" style="background:${bg}">${content}</div>`;
                    }
                    html += `</div>`;
                }

                html += `</div></div></div>`;
            });

            // Time Axis
            html += `<div class="time-axis" style="grid-template-columns: repeat(${p.time_slots}, 1fr); margin-left:65px">
                ${Array(p.time_slots).fill(0).map((_, i) => `<div class="time-label">${i}</div>`).join('')}
            </div>`;

            viz.innerHTML = html;

            // Chart
            const ctx = document.getElementById('powerChart').getContext('2d');
            if(chartInstance) chartInstance.destroy();
            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: p.time_slots}, (_, i) => i),
                    datasets: [{
                        label: 'Total Power (kW)',
                        data: s.power_profile,
                        borderColor: '#8e44ad',
                        backgroundColor: 'rgba(142, 68, 173, 0.2)',
                        fill: true,
                        tension: 0.3
                    }, {
                        label: 'Limit',
                        data: Array(p.time_slots).fill(p.P_max),
                        borderColor: 'red',
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: { responsive: true, scales: { y: { beginAtZero: true } } }
            });
        }

        // Init default
        addEV(); addEV(); addEV();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting EV Charging Visualization Server (ACO)...")
    print("Open your browser to: http://localhost:5002")
    app.run(debug=True, host='0.0.0.0', port=5002)