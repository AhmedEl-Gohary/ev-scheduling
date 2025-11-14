#!/usr/bin/env python3

"""
Flask Web Application for EV Charging Schedule Visualization - Genetic Algorithm
Run with: python ga_app.py
Then open: http://localhost:5001
"""

from flask import Flask, render_template_string, jsonify, request, Response
from flask_cors import CORS
import secrets
import json
import threading
import queue
from algorithms.ga import genetic_algorithm
from algorithms.greedy import greedy_schedule
from eval import *

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

algorithm_states = {}
progress_queues = {}

# ============================================================================
# PROGRESS CALLBACK FOR GA
# ============================================================================

def create_progress_callback(session_id):
    """Create a callback function that sends progress updates"""
    def callback(generation, current_fitness, best_fitness, avg_fitness, message=""):
        if session_id in progress_queues:
            progress_data = {
                'type': 'progress',
                'generation': generation,
                'current_fitness': float(current_fitness),
                'best_fitness': float(best_fitness),
                'avg_fitness': float(avg_fitness),
                'message': message
            }
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
    """Start the GA algorithm with given parameters"""
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

    # Create session
    session_id = secrets.token_hex(8)
    progress_queues[session_id] = queue.Queue()

    # Run GA in background thread
    def run_ga():
        try:
            progress_queues[session_id].put({
                'type': 'status',
                'message': 'Starting genetic algorithm...'
            })
            
            X_best, B_best, states = genetic_algorithm(
                params,
                imax=params.get('imax', 150),
                population_size=params.get('population_size', 100),
                survivor_rate=params.get('survivor_rate', 0.1),
                crossover_rate=params.get('crossover_rate', 0.4),
                mutation_rate=params.get('mutation_rate', 0.5),
                rng_seed=params.get('seed', 42),
                adaptive=params.get('adaptive', True),
                progress_callback=create_progress_callback(session_id)
            )

            # Store results
            algorithm_states[session_id] = {
                'params': params,
                'states': states,
                'current_generation': 0
            }

            progress_queues[session_id].put({
                'type': 'complete',
                'session_id': session_id,
                'total_generations': len(states)
            })

        except Exception as e:
            progress_queues[session_id].put({
                'type': 'error',
                'message': str(e)
            })

    thread = threading.Thread(target=run_ga, daemon=True)
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
                # Wait for updates with timeout
                data = q.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                
                # If complete or error, close stream
                if data['type'] in ['complete', 'error']:
                    break
                    
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/state/<session_id>/<int:generation>')
def get_state(session_id, generation):
    """Get state at specific generation"""
    if session_id not in algorithm_states:
        return jsonify({'error': 'Session not found'}), 404

    data = algorithm_states[session_id]
    states = data['states']

    if generation < 0 or generation >= len(states):
        return jsonify({'error': 'Invalid generation'}), 400

    state = states[generation]
    params = data['params']

    # Convert state to visualization format
    viz_data = convert_state_to_viz(state, params)

    return jsonify({
        'generation': generation,
        'total_generations': len(states),
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
            'avg_fitness': state['avg_fitness'],
            'tardiness': state['tardiness'],
            'peak_power': state['peak_power'],
            'generation': state['generation']
        }
    }


# ============================================================================
# HTML TEMPLATE (Updated with Progress Display)
# ============================================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV Charging Station Optimizer - Genetic Algorithm</title>
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
            border-left: 4px solid #27ae60;
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
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #27ae60;
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
            background: #27ae60;
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
            background: #229954;
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-small {
            padding: 6px 12px;
            font-size: 0.85em;
        }

        .progress-container {
            background: #ecf0f1;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }

        .progress-container.active {
            display: block;
        }

        .progress-header {
            font-weight: 600;
            margin-bottom: 15px;
            color: #2c3e50;
            font-size: 1.1em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .spinner {
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #27ae60;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .progress-bar-container {
            background: #ddd;
            border-radius: 10px;
            height: 24px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .progress-bar {
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }

        .progress-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
        }

        .progress-stat {
            background: white;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .progress-stat .label {
            font-size: 0.75em;
            color: #7f8c8d;
            text-transform: uppercase;
        }

        .progress-stat .value {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 4px;
        }

        .progress-log {
            background: white;
            border-radius: 6px;
            padding: 10px;
            margin-top: 15px;
            max-height: 150px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
        }

        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #ecf0f1;
        }

        .log-entry:last-child {
            border-bottom: none;
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
            background: #ddd;
        }

        .section-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }

        .section-content.open {
            max-height: 1000px;
        }

        .ga-badge {
            background: #27ae60;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧬 EV Charging Station Optimizer <span class="ga-badge">GENETIC ALGORITHM</span></h1>
            <p>Evolutionary Optimization with Real-time Progress</p>
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

                <div class="section-toggle" onclick="toggleSection('gaConfig')">
                    🧬 GA Parameters ▼
                </div>
                <div class="section-content" id="gaConfig">
                    <div class="form-row">
                        <div class="form-group">
                            <label>Generations</label>
                            <input type="number" id="imax" value="150" min="1">
                        </div>
                        <div class="form-group">
                            <label>Population Size</label>
                            <input type="number" id="popSize" value="100" min="10">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Survivor Rate</label>
                            <input type="number" id="survivorRate" value="0.1" step="0.05" min="0" max="1">
                        </div>
                        <div class="form-group">
                            <label>Crossover Rate</label>
                            <input type="number" id="crossoverRate" value="0.4" step="0.05" min="0" max="1">
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Mutation Rate</label>
                            <input type="number" id="mutationRate" value="0.5" step="0.05" min="0" max="1">
                        </div>
                        <div class="form-group">
                            <label>Adaptive</label>
                            <select id="adaptive">
                                <option value="true">Yes</option>
                                <option value="false">No</option>
                            </select>
                        </div>
                    </div>
                </div>

                <button class="btn" onclick="startOptimization()" id="startBtn">
                    🧬 Start Evolution
                </button>

                <div class="progress-container" id="progressContainer">
                    <div class="progress-header">
                        <div class="spinner"></div>
                        <span>Evolution in Progress...</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" id="progressBar">0%</div>
                    </div>
                    <div class="progress-stats">
                        <div class="progress-stat">
                            <div class="label">Generation</div>
                            <div class="value" id="progGen">0</div>
                        </div>
                        <div class="progress-stat">
                            <div class="label">Best Fitness</div>
                            <div class="value" id="progBest">-</div>
                        </div>
                        <div class="progress-stat">
                            <div class="label">Avg Fitness</div>
                            <div class="value" id="progAvg">-</div>
                        </div>
                    </div>
                    <div class="progress-log" id="progressLog"></div>
                </div>
            </div>

            <div class="right-panel">
                <div class="viz-panel">
                    <div id="visualization">
                        <div class="loading">Configure parameters and click "Start Evolution"</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let eventSource = null;
        let evs = [];
        let maxGenerations = 150;

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

        function addLogEntry(message) {
            const log = document.getElementById('progressLog');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const timestamp = new Date().toLocaleTimeString();
            entry.textContent = `[${timestamp}] ${message}`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
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

                maxGenerations = parseInt(document.getElementById('imax').value);

                const input = {
                    time_slots: timeSlots,
                    delta_t: 1.0,
                    spots_per_line: Array(numLines).fill(spotsPerLine),
                    level_powers: chargingLevels,
                    P_max: parseFloat(document.getElementById('powerMax').value),
                    imax: maxGenerations,
                    population_size: parseInt(document.getElementById('popSize').value),
                    survivor_rate: parseFloat(document.getElementById('survivorRate').value),
                    crossover_rate: parseFloat(document.getElementById('crossoverRate').value),
                    mutation_rate: parseFloat(document.getElementById('mutationRate').value),
                    adaptive: document.getElementById('adaptive').value === 'true',
                    evs: evs
                };

                // Show progress UI
                document.getElementById('progressContainer').classList.add('active');
                document.getElementById('progressLog').innerHTML = '';
                document.getElementById('startBtn').disabled = true;
                document.getElementById('visualization').innerHTML = '<div class="loading">Initializing evolution...</div>';

                addLogEntry('Starting genetic algorithm...');

                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(input)
                });

                const data = await response.json();
                sessionId = data.session_id;

                // Connect to progress stream
                eventSource = new EventSource(`/api/progress/${sessionId}`);

                eventSource.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleProgressUpdate(data);
                };

                eventSource.onerror = function(error) {
                    console.error('SSE Error:', error);
                    eventSource.close();
                };

            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('startBtn').disabled = false;
                document.getElementById('progressContainer').classList.remove('active');
            }
        }

        function handleProgressUpdate(data) {
            if (data.type === 'progress') {
                const progress = (data.generation / maxGenerations) * 100;
                document.getElementById('progressBar').style.width = progress + '%';
                document.getElementById('progressBar').textContent = Math.round(progress) + '%';
                
                document.getElementById('progGen').textContent = `${data.generation}/${maxGenerations}`;
                document.getElementById('progBest').textContent = data.best_fitness.toFixed(2);
                document.getElementById('progAvg').textContent = data.avg_fitness.toFixed(2);

                if (data.message) {
                    addLogEntry(data.message);
                } else if (data.generation % 10 === 0) {
                    addLogEntry(`Gen ${data.generation}: Best=${data.best_fitness.toFixed(2)}, Avg=${data.avg_fitness.toFixed(2)}`);
                }
            } else if (data.type === 'status') {
                addLogEntry(data.message);
            } else if (data.type === 'complete') {
                addLogEntry('Evolution complete! Loading results...');
                eventSource.close();
                loadResults(data.session_id, data.total_generations);
            } else if (data.type === 'error') {
                addLogEntry('ERROR: ' + data.message);
                eventSource.close();
                document.getElementById('startBtn').disabled = false;
            }
        }

        async function loadResults(sid, totalGen) {
            try {
                // Load final generation
                const response = await fetch(`/api/state/${sid}/${totalGen - 1}`);
                const data = await response.json();

                document.getElementById('progressContainer').classList.remove('active');
                document.getElementById('startBtn').disabled = false;

                // Display results
                displayResults(data);
                addLogEntry('Results displayed successfully!');

            } catch (error) {
                console.error('Error loading results:', error);
                addLogEntry('Error loading results: ' + error.message);
                document.getElementById('startBtn').disabled = false;
            }
        }

        function displayResults(data) {
            const viz = document.getElementById('visualization');
            const state = data.state;
            const params = data.params;

            let html = '<h2>🎉 Optimization Complete</h2>';
            
            // Metrics Summary
            html += '<div class="metrics-grid">';
            html += `
                <div class="metric-box">
                    <div class="label">Final Objective</div>
                    <div class="value">${state.metrics.objective.toFixed(2)}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Tardiness</div>
                    <div class="value">${state.metrics.tardiness.toFixed(2)}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Peak Power</div>
                    <div class="value">${state.metrics.peak_power.toFixed(2)} kW</div>
                </div>
            `;
            html += '</div>';

            // Vehicle Schedule Summary
            html += '<h3 style="margin-top: 20px; margin-bottom: 10px;">Vehicle Schedules</h3>';
            html += '<div style="background: #f8f9fa; padding: 15px; border-radius: 6px;">';
            
            state.vehicles.forEach((vehicle, idx) => {
                const assignments = vehicle.assignments;
                if (assignments.length > 0) {
                    const startSlot = Math.min(...assignments.map(a => a.slot));
                    const endSlot = Math.max(...assignments.map(a => a.slot));
                    const totalEnergy = assignments.reduce((sum, a) => sum + a.power * params.delta_t, 0);
                    
                    html += `
                        <div style="background: white; padding: 10px; margin-bottom: 8px; border-radius: 4px; border-left: 4px solid #27ae60;">
                            <strong>${vehicle.name}</strong> (Required: ${vehicle.energy_required} kWh)<br>
                            <small>Scheduled: Slot ${startSlot} → ${endSlot + 1} | Energy Delivered: ${totalEnergy.toFixed(2)} kWh | 
                            Line ${assignments[0].line + 1}, Spot ${assignments[0].spot + 1}</small>
                        </div>
                    `;
                } else {
                    html += `
                        <div style="background: white; padding: 10px; margin-bottom: 8px; border-radius: 4px; border-left: 4px solid #e74c3c;">
                            <strong>${vehicle.name}</strong> - ⚠️ Not Scheduled
                        </div>
                    `;
                }
            });
            
            html += '</div>';

            // Power Profile Chart
            html += '<div style="margin-top: 20px; background: #f8f9fa; padding: 15px; border-radius: 6px;">';
            html += '<h3 style="margin-bottom: 10px;">Power Profile</h3>';
            html += '<canvas id="finalPowerChart" width="800" height="250"></canvas>';
            html += '</div>';

            viz.innerHTML = html;

            // Draw power chart
            setTimeout(() => drawPowerChart(state.power_profile, params.P_max), 100);
        }

        function drawPowerChart(powerProfile, pMax) {
            const canvas = document.getElementById('finalPowerChart');
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            if (powerProfile.length === 0) return;

            const maxPower = Math.max(...powerProfile, pMax || 0) * 1.1;
            const padding = 50;
            const chartWidth = width - 2 * padding;
            const chartHeight = height - 2 * padding;

            ctx.clearRect(0, 0, width, height);

            // Draw axes
            ctx.strokeStyle = '#ddd';
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
                ctx.font = 'bold 13px sans-serif';
                ctx.fillText(`Limit: ${pMax.toFixed(1)} kW`, width - padding + 10, yMax + 5);
            }

            // Draw power profile area
            ctx.strokeStyle = '#27ae60';
            ctx.fillStyle = 'rgba(39, 174, 96, 0.2)';
            ctx.lineWidth = 3;

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

            // Draw line
            ctx.beginPath();
            for (let i = 0; i < powerProfile.length; i++) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                const y = height - padding - (powerProfile[i] / maxPower) * chartHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            // Draw points
            ctx.fillStyle = '#27ae60';
            for (let i = 0; i < powerProfile.length; i++) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                const y = height - padding - (powerProfile[i] / maxPower) * chartHeight;
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            }

            // Draw Y-axis labels
            ctx.fillStyle = '#2c3e50';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'right';
            for (let i = 0; i <= 5; i++) {
                const value = (maxPower / 5) * i;
                const y = height - padding - (i / 5) * chartHeight;
                ctx.fillText(value.toFixed(1), padding - 10, y + 4);
            }

            // Draw X-axis labels
            ctx.textAlign = 'center';
            const step = Math.max(1, Math.ceil(powerProfile.length / 15));
            for (let i = 0; i < powerProfile.length; i += step) {
                const x = padding + (i / (powerProfile.length - 1)) * chartWidth;
                ctx.fillText(i.toString(), x, height - padding + 20);
            }

            // Labels
            ctx.textAlign = 'center';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillStyle = '#2c3e50';
            ctx.fillText('Time Slot', width / 2, height - 8);

            ctx.save();
            ctx.translate(20, height / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText('Power (kW)', 0, 0);
            ctx.restore();
        }

        // Initialize with default EVs
        loadDefaultEVs();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print("Starting EV Charging GA Visualization Server...")
    print("Open your browser to: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)