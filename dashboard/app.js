document.addEventListener('DOMContentLoaded', () => {
    const trajectoryCanvas = document.getElementById('trajectoryChart');
    const logStream = document.getElementById('log-stream');
    const valScore = document.getElementById('val-score');
    const valDelta = document.getElementById('val-delta');
    const valStep = document.getElementById('val-step');
    const currentSolution = document.getElementById('current-solution');
    const rubricList = document.getElementById('rubric-list');
    const btnReset = document.getElementById('btn-reset');
    const btnRun = document.getElementById('btn-run');
    const taskSelect = document.getElementById('task-select');

    let trajectoryChart = null;
    if (trajectoryCanvas) {
        const ctx = trajectoryCanvas.getContext('2d');
        trajectoryChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'TOTAL SCORE',
                    data: [],
                    borderColor: '#ffffff',
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { min: 0, max: 1.0, grid: { color: '#111' }, ticks: { color: '#555', font: { family: 'JetBrains Mono' } } },
                    x: { display: false }
                }
            }
        });
    }

    let initialScore = 0.0;
    let ws = null;

    function connect() {
        ws = new WebSocket(`ws://${location.host}/ws/stream`);
        ws.onopen = () => logSystem("Link established to Gödel Env core.");
        ws.onclose = () => {
            logError("Link severed. Retrying...");
            setTimeout(connect, 2000);
        };
        ws.onmessage = (e) => {
            handleMessage(JSON.parse(e.data));
        };
    }

    function logSystem(msg) {
        if (!logStream) return;
        logStream.innerHTML += `<div class="log-entry system">> [SYSTEM] ${msg}</div>`;
        scrollLogs();
    }

    function logError(msg) {
        if (!logStream) return;
        logStream.innerHTML += `<div class="log-entry error">! [ERROR] ${msg}</div>`;
        scrollLogs();
    }

    function scrollLogs() {
        logStream.scrollTop = logStream.scrollHeight;
    }

    function handleMessage(msg) {
        if (msg.type === 'reset') {
            if (trajectoryChart) {
                trajectoryChart.data.labels = [0];
                trajectoryChart.data.datasets[0].data = [msg.data.initial_score];
                trajectoryChart.update();
            }
            
            initialScore = msg.data.initial_score;
            if (valScore) valScore.innerText = initialScore.toFixed(2);
            if (valDelta) valDelta.innerText = "0.00";
            if (valStep) valStep.innerText = "0";
            if (currentSolution) currentSolution.innerText = msg.data.initial_solution;
            
            logSystem(`Environment loaded. Task: ${msg.data.task_id.toUpperCase()}`);
            logSystem(`Prompt: "${msg.data.prompt.substring(0, 80)}..."`);
            logSystem(`Baseline score: ${initialScore.toFixed(3)} — improvement loop running...`);
        } else if (msg.type === 'step') {
            const data = msg.data;
            if (trajectoryChart) {
                trajectoryChart.data.labels.push(data.step);
                trajectoryChart.data.datasets[0].data.push(data.score);
                trajectoryChart.update();
            }

            if (valScore) valScore.innerText = data.score.toFixed(2);
            const delta = data.score - initialScore;
            if (valDelta) valDelta.innerText = (delta >= 0 ? "+" : "") + delta.toFixed(3);
            if (valStep) valStep.innerText = data.step;
            if (currentSolution) currentSolution.innerText = data.solution;
            
            logSystem(`[STEP ${data.step}] SCORE: ${data.score.toFixed(3)} | Δ: ${delta >= 0 ? "+" : ""}${delta.toFixed(3)} | ${data.terminated ? "TERMINATED" : "running"}`);
            renderRubrics(data.rubrics);
        }
    }

    function renderRubrics(rubrics) {
        if (!rubricList) return;
        rubricList.innerHTML = '';
        for (const [name, score] of Object.entries(rubrics)) {
            const row = document.createElement('div');
            row.className = 'rubric-row';
            row.innerHTML = `
                <div class="rubric-info">
                    <span>${name.toUpperCase()}</span>
                    <span>${(score * 100).toFixed(0)}%</span>
                </div>
                <div class="rubric-progress">
                    <div class="rubric-fill" style="width: ${score * 100}%"></div>
                </div>
            `;
            rubricList.appendChild(row);
        }
    }

    btnReset.onclick = () => {
        logSystem("Environment cleared.");
        if (trajectoryChart) {
            trajectoryChart.data.labels = [];
            trajectoryChart.data.datasets[0].data = [];
            trajectoryChart.update();
        }
        if (valScore) valScore.innerText = "0.00";
        if (valDelta) valDelta.innerText = "+0.00";
        if (valStep) valStep.innerText = "0";
        if (currentSolution) currentSolution.innerText = "Awaiting deployment...";
        if (rubricList) rubricList.innerHTML = '';
        initialScore = 0.0;
    };

    btnRun.onclick = async () => {
        const task = taskSelect.value;
        logSystem(`Activating task: ${task.toUpperCase().replace(/_/g, ' ')}...`);
        btnRun.disabled = true;
        btnRun.innerText = "LOADING...";

        try {
            await fetch('/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_type: task })
            });

            await new Promise(r => setTimeout(r, 800));

            if (task === 'strategy_optimization') {
                logSystem("Initiating Meta-Strategy Auto-Improvement Protocol...");
                btnRun.innerText = "IMPROVING...";
                const maxRuns = 10;
                for (let i = 1; i <= maxRuns; i++) {
                    logSystem(`LLM self-improvement step ${i}/${maxRuns}...`);
                    const res = await fetch('/run', { method: 'POST' });
                    const stepData = await res.json();
                    await new Promise(r => setTimeout(r, 1200));
                    
                    if (stepData.terminated || stepData.truncated) {
                        logSystem(`Environment signaled completion (terminated: ${stepData.terminated}).`);
                        break;
                    }
                }
                logSystem("Agent trajectory complete.");
            } else {
                logSystem("Executing Single-Shot RL Inference...");
                btnRun.innerText = "RUNNING...";
                await fetch('/run', { method: 'POST' });
                logSystem("Inference complete.");
            }

        } catch (err) {
            logError(`Activation failed: ${err.message || err}`);
        }

        btnRun.innerText = "RUN ▶";
        btnRun.disabled = false;
    };

    connect();
});
