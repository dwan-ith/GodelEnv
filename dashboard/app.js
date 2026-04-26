document.addEventListener("DOMContentLoaded", () => {
    const trajectoryCanvas = document.getElementById("trajectoryChart");
    const logStream = document.getElementById("log-stream");
    const valScore = document.getElementById("val-score");
    const valDelta = document.getElementById("val-delta");
    const valStep = document.getElementById("val-step");
    const valElo = document.getElementById("val-elo");
    const valGen = document.getElementById("val-gen");
    const valBudget = document.getElementById("val-budget");
    const valMode = document.getElementById("val-mode");
    const currentSolution = document.getElementById("current-solution");
    const currentStrategy = document.getElementById("current-strategy");
    const childStrategy = document.getElementById("child-strategy");
    const utilityComparison = document.getElementById("utility-comparison");
    const failureList = document.getElementById("failure-list");
    const rubricList = document.getElementById("rubric-list");
    const btnReset = document.getElementById("btn-reset");
    const btnRun = document.getElementById("btn-run");
    const taskSelect = document.getElementById("task-select");

    let trajectoryChart = null;
    let initialScore = 0.0;
    let ws = null;
    let currentObservation = null;
    let lastAgentMode = "unknown";



    async function fetchProviderStatus() {
        try {
            const response = await fetch("/demo/provider-status");
            if (response.ok) {
                const status = await response.json();
                const hasLlm = status.providers?.some(p => p.configured && !p.disabled);
                logSystem(`Provider status: ${hasLlm ? 'LLM available' : 'No LLM configured'}`);
                if (status.env_presence) {
                    const keys = Object.entries(status.env_presence)
                        .filter(([_, v]) => v)
                        .map(([k]) => k);
                    if (keys.length > 0) {
                        logSystem(`API keys detected: ${keys.join(', ')}`);
                    }
                }
            }
        } catch (e) {
            logSystem(`Provider status check failed: ${e.message}`);
        }
    }

    fetchProviderStatus();

    if (trajectoryCanvas) {
        const ctx = trajectoryCanvas.getContext("2d");
        trajectoryChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "TOTAL SCORE",
                    data: [],
                    borderColor: "#ffffff",
                    backgroundColor: "rgba(255, 255, 255, 0.05)",
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
                    y: {
                        min: 0,
                        max: 1.0,
                        grid: { color: "#111" },
                        ticks: { color: "#555", font: { family: "JetBrains Mono" } }
                    },
                    x: { display: false }
                }
            }
        });
    }

    function logSystem(message) {
        if (!logStream) return;
        logStream.innerHTML += `<div class="log-entry system">&gt; [SYSTEM] ${message}</div>`;
        logStream.scrollTop = logStream.scrollHeight;
    }

    function logError(message) {
        if (!logStream) return;
        logStream.innerHTML += `<div class="log-entry error">! [ERROR] ${message}</div>`;
        logStream.scrollTop = logStream.scrollHeight;
    }

    function describeGrading(source, error) {
        if (source === "llm") {
            return "Grading source: API-backed LLM.";
        }
        if (error) {
            return `Grading source: deterministic fallback (${error}).`;
        }
        return "Grading source: deterministic fallback.";
    }

    function renderRubrics(rubrics) {
        if (!rubricList) return;
        rubricList.innerHTML = "";
        Object.entries(rubrics || {}).forEach(([name, score]) => {
            const row = document.createElement("div");
            row.className = "rubric-row";
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
        });
    }

    function updateUiFromObservation(stepResult) {
        const obs = stepResult.observation;
        const metadata = obs.metadata || {};
        const patchDecision = obs.patch_decision || metadata.patch_decision;
        const rewardBreakdown = obs.reward_breakdown || metadata.reward_breakdown;
        currentObservation = obs;

        if (trajectoryChart) {
            if (obs.step === 0) {
                trajectoryChart.data.labels = [0];
                trajectoryChart.data.datasets[0].data = [obs.total_score];
            } else {
                trajectoryChart.data.labels.push(obs.step);
                trajectoryChart.data.datasets[0].data.push(obs.total_score);
            }
            trajectoryChart.update();
        }

        if (obs.step === 0) {
            initialScore = obs.total_score;
            logSystem(`Environment loaded. Task: ${obs.task_id.toUpperCase()}`);
            logSystem(`Prompt: "${obs.task_prompt.substring(0, 80)}..."`);
            logSystem(`Baseline score: ${initialScore.toFixed(3)}`);
        }

        if (valScore) valScore.innerText = obs.total_score.toFixed(2);
        const delta = obs.total_score - initialScore;
        if (valDelta) valDelta.innerText = (delta >= 0 ? "+" : "") + delta.toFixed(3);
        if (valStep) valStep.innerText = obs.step;
        if (valElo) valElo.innerText = Math.round(obs.strategy_elo || 1000);
        if (valGen) valGen.innerText = obs.strategy_generation || 0;
        if (valBudget) valBudget.innerText = obs.budget_remaining || 0;
        
        if (currentSolution) currentSolution.innerText = obs.current_solution;
        if (currentStrategy) currentStrategy.innerText = obs.current_strategy || "No strategy defined.";

        if (failureList) {
            failureList.innerHTML = "";
            const failures = obs.recent_failures || [];
            
            // Show strategy-level rejection reasons if available
            if (patchDecision && patchDecision.rejection_reasons?.length > 0) {
                patchDecision.rejection_reasons.forEach(r => {
                    const item = document.createElement("div");
                    item.className = "failure-item";
                    item.style.color = "#ff8888"; // Subtle indicator for strategy errors
                    item.innerText = `[REJECTED PATCH] ${r}`;
                    failureList.appendChild(item);
                });
            }

            if (failures.length === 0 && (!patchDecision || !patchDecision.rejection_reasons?.length)) {
                failureList.innerHTML = '<div class="empty">No failures recorded.</div>';
            } else {
                failures.forEach(f => {
                    const item = document.createElement("div");
                    item.className = "failure-item";
                    item.innerText = f.substring(0, 100) + (f.length > 100 ? "..." : "");
                    failureList.appendChild(item);
                });
            }
        }

        renderRubrics(obs.rubric_scores?.scores);
        logSystem(describeGrading(obs.grading_source, obs.grading_error));

        if (obs.step > 0) {
            const isPatch = !!stepResult.action?.strategy_patch;
            const actionType = isPatch ? "STRATEGY PATCH" : "SOLUTION EDIT";
            
            logSystem(
                `[STEP ${obs.step}] ${actionType} | SCORE: ${obs.total_score.toFixed(3)} | delta: ` +
                `${delta >= 0 ? "+" : ""}${delta.toFixed(3)} | ` +
                `${stepResult.done ? "DONE" : "running"}`
            );
            
            if (isPatch) {
                const improvedStrategy = stepResult.action.strategy_patch?.improved_strategy;
                if (childStrategy) {
                    childStrategy.innerText = improvedStrategy || "(Patch proposed — no improved_strategy text returned by model)";
                }
                // Don't write N/A here — set below based on patchDecision
            } else {
                if (childStrategy) childStrategy.innerHTML = '<div class="empty">No patch proposed in this step.</div>';
                if (utilityComparison) utilityComparison.innerHTML = '<div class="empty">N/A</div>';
            }

            if (patchDecision) {
                const verdict = patchDecision.accepted ? "ACCEPTED" : "REJECTED";
                logSystem(
                    `Governor ${verdict}: utility ` +
                    `${(patchDecision.parent_utility ?? 0).toFixed(3)} -> ` +
                    `${(patchDecision.child_utility ?? 0).toFixed(3)} ` +
                    `(Δ ${(patchDecision.improvement ?? 0).toFixed(3)})`
                );
                if (!patchDecision.accepted && patchDecision.rejection_reasons?.length) {
                    logSystem(`Rejection reasons: ${patchDecision.rejection_reasons.join("; ")}`);
                }

                if (utilityComparison) {
                    const imp = patchDecision.improvement ?? 0;
                    const deltaClass = imp > 0 ? "delta-pos" : (imp < 0 ? "delta-neg" : "delta-neu");
                    const sign = imp > 0 ? "+" : "";
                    let html = `
                    <div class="utility-board" style="padding:1rem">
                        <div style="font-weight:bold;margin-bottom:1rem;color:${patchDecision.accepted ? '#00ff88' : '#ff3e3e'}">
                            VERDICT: ${verdict}
                        </div>
                        <div class="utility-scores-grid">
                            <div class="utility-score-card">
                                <div class="label">PARENT UTILITY</div>
                                <div class="value">${(patchDecision.parent_utility ?? 0).toFixed(3)}</div>
                            </div>
                            <div class="utility-score-card">
                                <div class="label">CHILD UTILITY</div>
                                <div class="value">${(patchDecision.child_utility ?? 0).toFixed(3)}</div>
                            </div>
                        </div>
                        <div class="utility-delta">
                            NET IMPROVEMENT: <span class="${deltaClass}">${sign}${imp.toFixed(3)}</span>
                        </div>`;
                    if (patchDecision.rejection_reasons?.length > 0) {
                        html += `<div class="rejection-reasons-box"><div class="title">REJECTION REASONS:</div><ul>` +
                            patchDecision.rejection_reasons.map(r => `<li>${r}</li>`).join('') +
                            `</ul></div>`;
                    }
                    html += `</div>`;
                    utilityComparison.innerHTML = html;
                }
            } else if (isPatch && utilityComparison) {
                // Patch was proposed but no governor decision returned — pre-eval guard rejected it
                utilityComparison.innerHTML = '<div style="color:#ff8888;padding:0.75rem;font-size:0.8rem">PATCH REJECTED BY PRE-EVAL GUARD<br><span style="color:#888">improved_strategy too short or failed format check — Governor was not reached.</span></div>';
            }
            if (rewardBreakdown) {
                logSystem(
                    `Reward channels: patch=${(rewardBreakdown.patch_quality || 0).toFixed(3)}, ` +
                    `generalization=${(rewardBreakdown.generalization_score || 0).toFixed(3)}, ` +
                    `anti_hack=${(rewardBreakdown.anti_hack_penalty || 0).toFixed(3)}, ` +
                    `total=${(rewardBreakdown.total || stepResult.reward || 0).toFixed(3)}`
                );
            }
            if (obs.rubric_scores?.feedback) {
                const preview = JSON.stringify(obs.rubric_scores.feedback).substring(0, 120);
                logSystem(`Feedback: ${preview}...`);
            }
        }
    }

    async function ensureSession() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            return;
        }
        if (ws && ws.readyState === WebSocket.CONNECTING) {
            await new Promise((resolve) => {
                ws.addEventListener("open", resolve, { once: true });
            });
            return;
        }

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${protocol}//${location.host}/ws`);

        await new Promise((resolve, reject) => {
            ws.addEventListener("open", () => {
                logSystem("Linked to OpenEnv websocket session.");
                resolve();
            }, { once: true });
            ws.addEventListener("error", () => reject(new Error("WebSocket connection failed")), { once: true });
        });
    }

    async function sendWs(message) {
        await ensureSession();
        ws.send(JSON.stringify(message));
        return await new Promise((resolve, reject) => {
            ws.addEventListener("message", (event) => {
                const payload = JSON.parse(event.data);
                if (payload.type === "error") {
                    reject(new Error(payload.data?.message || "Unknown websocket error"));
                    return;
                }
                resolve(payload);
            }, { once: true });
        });
    }

    async function resetSession(taskType) {
        const response = await sendWs({ type: "reset", data: { task_type: taskType } });
        updateUiFromObservation(response.data);
        return response.data;
    }

    async function requestAgentAction(observation) {
        const response = await fetch("/demo/act", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                task_prompt: observation.task_prompt,
                current_solution: observation.current_solution,
                task_type: observation.task_type,
                current_strategy: observation.current_strategy || "",
                recent_failures: observation.recent_failures || [],
                downstream_scores: observation.downstream_scores || {}
            })
        });
        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            const detail = errorPayload.detail;
            if (detail && detail.error) {
                throw new Error(`${detail.error}: ${detail.reason || 'unknown'}`);
            }
            throw new Error(errorPayload.detail || `Demo act failed: ${response.status}`);
        }
        const actionResult = await response.json();

        if (actionResult.is_llm_generated) {
            logSystem(`[LLM] Action generated by ${actionResult.agent_provider || 'unknown'}`);
        } else {
            logSystem(`[DETERMINISTIC] Fallback action: ${actionResult.agent_error || 'no LLM configured'}`);
        }

        if (actionResult.strategy_patch) {
            const patchSource = actionResult.is_llm_generated ? "LLM" : "HEURISTIC";
            logSystem(`[${patchSource}] Strategy patch proposed: ${actionResult.strategy_patch.diff_description?.substring(0, 60) || 'n/a'}...`);
        }

        // Return only the action fields expected by the WebSocket /step endpoint
        // Filter out diagnostic fields (agent_source, agent_provider, agent_error, is_llm_generated)
        return {
            solution: actionResult.solution,
            edit_type: actionResult.edit_type,
            strategy_note: actionResult.strategy_note,
            strategy_patch: actionResult.strategy_patch
        };
    }

    btnReset.onclick = async () => {
        try {
            logSystem("Resetting environment...");
            await resetSession(taskSelect.value);
        } catch (error) {
            logError(`Reset failed: ${error.message}`);
        }
    };

    btnRun.onclick = async () => {
        const task = taskSelect.value;
        btnRun.disabled = true;
        btnRun.innerText = "RUNNING...";

        try {
            await resetSession(task);
            await new Promise((resolve) => setTimeout(resolve, 500));

            const maxSteps = task === "strategy_optimization" ? 10 : 6;
            logSystem(`Running improvement loop (max ${maxSteps} steps)...`);

            for (let step = 1; step <= maxSteps; step += 1) {
                logSystem(`Agent step ${step}/${maxSteps} - requesting next action...`);
                btnRun.innerText = `STEP ${step}/${maxSteps}...`;

                const action = await requestAgentAction(currentObservation);
                const stepResponse = await sendWs({ type: "step", data: action });
                
                // Include the action in the step response for UI rendering
                stepResponse.data.action = action;
                updateUiFromObservation(stepResponse.data);

                await new Promise((resolve) => setTimeout(resolve, 300));
                if (stepResponse.data.done) {
                    logSystem("Episode complete.");
                    break;
                }
            }

            logSystem("Agent trajectory complete.");
        } catch (error) {
            logError(`Run failed: ${error.message || error}`);
        }

        btnRun.innerText = "RUN";
        btnRun.disabled = false;
    };
});
