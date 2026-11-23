# HackFusion 4.0 — 3-Minute Video Demo Script
**System:** Shikayat Sarathi (Agentic AI for Intelligent Grievance Redressal, Nanded District)

---

## Overview for Judges & Evaluators
This 3-minute live demonstration walks through the three required scenarios mandated by HackFusion 4.0:
1. **Scenario A (Routine Auto-Handled Grievance):** Marathi/English complaint processed with full RAG retrieval, policy gate pass (`RULE_06_AUTO_HANDLE_PERMITTED`), mock work-order dispatch, simulated citizen notifications, and final closure only upon citizen confirmation.
2. **Scenario B (Sensitive Safety Lock):** Women/child safety grievance intercepted by Semantic Safety Guard and Category Master. Deterministic lockdown (`RULE_01_SENSITIVE_CATEGORY_HOLD`), identity masking, restricted escalation webhook (no raw text), and strict prohibition of automated closure.
3. **Scenario C (Duplicate Detection & Predictive SLA):** Multiple complaints for the same incident grouped into a suggested Incident cluster without silent merges. Both tickets retain unique IDs and history. Officer confirms cluster membership.

---

## Minute 00:00 – 01:00 | Scenario A: Routine Auto-Handling & Citizen Confirmation Gate
* **Citizen Input (Marathi/English):**
  > "Streetlight not working on pole 12 in Vazirabad Nanded since two days"
* **System Actions Demonstrated:**
  1. **Intake Agent:** Extracts problem description, taluka (`Nanded`), and locality (`Vazirabad`). Recognizes category `C009: Street Lighting and Electricity - Streetlight not working`.
  2. **Vector/BM25 RAG Retrieval:** Fetches 5 nearest precedents from 800-case history. Retrieves strong historical resolutions (`retrieval_quality: STRONG`, similarity ~0.85).
  3. **Deterministic Policy Gate:** Checks Category Master runtime policy.
     - `sensitive == False`
     - `auto_resolution_allowed == True`
     - `fused_confidence >= 0.75`
     - Verdict: `AUTO_HANDLE` under rule `RULE_06_AUTO_HANDLE_PERMITTED`.
  4. **Action Agent Execution:**
     - Dispatches Work Order `WO-GRV-...` to Electricity Distribution Department mock webhook `MOCK-WEBHOOK-D005`.
     - Issues simulated Email (`citizen-...@example.invalid`) and WhatsApp notifications with status `SIMULATED`.
     - Sets grievance status to `IN_PROGRESS`.
  5. **Officer Resolution & Citizen Confirmation Gate:**
     - Duty Officer marks status `RESOLVED`.
     - Citizen logs in and triggers `/confirm-resolution` -> Status transitions to `CITIZEN_SATISFIED`.

* **Expected Trace Spans in Langfuse:**
  - `Intake_Turn`
  - `Vector_RAG_Retrieval` (precedents, similarity)
  - `Classification_Confidence` (components)
  - `Policy_Agent_Master_Gate` (verdict: `AUTO_HANDLE`, rule: `RULE_06`)
  - `Webhook_Notification_Action` (dispatched work order)

---

## Minute 01:00 – 02:00 | Scenario B: Sensitive Safety Lock & Zero-Leakage Guarantee
* **Citizen Input (English/Hindi):**
  > "Some boys are eve teasing college girls near bus stand every evening"
* **System Actions Demonstrated:**
  1. **Semantic Safety Guard Interception:**
     - Identifies high-risk concept: `Women Safety / Harassment`.
     - Forces immediate identity protection (`identity_mask: True`).
  2. **Deterministic Policy Gate:**
     - Category `C017` is designated as `sensitive: True` in Category Master.
     - Verdict: `SENSITIVE_HOLD` under `RULE_01_SENSITIVE_CATEGORY_HOLD`.
     - `can_auto_resolve: False` permanently locked.
  3. **Action Agent Execution:**
     - Creates **restricted** escalation work order: complaint raw text and citizen identity are strictly stripped from payload.
     - Citizen notifications are discreet: no disclosure of sensitive details.
     - State moved to `SENSITIVE_HOLD` (Restricted Officer Queue).
  4. **Attempted Unauthorized Closure (Forbidden Invariant):**
     - Demo attempting to PATCH status to `RESOLVED` via `AI_SYSTEM` -> **HTTP 403 Forbidden**.
     - Demo attempting citizen auto-confirmation -> **HTTP 403 Forbidden**.
     - ONLY a verified human officer (`role=OFFICER`) can inspect and progress this grievance.

* **Expected Trace Spans in Langfuse:**
  - `Vector_RAG_Retrieval` (raw text redacted, length logged only)
  - `Policy_Agent_Master_Gate` (verdict: `SENSITIVE_HOLD`, identity_masked: True)
  - `Webhook_Notification_Action` (restricted escalation)

---

## Minute 02:00 – 03:00 | Scenario C: Duplicate Clustering & Predictive SLA Breach Alerts
* **Citizen Inputs:**
  > **Citizen 1:** "Drinking water pipeline burst near Shivaji statue, water flooding the road in Vazirabad"
  > **Citizen 2:** "Drinking water pipeline leakage and flooding near Shivaji statue in Vazirabad"
* **System Actions Demonstrated:**
  1. **Zero Silent Merges:** Both submissions receive independent reference numbers (`GRV-...-1` and `GRV-...-2`) and independent status histories.
  2. **Suggested Incident Cluster:** Duplicate clustering engine compares semantic text, taluka (`Nanded`), and locality (`Vazirabad`).
     - Detects incident cluster and creates `Incident #... (Status: SUGGESTED)`.
     - Attaches both complaints as suggested members without merging or dropping tickets.
  3. **Officer Review Portal:**
     - Officer views the incident cluster.
     - Clicks "Confirm Member" -> Member status updates to `CONFIRMED` by officer.
  4. **Predictive SLA Breach Intelligence:**
     - Background SLA engine calculates survival estimate and duration hazard against Category `C003` SLA (3 days).
     - Identifies high probability of breach due to pipeline backlog in Nanded.
     - Pre-emptively raises an `ACTIVE` alert with root cause explanation.
