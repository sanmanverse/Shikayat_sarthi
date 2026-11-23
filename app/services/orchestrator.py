"""
End-to-End Grievance Orchestrator.
Groups 6 & 11 (+ v4 hardening):
- Order: intake → two-stage safety → hybrid retrieval → (guarded agent loop) → confidence → policy gate → action
- Safety is two-stage (regex + LLM) and any sensitive signal overrides the proposed category using the
  Category master (no hard-coded IDs); a sensitive-precedent proximity net backs it up
- The agent loop (only when a Sarvam key is configured) proposes; the deterministic gate decides
- Compute identity masking BEFORE trace creation
- Real confidence formula (no +0.10 constant, no 0.85 default)
- When LLM confidence unavailable: confidence=null, force HUMAN_REVIEW
- Fail closed on AI step failure (store + route to HUMAN_REVIEW)
- Pass trace_id to action agent for AuditLog
- Flush Langfuse after each pipeline run
"""
import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.models import Grievance, Category, AuditLog, StatusHistory
from app.core.config import settings
from app.nlp.retrieval import retrieval_engine
from app.services.safety_assessment import assess as assess_safety, resolve_category_for_concept
from app.services.policy_gate import evaluate_policy_gate
from app.services.action_agent import action_agent
from app.services.agent_loop import GuardedAgentLoop, ToolSpec
from app.services.sarvam import sarvam_client
from app.core.observability import tracer

logger = logging.getLogger(__name__)


def calculate_fused_confidence(
    llm_conf: Optional[float],
    top_similarity: float,
    neighbor_agreement: float,
) -> Optional[float]:
    """
    Groups 6: No fabricated numbers.
    Formula: llm_conf * 0.40 + neighbor_agreement * 0.30 + top_similarity * 0.20
    Returns None when llm_conf is unavailable — policy gate will force HUMAN_REVIEW.
    No base constant (+0.10 removed), no default 0.85.
    """
    if llm_conf is None:
        return None
    fused = (0.40 * llm_conf) + (0.30 * neighbor_agreement) + (0.20 * top_similarity)
    return round(min(max(fused, 0.0), 1.0), 3)


class GrievanceOrchestrator:
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_protected(cat: Optional[Category]) -> bool:
        return bool(cat and (cat.sensitive or cat.identity_hiding))

    def _build_agent(self, db: Session):
        """Read-only tools for the agent. No tool can write state."""
        def search_similar_cases(query: str, top_k: int = 5):
            k = max(1, min(int(top_k), 8))
            res = retrieval_engine.search(query=query, db=db, top_k=k, raw_query=query)
            return {
                "retrieval_quality": res["retrieval_quality"],
                "cases": [
                    {
                        "id": n["grievance_id"], "category_id": n["category_id"],
                        "subcategory": n["subcategory"], "similarity": n["similarity"],
                        "sensitive": bool(n["sensitive"]),
                        "resolution": (n["resolution_text"] or "")[:140],
                    }
                    for n in res["neighbors"]
                ],
            }

        def get_category_policy(category_id: str):
            c = db.query(Category).filter(Category.category_id == category_id.strip().upper()).first()
            if not c:
                return {"error": "unknown category_id"}
            return {
                "category_id": c.category_id, "category": c.category, "subcategory": c.subcategory,
                "sla_days": c.sla_days, "sensitive": bool(c.sensitive),
                "identity_hiding": bool(c.identity_hiding),
                "auto_resolution_allowed": bool(c.auto_resolution_allowed),
            }

        def find_categories(keyword: str):
            kw = keyword.strip().lower()
            rows = [
                c for c in db.query(Category).all()
                if kw and (kw in (c.category or "").lower() or kw in (c.subcategory or "").lower())
            ][:8]
            return [
                {"category_id": c.category_id, "category": c.category, "subcategory": c.subcategory,
                 "sensitive": bool(c.sensitive)}
                for c in rows
            ]

        tools = [
            ToolSpec(
                "search_similar_cases",
                "Search past grievances (any language). Use your own wording to test alternative categories.",
                {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                 "required": ["query"]},
                search_similar_cases,
            ),
            ToolSpec(
                "get_category_policy",
                "Look up one Category-master entry (SLA and safety flags).",
                {"type": "object", "properties": {"category_id": {"type": "string"}}, "required": ["category_id"]},
                get_category_policy,
            ),
            ToolSpec(
                "find_categories",
                "List Category-master entries whose name contains a keyword (English).",
                {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]},
                find_categories,
            ),
        ]
        return GuardedAgentLoop(
            llm_chat=lambda msgs, defs: sarvam_client.chat_with_tools(msgs, defs),
            tools=tools,
            category_exists=lambda cid: db.query(Category).filter(Category.category_id == cid).first() is not None,
            max_steps=settings.AGENT_MAX_STEPS,
        )

    # ------------------------------------------------------------------ pipeline
    def process_grievance(
        self,
        grievance: Grievance,
        db: Session,
        llm_confidence: Optional[float] = None,  # None by default — no fabricated value
    ) -> Dict[str, Any]:
        """
        Runs the full pipeline for a confirmed grievance with full tracing.
        Pipeline: Safety (2-stage) → Retrieval (hybrid) → [Agent proposal] → Confidence → Policy Gate → Action
        """
        # 0. Two-stage safety — BEFORE trace creation so the identity mask is known
        safety = assess_safety(grievance.raw_text)
        is_identity_masked = safety.is_sensitive or safety.identity_mask
        if is_identity_masked:
            grievance.identity_mask = True

        trace_id = tracer.create_trace(
            reference_no=grievance.reference_no,
            citizen_id=grievance.citizen_id,
            is_identity_masked=is_identity_masked,
        )
        is_sensitive_text = safety.is_sensitive

        tracer.log_span(
            trace_id=trace_id,
            name="Safety_Check",
            input_data={"complaint_text": grievance.raw_text},
            output_data={
                "stage_a_sensitive": safety.stage_a_sensitive,
                "stage_b_status": safety.stage_b_status,
                "stage_b_sensitive": safety.stage_b_sensitive,
                "stage_b_confidence": safety.stage_b_confidence,
                "disagreement": safety.disagreement,
                "is_sensitive": safety.is_sensitive,
                "concept": safety.detected_concept,
                "note": (
                    "Stage B (LLM) not configured: only the regex stage and retrieval proximity guard this text."
                    if safety.stage_b_status == "not_configured" else None
                ),
            },
            is_identity_masked=is_identity_masked,
            is_sensitive_text=is_sensitive_text,
        )

        agent_summary: Dict[str, Any] = {"mode": "not_run", "reason": None}

        try:
            # 2. Hybrid RAG retrieval (embeddings + BM25; English-normalised query for BM25)
            search_query = grievance.normalized_text or grievance.raw_text
            search_results = retrieval_engine.search(
                query=search_query,
                db=db,
                top_k=5,
                taluka_filter=grievance.taluka,
                raw_query=grievance.raw_text,
            )

            neighbors = search_results["neighbors"]
            sensitive_neighbor_count = search_results["sensitive_neighbor_count"]
            retrieval_quality = search_results["retrieval_quality"]
            top_vote = search_results["top_category_vote"]
            sensitive_best = search_results.get("sensitive_best")

            tracer.log_span(
                trace_id=trace_id,
                name="Vector_RAG_Retrieval",
                input_data={"query": search_query, "taluka_filter": grievance.taluka},
                output_data={
                    "retrieval_mode": search_results.get("retrieval_mode"),
                    "retrieval_quality": retrieval_quality,
                    "sensitive_neighbors_found": sensitive_neighbor_count,
                    "sensitive_best": sensitive_best,
                    "top_precedents": [
                        {"id": n["grievance_id"], "sim": n["similarity"], "category": n["category_id"],
                         "source": n.get("similarity_source")}
                        for n in neighbors[:3]
                    ],
                },
                is_identity_masked=is_identity_masked,
                is_sensitive_text=is_sensitive_text,
            )

            # 3. Category determination (master is the sole source of truth)
            intake_cat_id = grievance.category_id
            cat_id = intake_cat_id or top_vote
            category = db.query(Category).filter(Category.category_id == cat_id).first() if cat_id else None
            safety_override_note = None

            if safety.is_sensitive and not self._is_protected(category):
                # A safety signal beats whatever category the intake model or retrieval proposed.
                # Preference: a protected category already proposed by intake/retrieval, else the master
                # row that matches the detected concept. Resolved from the master at runtime.
                protected = None
                for cand_id in (intake_cat_id, top_vote):
                    row = db.query(Category).filter(Category.category_id == cand_id).first() if cand_id else None
                    if self._is_protected(row):
                        protected = row
                        break
                if protected is None:
                    protected = resolve_category_for_concept(
                        safety.detected_concept,
                        db.query(Category).filter((Category.sensitive == True) | (Category.identity_hiding == True)).all(),  # noqa: E712
                    )
                if protected is not None:
                    safety_override_note = (
                        f"Safety signal '{safety.detected_concept}' overrode proposed category "
                        f"{cat_id or 'none'} -> {protected.category_id} (resolved from Category master)."
                    )
                    cat_id, category = protected.category_id, protected
                    grievance.category_id = protected.category_id
                    db.add(AuditLog(
                        actor="AI_ORCHESTRATOR", agent_name="SafetyOverride", model_name="deterministic",
                        tool_action="override_category_by_safety_signal",
                        affected_record=grievance.reference_no,
                        policy_rule="RULE_01_SENSITIVE_CATEGORY_HOLD",
                        summary=safety_override_note, trace_id=trace_id,
                    ))

            # 3b. Guarded agent loop (real Sarvam key only; skipped for sensitive text, which is held anyway)
            agent_escalation: Optional[str] = None
            effective_llm_conf = llm_confidence
            if settings.AGENT_LOOP_ENABLED and sarvam_client.is_configured and not safety.is_sensitive:
                agent = self._build_agent(db).run(
                    complaint_text=grievance.raw_text,
                    context={
                        "intake_category_id": intake_cat_id,
                        "baseline_retrieval": {
                            "quality": retrieval_quality, "top_category_vote": top_vote,
                            "top3": [{"category_id": n["category_id"], "similarity": n["similarity"]} for n in neighbors[:3]],
                        },
                        "safety": {"stage_a": safety.stage_a_sensitive, "stage_b": safety.stage_b_status},
                    },
                )
                agent_summary = {
                    "mode": agent.mode, "steps": agent.steps, "guard_notes": agent.guard_notes,
                    "proposal": {
                        "category_id": agent.category_id, "confidence": agent.confidence,
                        "escalate_to_human": agent.escalate_to_human, "rationale": agent.rationale,
                    },
                    "error": agent.error,
                }
                if agent.mode == "agentic":
                    if agent.category_id:
                        a_row = db.query(Category).filter(Category.category_id == agent.category_id).first()
                        if self._is_protected(a_row):
                            cat_id, category = a_row.category_id, a_row      # protected proposal always wins
                            grievance.category_id = a_row.category_id
                        elif intake_cat_id and agent.category_id != intake_cat_id:
                            agent_escalation = (
                                f"category disagreement: intake proposed {intake_cat_id}, agent proposed {agent.category_id}"
                            )
                        elif not intake_cat_id:
                            cat_id, category = a_row.category_id, a_row
                    if agent.escalate_to_human and not agent_escalation:
                        agent_escalation = agent.rationale or "agent requested human review"
                    confs = [c for c in (llm_confidence, agent.confidence) if c is not None]
                    effective_llm_conf = min(confs) if confs else None     # conservative: lowest wins
                tracer.log_span(
                    trace_id=trace_id,
                    name="Triage_Agent_Loop",
                    input_data={"intake_category_id": intake_cat_id},
                    output_data={k: v for k, v in agent_summary.items() if k != "steps"} | {
                        "tool_calls": [f"{x.get('tool')}{'' if x.get('ok') else ' (rejected)'}" for x in agent.steps],
                    },
                    is_identity_masked=is_identity_masked,
                )
            else:
                agent_summary = {
                    "mode": "not_run",
                    "reason": (
                        "sensitive text: held by policy" if safety.is_sensitive
                        else "agent loop disabled" if not settings.AGENT_LOOP_ENABLED
                        else "no SARVAM_API_KEY: deterministic pipeline"
                    ),
                }

            # 3c. Sensitive-precedent proximity net (independent of the text guard and of the proposed category)
            proximity: Dict[str, Any] = {"hit": False}
            if sensitive_best and not self._is_protected(category) and sensitive_best["category_id"] != cat_id:
                if sensitive_best.get("similarity_source") == "embedding_cosine":
                    near = sensitive_best["gap"] <= settings.SENSITIVE_PROXIMITY_GAP
                else:
                    near = sensitive_best["ratio"] >= settings.SENSITIVE_PROXIMITY_RATIO
                if near:
                    proximity = {"hit": True, **sensitive_best}

            # 4. Confidence fusion (no fabricated numbers)
            top_sim = neighbors[0]["similarity"] if neighbors else 0.0
            neighbor_agreement = (
                sum(1 for n in neighbors if n["category_id"] == cat_id) / len(neighbors)
                if neighbors and cat_id else 0.0
            )
            fused_confidence = calculate_fused_confidence(effective_llm_conf, top_sim, neighbor_agreement)
            llm_confidence_missing = effective_llm_conf is None

            tracer.log_span(
                trace_id=trace_id,
                name="Classification_Confidence",
                input_data={"cat_id": cat_id, "llm_confidence_missing": llm_confidence_missing},
                output_data={
                    "fused_confidence": fused_confidence,
                    "llm_conf": effective_llm_conf,
                    "neighbor_agreement": neighbor_agreement,
                    "top_similarity": top_sim,
                    "safety_override": safety_override_note,
                },
                is_identity_masked=is_identity_masked,
            )

            # 5. Policy gate (deterministic; the agent can only make it stricter)
            decision = evaluate_policy_gate(
                category=category,
                fused_confidence=fused_confidence if fused_confidence is not None else 0.0,
                retrieval_quality=retrieval_quality,
                sensitive_neighbor_count=sensitive_neighbor_count,
                complaint_text=grievance.raw_text,
                llm_confidence=effective_llm_conf,
                top_similarity=top_sim,
                neighbor_agreement=neighbor_agreement,
                safety_result=safety,
                sensitive_proximity=proximity,
                agent_escalation=agent_escalation,
            )

            tracer.log_span(
                trace_id=trace_id,
                name="Policy_Agent_Master_Gate",
                input_data={
                    "category_id": cat_id,
                    "fused_confidence": fused_confidence,
                    "retrieval_quality": retrieval_quality,
                    "sensitive_neighbors": sensitive_neighbor_count,
                    "sensitive_proximity": proximity,
                    "agent_escalation": agent_escalation,
                    "llm_confidence_missing": llm_confidence_missing,
                },
                output_data={
                    "verdict": decision.verdict,
                    "rule_id": decision.rule_id,
                    "can_auto_resolve": decision.can_auto_resolve,
                    "identity_masked": decision.identity_mask,
                    "reason": decision.reason,
                },
                is_identity_masked=decision.identity_mask,
                is_sensitive_text=is_sensitive_text,
            )

            # 6. Action execution
            action_result = action_agent.execute_workflow(
                grievance=grievance,
                category=category,
                decision=decision,
                db=db,
                trace_id=trace_id,
            )

            tracer.log_span(
                trace_id=trace_id,
                name="Webhook_Notification_Action",
                input_data={"verdict": decision.verdict, "status": grievance.status},
                output_data={"actions_executed": action_result["actions_taken"]},
                is_identity_masked=decision.identity_mask,
            )

        except Exception as exc:
            # Fail closed: store grievance, route to HUMAN_REVIEW, audit
            logger.error("Orchestrator pipeline failure: %s", exc, exc_info=True)
            try:
                grievance.status = "TRIAGED"
                grievance.policy_verdict = "HUMAN_REVIEW"
                grievance.policy_rule = "RULE_PIPELINE_FAILURE"
                db.add(StatusHistory(
                    grievance_id=grievance.reference_no,
                    from_status="RECEIVED",
                    to_status="TRIAGED",
                    actor="AI_ORCHESTRATOR",
                    note=f"Pipeline error: routed to human review. Error: {str(exc)[:200]}",
                ))
                db.add(AuditLog(
                    actor="AI_ORCHESTRATOR",
                    agent_name="GrievanceOrchestrator",
                    model_name="system_error",
                    tool_action="process_grievance_failure",
                    affected_record=grievance.reference_no,
                    summary=f"Pipeline failure, fail-closed to HUMAN_REVIEW. Error: {str(exc)[:200]}",
                    trace_id=trace_id,
                ))
                db.commit()
            except Exception:
                pass
            tracer.finalize(trace_id, public=False, output={"verdict": "HUMAN_REVIEW", "rule_id": "RULE_PIPELINE_FAILURE"})
            tracer.flush()
            return {
                "reference_no": grievance.reference_no,
                "status": "TRIAGED",
                "verdict": "HUMAN_REVIEW",
                "rule_id": "RULE_PIPELINE_FAILURE",
                "public_trace_link": None,
                "trace_visibility": tracer.visibility(trace_id),
                "error": "Pipeline error — routed to human review",
            }

        # Trace visibility is decided AFTER the policy decision: only non-sensitive, non-masked cases are public.
        protected_case = (
            decision.verdict == "SENSITIVE_HOLD" or decision.identity_mask or is_identity_masked
        )
        make_public = (
            settings.TRACE_PUBLIC_FOR_SENSITIVE if protected_case else settings.TRACE_PUBLIC_NON_SENSITIVE
        )
        tracer.finalize(
            trace_id,
            public=make_public,
            output={"verdict": decision.verdict, "rule_id": decision.rule_id, "agent_mode": agent_summary.get("mode")},
        )
        tracer.flush()
        public_trace_link = tracer.get_public_trace_link(trace_id)
        visibility = tracer.visibility(trace_id)

        return {
            "reference_no": grievance.reference_no,
            "status": grievance.status,
            "trace_id": trace_id,
            "public_trace_link": public_trace_link,
            "trace_visibility": visibility,
            "langfuse_note": (
                None if public_trace_link
                else "Tracing disabled: set LANGFUSE_* keys in .env" if visibility == "disabled"
                else "Restricted trace (sensitive or identity-masked case): visible to Langfuse project members only."
            ),
            "category": {
                "category_id": category.category_id if category else None,
                "category": category.category if category else None,
                "subcategory": category.subcategory if category else None,
                "sla_days": category.sla_days if category else None,
                "sensitive_flag": category.sensitive if category else False,
                "identity_hiding_flag": category.identity_hiding if category else False,
            },
            "safety": {
                "stage_a_sensitive": safety.stage_a_sensitive,
                "stage_b_status": safety.stage_b_status,
                "stage_b_sensitive": safety.stage_b_sensitive,
                "disagreement": safety.disagreement,
                "concept": safety.detected_concept,
                "category_override": safety_override_note,
            },
            "agent": {k: v for k, v in agent_summary.items() if k != "steps"} | {
                "tool_calls": [x.get("tool") for x in agent_summary.get("steps", [])],
            },
            "decision": {
                "verdict": decision.verdict,
                "rule_id": decision.rule_id,
                "rule_reason": decision.reason,
                "fused_confidence": fused_confidence,
                "llm_confidence_missing": llm_confidence_missing,
                "confidence_components": decision.confidence_components,
                "can_auto_resolve": decision.can_auto_resolve,
                "identity_masked": decision.identity_mask,
            },
            "retrieval_evidence": {
                "retrieval_mode": search_results.get("retrieval_mode"),
                "retrieval_quality": retrieval_quality,
                "similar_precedents": [
                    {
                        "grievance_id": n["grievance_id"],
                        "similarity": n["similarity"],
                        "similarity_source": n.get("similarity_source"),
                        "subcategory": n["subcategory"],
                        "locality": n["locality"],
                        "resolution_summary": n["resolution_text"],
                        "citizen_feedback": n["citizen_feedback"],
                        "sensitive": n["sensitive"],
                    }
                    for n in neighbors[:3]
                ],
            },
            "actions_executed": action_result["actions_taken"],
        }


orchestrator = GrievanceOrchestrator()
