"""
Demo Seeding Script for HackFusion 4.0 Evaluators.
Submits the 3 mandatory demo scenarios through the public API:
1. Scenario A: Routine auto-handled grievance (Streetlight, C009)
2. Scenario B: Sensitive safety locked grievance (Women Safety, C017)
3. Scenario C: Duplicate complaints linked into suggested Incident (Water pipeline, C003)
"""
import sys
import requests

BASE_URL = "http://localhost:8000/api"

def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print("Backend is online:", r.json())
        return True
    except Exception as e:
        print(f"Could not connect to backend at {BASE_URL}: {e}")
        print("Please start the server first: uvicorn app.main:app --port 8000")
        return False

def seed_scenario_a():
    print("\n--- Seeding Scenario A (Routine Auto-Handled: Streetlight C009) ---")
    draft = {
        "category_id": "C009",
        "category_name": "Street Lighting and Electricity - Streetlight not working",
        "description": "Streetlight pole 14 in Vazirabad is dark since Monday night",
        "taluka": "Nanded",
        "locality": "Vazirabad",
        "urgency": "MEDIUM",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.90
    }
    r = requests.post(f"{BASE_URL}/intake/confirm", json={"draft": draft, "citizen_id": "citizen_demo_a"})
    res = r.json()
    print(f"Grievance created: {res.get('reference_no')}")
    print(f"Policy Verdict: {res.get('decision', {}).get('verdict')} ({res.get('decision', {}).get('rule_id')})")
    print(f"Status: {res.get('status')}")

def seed_scenario_b():
    print("\n--- Seeding Scenario B (Sensitive Hold: Women Safety C017) ---")
    draft = {
        "category_id": "C017",
        "category_name": "Women and Child Welfare - Women safety / harassment concern",
        "description": "Some boys are eve teasing college girls near bus stand every evening",
        "taluka": "Nanded",
        "locality": "Bus Stand Area",
        "urgency": "CRITICAL",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.95
    }
    r = requests.post(f"{BASE_URL}/intake/confirm", json={"draft": draft, "citizen_id": "citizen_demo_b"})
    res = r.json()
    print(f"Grievance created: {res.get('reference_no')}")
    print(f"Policy Verdict: {res.get('decision', {}).get('verdict')} ({res.get('decision', {}).get('rule_id')})")
    print(f"Status: {res.get('status')} | Identity Masked: {res.get('decision', {}).get('identity_masked')}")

def seed_scenario_c():
    print("\n--- Seeding Scenario C (Duplicate Incident Grouping: Water C003) ---")
    d1 = {
        "category_id": "C003",
        "category_name": "Water Supply - Pipeline leakage",
        "description": "Drinking water pipeline burst near Shivaji statue, water flooding the road",
        "taluka": "Nanded",
        "locality": "Vazirabad",
        "urgency": "HIGH",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.90
    }
    r1 = requests.post(f"{BASE_URL}/intake/confirm", json={"draft": d1, "citizen_id": "citizen_c1"}).json()
    print(f"Complaint 1: {r1.get('reference_no')}")

    d2 = {
        "category_id": "C003",
        "category_name": "Water Supply - Pipeline leakage",
        "description": "Drinking water pipeline leakage and flooding near Shivaji statue",
        "taluka": "Nanded",
        "locality": "Vazirabad",
        "urgency": "HIGH",
        "language": "English",
        "is_ready_for_confirmation": True,
        "confidence": 0.90
    }
    r2 = requests.post(f"{BASE_URL}/intake/confirm", json={"draft": d2, "citizen_id": "citizen_c2"}).json()
    print(f"Complaint 2: {r2.get('reference_no')}")
    if "duplicate_grouping" in r2:
        print(f"Incident Cluster Detected: #{r2['duplicate_grouping']['incident_id']}")

if __name__ == "__main__":
    if check_health():
        seed_scenario_a()
        seed_scenario_b()
        seed_scenario_c()
        print("\nAll demo scenarios seeded successfully!")
    else:
        sys.exit(1)
