from fastapi import FastAPI
from fastapi.testclient import TestClient
from app import app
import time
from test_data import accepted_rejected_data, joint_sample, single_sample

client = TestClient(app)


def test_accepted_rejected(data):
    t0 = time.time_ns() // 1_000_000
    response = client.post("/loan_acceptance", json=data)
    t1 = time.time_ns() // 1_000_000
    print(f"Time taken: {t1-t0} ms")
    print(response.status_code)
    print(response.json())


def test_single(data):
    t0 = time.time_ns() // 1_000_000
    response = client.post("/loan_quality", json=data)
    t1 = time.time_ns() // 1_000_000
    print(f"Time taken: {t1-t0} ms")
    print(response.status_code)
    print(response.json())


def test_joint(data):
    t0 = time.time_ns() // 1_000_000
    response = client.post("/loan_quality", json=data)
    t1 = time.time_ns() // 1_000_000
    print(f"Time taken: {t1-t0} ms")
    print(response.status_code)
    print(response.json())


test_accepted_rejected(accepted_rejected_data)
test_single(single_sample)
test_joint(joint_sample)
