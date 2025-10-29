## Project Overview: Real-Time Sign Language Recognition System

### 1. Overview

The project aims to develop a **multi-component AI system** for **real-time recognition and interpretation of sign language gestures** through video or camera input. It combines optimized **deep learning models**, **fast API serving**, and a **modern web client** for seamless user interaction and accessibility.

---

### 2. Core Objectives

1. **Deliver Low-Latency Inference:** < 100 ms for static gestures; < 50 ms for hand detection.
2. **Ensure Scalable Architecture:** Asynchronous task management (Celery + Redis) for video processing.
3. **Provide Reliable Multi-Model Design:** Separate models for static, dynamic, and sequential gestures.
4. **Offer Frontend Interactivity & Auth:** React + Clerk.com integration.
5. **Deliver Developer-Friendly APIs:** Documented REST endpoints with validation.

---

### 3. Tech Stack Summary

| **Category**        | **Key Technologies**                   | **Purpose**                                                   |
| ------------------- | -------------------------------------- | ------------------------------------------------------------- |
| **Backend & API**   | Python 3.10+, FastAPI, BentoML, Celery | Gateway for inference requests, ML serving, and orchestration |
| **Data Storage**    | Redis, PostgreSQL, Local               | Broker/cache (Redis), audit (PostgreSQL), artifacts (local)   |
| **Frontend & Auth** | React (TypeScript), Clerk.com          | Web UI + authentication                                       |
| **ML Frameworks**   | PyTorch / TensorFlow + MediaPipe       | Core inference & feature extraction                           |

---

### 4. Model Strategy

#### 4.1 Static Sign Recognition (`/v1/inference/letter`)

* **Goal:** Classify a single image into a sign letter.
* **Model:** TSM-ResNet50 / Inception V3 CNN.
* **Latency:** < 100 ms.

#### 4.2 Real-Time Hand Detection (`/v1/stream/hands`)

* **Goal:** Detect hands and output 21 landmarks.
* **Tech:** MediaPipe Gesture Recognizer.
* **Latency:** < 50 ms.

#### 4.3 Gesture Sequence Interpretation (`/v1/inference/sequence`)

* **Goal:** Convert gesture sequences into words/phrases.
* **Model:** RNN / Transformer with TSM.
* **Latency:** < 500 ms.

---

### 5. API Design

| **Endpoint**             | **Method** | **Function**             | **Latency**     | **Model**         | **Service** |
| ------------------------ | ---------- | ------------------------ | --------------- | ----------------- | ----------- |
| `/v1/inference/letter`   | POST       | Static sign recognition  | < 100 ms        | CNN               | BentoML     |
| `/v1/stream/hands`       | POST / WS  | Real-time hand landmarks | < 50 ms         | MediaPipe         | BentoML     |
| `/v1/inference/sequence` | POST       | Sequence translation     | < 500 ms        | RNN / Transformer | BentoML     |
| `/v1/video/upload`       | POST       | Async video processing   | Immediate (202) | Celery            | FastAPI     |
| `/v1/task/{id}/status`   | GET        | Task status polling      | Low             | —                 | FastAPI     |

---

### 6. Asynchronous Workflow

1. Upload video → `/v1/video/upload`.
2. FastAPI validates and stores.
3. Celery enqueues task → returns `task_id`.
4. React polls `/v1/task/{id}/status`.
5. Redis tracks task states (PENDING → PROGRESS → SUCCESS).

---

### 7. Deliverables & Success Criteria

* ✅ Three working APIs with docs.
* ✅ ML models deployed via BentoML.
* ✅ React frontend with auth.
* ✅ Redis-based async tracking.
* ✅ Latency targets achieved.
