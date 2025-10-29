## Prompt Checklist: Real-Time Sign Language Recognition

### Project Definition

* [ ] **Problem:** Real-time sign language recognition.
* [ ] **Goal:** High accuracy, low latency, multi-model system.
* [ ] **Target Users:** Web users with camera input.

---

### Architecture

* [ ] **Backend:** FastAPI + Celery + BentoML (ML serving).
* [ ] **Frontend:** React + Clerk.com (authentication).
* [ ] **Storage:** Redis (cache), PostgreSQL (audit), local artifacts.

---

### Machine Learning Models

* [ ] **Static Signs:** CNN (TSM-ResNet50 / Inception V3) for single images.
* [ ] **Real-Time Detection:** MediaPipe Gesture Recognizer for landmarks.
* [ ] **Sequence Interpretation:** RNN / Transformer with TSM for dynamic gestures.

---

### API Endpoints

* [ ] `/v1/inference/letter` — classify a static gesture.
* [ ] `/v1/stream/hands` — detect and track hands in real-time.
* [ ] `/v1/inference/sequence` — translate gesture sequences to words.
* [ ] `/v1/video/upload` — start async video processing.
* [ ] `/v1/task/{id}/status` — get task progress and results.

---

### Latency & Performance Targets

* [ ] Static inference: **< 100 ms**.
* [ ] Stream detection: **< 50 ms**.
* [ ] Sequence inference: **< 500 ms**.
* [ ] Async pipeline (Celery): reliable & scalable.
* [ ] Frontend: responsive UI with progress feedback.

---

### Success Criteria

* [ ] Three functional APIs with validation & docs.
* [ ] ML models deployed via BentoML containers.
* [ ] React frontend integrated with Clerk.com.
* [ ] Redis task tracking with real-time updates.
* [ ] End-to-end latency benchmarks achieved.
