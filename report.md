# Documentation Report — ML Development & Deployment Phase
**Task:** Dog vs Cat Image Classification using ResNet18 | **Instructor:** Abdul Muqtadar
**Dataset:** [Kaggle — Dog and Cat Classification](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)

---

## What is an API?
An **API (Application Programming Interface)** is a contract between software systems that defines how they communicate. In ML, an API exposes a trained model as a service — clients send input data (e.g., an image) over HTTP and receive predictions (e.g., "dog" or "cat") in JSON format, without needing access to the model code or weights. This enables any application, regardless of language, to use the model.

---

## What is the Difference Between Flask and FastAPI?
| Feature | Flask | FastAPI |
|---|---|---|
| Type | Micro web framework (sync) | Modern async framework |
| Performance | Moderate (synchronous) | High (async via uvicorn) |
| Data Validation | Manual | Automatic via Pydantic models |
| Auto Documentation | None (manual setup needed) | Built-in Swagger UI at `/docs` |
| Learning Curve | Very beginner-friendly | Moderate |
| Best Use Case | Quick prototypes & ML demos | Production-grade, scalable APIs |

**In this project** Flask was used because it is simpler to set up for an ML prototype, with fewer dependencies and a cleaner entry point.

---

## What is REST Framework for APIs?
**REST (Representational State Transfer)** is a set of architectural principles for designing HTTP-based APIs. Key rules:
- **Stateless**: Each request contains all information needed; the server stores no session state
- **HTTP Methods**: `GET` (read/check), `POST` (send data/predict), `PUT` (update), `DELETE` (remove)
- **Resources via URLs**: Each endpoint represents a resource (e.g., `/predict`, `/batch_predict`)
- **JSON responses**: Data is exchanged in lightweight, language-agnostic JSON format

In this project, `/predict` (POST) accepts an image file and returns a classification label, following REST principles.

---

## What are Microservices and How Do ML Developers Use Them?
**Microservices** is an architecture where an application is broken into small, independently deployable services — each responsible for one task. Contrast with a monolith where everything is one big app.

**ML developers** use microservices to isolate:
- A **training service** that fine-tunes models on new data
- An **inference service** (this project) that only handles predictions
- A **preprocessing service** that resizes and normalizes incoming images
- A **monitoring service** that tracks model accuracy drift over time

This allows teams to update the model without redeploying the whole system, and to scale the inference service independently during high-traffic periods.

---

## What is Docker and How Do ML Developers Use It?
**Docker** is a platform for packaging an application and all its dependencies (Python version, libraries, model weights) into a **container** — an isolated, portable unit that runs identically on any machine.

ML developers use Docker to:
- Eliminate environment mismatches ("works on my laptop, fails on the server")
- Ship a model along with its exact Python, CUDA, and library versions
- Deploy containers to any cloud provider (AWS ECS, Google Cloud Run, Azure ACI) without reconfiguration
- Reproduce experimental environments months later for audit or debugging

---

## What is a Docker File and Why Do We Create It?
A **Dockerfile** is a plain-text recipe that tells Docker how to build a container image step by step: which base OS/language to use, which files to copy, which packages to install, and which command to run on startup.

We create a Dockerfile because:
- It makes the build **fully reproducible** from a single file
- It **automates** environment setup — no manual installation on each server
- It allows the image to be **version-controlled** in Git alongside the code
- It enables **CI/CD pipelines** to automatically build and push new images when code changes

---

## What is the Difference Between a Docker Image and a Docker Container?

| | Docker Image | Docker Container |
|---|---|---|
| **Definition** | A read-only, static blueprint | A running instance created from an image |
| **State** | Frozen (immutable) | Live (has memory, processes, I/O) |
| **Analogy** | A class in OOP | An object (instance of the class) |
| **Created by** | `docker build -t myapp .` | `docker run -p 5000:5000 myapp` |
| **Storage** | Stored on disk / Docker Hub | Exists only while running |
| **Reusability** | One image → many containers | One container = one running process |

**In this project:** `docker build` creates the image containing ResNet18 + Flask. `docker run` starts a container from it, which serves predictions on port 5000.

---

## Evaluation Metrics Used
**Metrics:** Accuracy & Weighted F1-Score

| Metric | Formula | Why Used |
|---|---|---|
| **Accuracy** | Correct predictions / Total | Simple overall correctness measure |
| **Weighted F1** | Harmonic mean of Precision & Recall (weighted by class count) | Handles slight class imbalance between cats and dogs |

F1-Score is preferred over accuracy alone because it penalizes models that simply predict the majority class, giving a more honest picture of performance on both cat and dog classes.

---
*Model: ResNet18 (ImageNet pretrained, fine-tuned) | Framework: PyTorch + Flask | Deployment: Docker*
