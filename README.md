# Production-ready RAG Pipeline with Deployment

> **TL;DR**: A **scalable**, **containerized**, and **production-ready** Retrieval-Augmented Generation (RAG) pipeline built to handle multiple concurrent requests. This repository demonstrates how to deploy a stateless RAG system using **FastAPI** (backend), **Streamlit** (frontend), **Pinecone** (vector database), **Redis** (ephemeral session management), and **Kubernetes** on **AWS EKS** (infrastructure). Continuous Integration & Continuous Deployment (CI/CD) is configured using **GitHub Actions**.

---

## Overview

This repository showcases an **end-to-end** Retrieval-Augmented Generation (RAG) solution that focuses not only on the RAG methodology but, more importantly, on **production-level scalability**. The goal is to demonstrate how you can serve **thousands of users** concurrently, manage ephemeral sessions, and deploy multiple replicas of models seamlessly.

While the RAG technique itself is basic here, the overarching **design** and **deployment architecture** can be used in real-world scenarios where high availability, fault tolerance, and horizontal scalability are critical. If you’re looking to learn how to **deploy** a RAG pipeline—from code to container to Kubernetes—this repository is for you.

---

## Key Features

1. **Concurrent, Multi-User Support**  
   - Built with **FastAPI** and configured for high-throughput request handling.
   - Easily scalable through **Kubernetes** deployments.

2. **Stateless Design**  
   - All user conversations and uploaded files are ephemeral, stored in **Redis** or memory, ensuring minimal server state and simpler scaling.

3. **Basic RAG Methodology**  
   - Uses a standard approach to RAG, focusing on the architecture rather than advanced retrieval algorithms.

4. **Conversational Frontend (Streamlit)**  
   - A ChatGPT-like interface for user interactions.
   - Users can upload documents (PDF, DOCX) and immediately query them.

5. **Powerful Vector Storage (Pinecone)**  
   - Stores embeddings for quick retrieval.
   - Ideal for large-scale document indexing and semantic search.

6. **Containerization & Orchestration**  
   - All services (frontend, backend, Redis, etc.) are containerized using **Docker**.
   - Production deployment on **Kubernetes** (AWS EKS) for autoscaling and high availability.

7. **CI/CD with GitHub Actions**  
   - Automated build and deployment pipeline.
   - Streamlined code integration, testing, and rollout.

---

## Tech Stack

- **Frontend**:  
  - [Streamlit](https://streamlit.io/) – For the conversational UI.  
  - Session Management – Ephemeral chat sessions, no persistent user data.

- **Backend**:  
  - [FastAPI](https://fastapi.tiangolo.com/) – High-performance Python framework for handling requests and background tasks.  
  - [Redis](https://redis.io/) – In-memory store for ephemeral session data.  
  - [Pinecone](https://www.pinecone.io/) – Vector database for embeddings and semantic search.  
  - **LLM Inference** (Deepseek via Ollama/Hugging Face) – Basic language model inference for generating responses.  
  - [PyPDF2](https://pypi.org/project/PyPDF2/) & [python-docx](https://pypi.org/project/python-docx/) – Document loaders for PDF and DOCX.

- **Infrastructure & Deployment**:  
  - [Docker](https://www.docker.com/) – Containerization.  
  - [Kubernetes](https://kubernetes.io/) (AWS EKS) – Scaling and container orchestration.  
  - [GitHub Actions](https://github.com/features/actions) – CI/CD pipeline for automated deployments.

---

## Project Structure

```plaintext
project_root/
├── .dockerignore
├── .gitignore
├── README.md
├── docker-compose.yml
├── github/
│   └── workflows/
│       └── deploy.yaml
├── project_documents/
│   └── technologies used in the project.txt
├── backend/
│   ├── Dockerfile
│   ├── config.py
│   ├── config.yaml
│   ├── document_loader.py
│   ├── ilm.py
│   ├── logger_config.py
│   ├── main.py
│   ├── metrics.py
│   ├── prompt_manager.py
│   ├── redis_cache.py
│   ├── requirements.txt
│   ├── vector_store.py
│   └── c-cd/
│       └── github-actions.yml
├── frontend/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── k8s/
│   ├── backend/
│   │   ├── backend-deployment.yaml
│   │   └── backend-service.yaml
│   ├── frontend/
│   │   ├── frontend-deployment.yaml
│   │   └── frontend-service.yaml
│   └── ingress/
│       └── frontend-ingress.yaml
└── redis/
    ├── ebs-sc.yaml
    ├── redis-deployment.yaml
    ├── redis-pvc.yaml
    └── redis-service.yaml
```

### Notable Directories & Files

- **backend/**  
  - `main.py`: Entry point for FastAPI.  
  - `document_loader.py`: Logic to parse and chunk documents (PDF, DOCX).  
  - `vector_store.py`: Pinecone integration for embeddings.  
  - `redis_cache.py`: Session management and ephemeral data storage.  
  - `Dockerfile`: Containerize the FastAPI application.

- **frontend/**  
  - `app.py`: Streamlit code for the conversational interface.  
  - `Dockerfile`: Containerize the Streamlit application.

- **k8s/**  
  - `backend-deployment.yaml` / `frontend-deployment.yaml`: Describes how to deploy containers in Kubernetes.  
  - `backend-service.yaml` / `frontend-service.yaml`: Exposes the deployments inside the cluster.  
  - `frontend-ingress.yaml`: Route external traffic to the frontend service.

- **redis/**  
  - Deployment, Service, and Persistent Volume Claims for Redis.  
  - This ensures ephemeral data can be managed in-memory while also giving you control over storage classes.

- **github/workflows/**  
  - `deploy.yaml`: GitHub Actions workflow for CI/CD.

---

## Getting Started

### 1. Prerequisites

- **Docker**: Ensure Docker is installed and running.
- **Python 3.8+**: Some local testing scripts may require Python.
- **kubectl** & **AWS CLI**: If you plan to deploy to AWS EKS, install and configure these.

### 2. Local Development (Docker Compose)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YourUsername/Production-ready-RAG-pipeline-with-Deployment.git
   cd Production-ready-RAG-pipeline-with-Deployment
   ```

2. **Start services with Docker Compose**  
   ```bash
   docker-compose up --build
   ```
   - This will spin up the **FastAPI** backend, **Streamlit** frontend, and **Redis** in containers (Pinecone is external and needs separate setup).

3. **Access the application**  
   - Open your browser: [http://localhost:8501](http://localhost:8501) (Streamlit interface).  
   - The backend typically runs on port **8000**: [http://localhost:8000/docs](http://localhost:8000/docs) for FastAPI docs.

### 3. Deploying on Kubernetes (AWS EKS)

1. **Build & Push Images**  
   - Update the `Dockerfile` in **backend** and **frontend** to match your container registry path.  
   - Build & push:
     ```bash
     docker build -t <registry>/<image_name>:<tag> .
     docker push <registry>/<image_name>:<tag>
     ```

2. **Configure AWS EKS**  
   - Create or configure your EKS cluster.  
   - Ensure `kubectl` context is set to your EKS cluster.

3. **Apply Kubernetes Manifests**  
   ```bash
   kubectl apply -f k8s/redis/
   kubectl apply -f k8s/backend/
   kubectl apply -f k8s/frontend/
   kubectl apply -f k8s/ingress/
   ```
   - Wait for the pods to become ready.  
   - Retrieve the Load Balancer address or domain from your ingress configuration.

4. **Access the Application**  
   - Navigate to the provided domain or public IP to access the **Streamlit** frontend.  
   - All requests to the **FastAPI** backend will be routed internally via Kubernetes services.

---

## CI/CD with GitHub Actions

- The `.github/workflows/deploy.yaml` (and additional YAML files) contain your CI/CD pipeline.
- Typical flow:  
  1. **Trigger** on push or pull request.  
  2. **Build & Test** Docker images.  
  3. **Push** images to container registry.  
  4. **Deploy** changes to EKS automatically (or manually, depending on your workflow setup).

This ensures each commit can be tested and deployed seamlessly to your cluster.

---

## Improvements

There are many possible enhancements to this repository that could further optimize performance, scalability, and user experience:

- **Deployment with Ray Serve or vLLM**  
  - Utilize worker nodes more effectively, handle dynamic scaling of inference jobs, and optimize resource usage.  

- **Caching Strategy for Frequent Queries**  
  - Reduce retrieval overhead and improve response times by implementing a caching layer for commonly asked questions or embeddings.  

- **Persistent Session Data**  
  - Replace ephemeral Redis storage with a dedicated database (e.g., PostgreSQL) to store long-term session history or advanced analytics.  

- **Advanced RAG Pipelines**  
  - Integrate **Agentic RAG** or **Self-Reflection RAG** for more complex query handling and iterative, context-aware generation flows.  

These improvements may be added in future releases to refine the pipeline and keep it at the forefront of scalable AI deployments.

---

## Contributing

Contributions, issues, and feature requests are welcome!

- Feel free to **open an issue** if you find a bug or want to suggest an enhancement.  
- Create a **pull request** with clear instructions on what you changed.

---

## Contact & Acknowledgments

**Author:** [Karan Sharma]  
- **GitHub**: (https://github.com/KaranSharma18)  
- **LinkedIn**: (https://www.linkedin.com/in/karansharma18/)

Thank you to all open-source libraries and tools used in this project, particularly **FastAPI**, **Streamlit**, **Redis**, **Pinecone**, and the **Kubernetes** community for making scalable deployments more accessible than ever.

---

Feel free to reach out via [email or other channels](mailto:karansharma.professional@gmail.com) if you have any questions or would like to collaborate. Happy building!
