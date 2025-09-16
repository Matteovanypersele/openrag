# Deploying OpenRAG on Kubernetes

This guide explains how to deploy the **OpenRAG** stack on a Kubernetes cluster using Helm.

---

## Prerequisites

- A **Kubernetes cluster** with **GPU nodes** available (NVIDIA runtime) and nvidia-gpu-operator installed.
- A **StorageClass** that supports **ReadWriteMany** (`RWX`) access mode.  
  This is required because the Ray cluster workers and the OpenRAG app need to access the same shared volumes (e.g. for `.venv`, model weights, logs, data).
- If using ingress, the ingress-nginx controller needs to be installed on the cluster.

---

## Steps

1. **Clone the repository** (if not already done):

   ```bash
   git clone https://github.com/linagora/openrag.git
   cd openrag
   ```

2. **Prepare your `values.yaml` file**:

   - Copy or create a new `values.yaml` at the root of your repo.
   - You can see the full example file inside the chart:
     [../charts/openrag-stack/values.yaml](../charts/openrag-stack/values.yaml)
   - Customize the values you need (e.g., image tags, resources, ingress host, storage class, environment variables, secrets).

3. **Set environment and secrets**:

   - Edit the `env.config` and `env.secrets` sections in your `values.yaml`.
   - Secrets (API keys, tokens, Hugging Face credentials, etc.) will be mounted into the cluster as Kubernetes secrets.

4. **Update Helm dependencies**:

   ```bash
   helm dependency update charts/openrag-stack
   ```

   This will pull in required subcharts (e.g. PostgreSQL, Milvus, vLLM, Infinity reranker).

5. **Install or upgrade the release**:

   ```bash
   helm upgrade --install openrag ./charts/openrag-stack -f ./values.yaml
   ```

   - `openrag` is the Helm release name.
   - `./charts/openrag-stack` is the path to the chart.
   - `-f ./values.yaml` specifies your custom configuration.

---

## Notes

- If using a public IP instead of a hostname, you can leave `ingress.host` empty in your `values.yaml`.  
  The ingress will then match all hosts.

- If you later configure a hostname + TLS (via cert-manager), just update `ingress.host` and redeploy.

- Ensure your GPU nodes have the correct NVIDIA drivers and `nvidia` `RuntimeClass` configured.

---
