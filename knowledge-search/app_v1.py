import os
import ssl

# Disable SSL verification
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# app.py
from flask import Flask, request, jsonify, render_template
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from transformers import pipeline
from langchain.schema import Document

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'xxxx'

# Expanded knowledge documents
docs = [
    {"source": "helm", "page_content": "To rollback a Helm release, use: helm rollback <release> <revision> --namespace <ns>."},
    {"source": "terraform_backend", "page_content": "Terraform state is stored in a backend. To configure S3 backend: terraform { backend \"s3\" { bucket = \"my-bucket\" key = \"path/to/state\" region = \"us-west-2\" } }"},
    {"source": "github_actions", "page_content": "Global Platform deployments use a GitHub Actions workflow named 'deploy-platform.yml' that builds and applies your Kubernetes manifests."},
    {"source": "k8s_debug", "page_content": "To troubleshoot a Kubernetes deployment, use kubectl describe pod <pod-name> and kubectl logs <pod-name>. Use `kubectl get events` to view failure events."},
    {"source": "gitops", "page_content": "GitOps pipelines can be implemented using ArgoCD or Flux to sync your Kubernetes manifests from Git repos. Ensure repository structure follows <app>/<env>/manifests pattern."},
    {"source": "rbac", "page_content": "To set RBAC in Kubernetes, define Roles and RoleBindings in YAML, then apply via kubectl apply -f role.yaml. Use `kubectl auth can-i` to test permissions."},
    {"source": "ci_cd", "page_content": "A typical GitHub Actions CI workflow includes jobs for build, test, and deploy, triggered on push to main branch. Use matrix builds for parallel testing."},
    {"source": "monitoring", "page_content": "Use Prometheus and Grafana for monitoring. Define ServiceMonitors for scraping metrics and set alerting rules in PrometheusRule CRDs."},
    {"source": "confluence_search", "page_content": "Confluence pages can be indexed via the REST API. Use `/rest/api/content?spaceKey=PLAT&expand=body.storage` to fetch page bodies for ingestion."},
    {"source": "slack_history", "page_content": "Slack messages can be exported or fetched via the Slack API using `conversations.history` to capture Q&A threads for context."},
    {"source": "helm_values", "page_content": "Helm charts support `values.yaml`. Override default values using `helm upgrade --install mychart ./chart --values custom-values.yaml`. Validate templates with `helm lint`."},
    {"source": "terraform_modules", "page_content": "Terraform modules help organize code. Use `module \"s3_bucket\" { source = \"terraform-aws-modules/s3-bucket/aws\" version = \"~>3.0\" }`."}
]

# Convert docs to LangChain Document objects
documents = [Document(**d) for d in docs]

# 1. Embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# 2. Build FAISS vector store
vectorstore = FAISS.from_documents(documents, embed_model)
# 3. Generation model pipeline
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,
    max_length=256,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=pipe)

# 4. Define prompts for the refine chain
question_prompt = PromptTemplate(
    input_variables=["context_str", "question"],
    template="""
You are a Cloud Platform expert. Use the following context to draft a clear, concise answer to the question in natural English. Do not just repeat commands—explain the why and how.

Context:
{context_str}

Question:
{question}

Answer:"""
)

refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template="""
The user asked: {question}

Existing answer:
{existing_answer}

Here is another context snippet that may help refine your answer:
{context_str}

Update and improve the answer in clear, concise English."""
)

# 5. Build a RAG chain using the refine strategy
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={
        "question_prompt": question_prompt,
        "refine_prompt": refine_prompt
    }
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    # Run the refine-based RAG chain
    answer = qa_chain.run(query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
