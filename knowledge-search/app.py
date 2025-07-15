import os
import ssl
import traceback

# Disable SSL verification
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

from flask import Flask, request, jsonify, render_template
# Use langchain_community for integrations
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # Use CausalLM for TinyLlama
from langchain.schema import Document

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'xxxx') # Use env var for secret key

# Expanded knowledge documents (Consider loading from file for larger bases)
docs = [
    {"source": "helm", "page_content": "To rollback a Helm release, use the command: helm rollback <release> <revision> --namespace <ns>. This reverts the release to a previous specified revision number."},
    {"source": "terraform_backend", "page_content": "Terraform state, which tracks your infrastructure, is stored in a configured backend. To configure an S3 backend, use the following block within your Terraform configuration: terraform { backend \"s3\" { bucket = \"my-tf-state-bucket\" key = \"path/to/my/statefile.tfstate\" region = \"us-east-1\" } }"},
    {"source": "github_actions", "page_content": "Our platform deployments leverage a standardized GitHub Actions workflow defined in '.github/workflows/deploy-platform.yml'. This workflow handles building container images, running tests, and applying Kubernetes manifests to the target environment."},
    {"source": "k8s_debug", "page_content": "When troubleshooting issues with a Kubernetes pod, start by examining its description using `kubectl describe pod <pod-name>`. Check the pod's logs with `kubectl logs <pod-name>`. Additionally, view cluster-level events related to the pod or its deployment using `kubectl get events --sort-by='.metadata.creationTimestamp'`."},
    {"source": "gitops", "page_content": "GitOps practices involve using Git as the single source of truth for declarative infrastructure and applications. Tools like ArgoCD or FluxCD can be used to automatically synchronize the state defined in a Git repository with your Kubernetes cluster. A common repository structure is `<application-name>/<environment>/manifests/`."},
    {"source": "rbac", "page_content": "Kubernetes Role-Based Access Control (RBAC) is configured using Role or ClusterRole objects (defining permissions) and RoleBinding or ClusterRoleBinding objects (linking roles to users, groups, or service accounts). Define these in YAML files and apply them using `kubectl apply -f <your-rbac-file.yaml>`. You can check if a specific user or service account has permission for an action using `kubectl auth can-i <verb> <resource> --namespace <ns> --as <user-or-sa>`."},
    {"source": "ci_cd", "page_content": "A typical Continuous Integration (CI) workflow in GitHub Actions, triggered on push to the main branch, includes steps like checking out code, setting up the environment, running linters, executing unit tests, and building artifacts (e.g., Docker images). Continuous Deployment (CD) might involve deploying to a staging environment first, running integration tests, and then promoting to production after approval."},
    {"source": "monitoring", "page_content": "We use Prometheus for metrics collection and Grafana for visualization and dashboards. Prometheus discovers and scrapes metrics from applications configured via ServiceMonitor custom resources. Alerting rules are defined in PrometheusRule custom resources and managed by the Prometheus Operator or Alertmanager."},
    {"source": "confluence_search", "page_content": "To integrate Confluence content, you can use the Confluence REST API. For example, fetch page content from a specific space using a GET request to `/rest/api/content?spaceKey=YOURSPACE&expand=body.storage`. The response body will contain the page content in Confluence Storage Format (XML)."},
    {"source": "slack_history", "page_content": "Relevant Q&A or troubleshooting discussions from Slack can be archived or indexed. Use the Slack API method `conversations.history` with the appropriate channel ID to fetch message history. You might need to filter for threads or specific user interactions."},
    {"source": "helm_values", "page_content": "Helm charts use a `values.yaml` file for default configuration parameters. You can override these defaults during installation or upgrade using the `--values` (or `-f`) flag followed by a path to your custom YAML file, like so: `helm upgrade --install my-release ./my-chart -f custom-values.yaml`. Use `helm lint ./my-chart` to check the chart for potential issues."},
    {"source": "terraform_modules", "page_content": "Terraform modules promote code reuse and organization. Define a module block in your Terraform code, specifying the `source` (e.g., path to local module, Git repository, or Terraform Registry) and any required input variables. Example using a Registry module: `module \"vpc\" { source = \"terraform-aws-modules/vpc/aws\" version = \"~> 5.0\" name = \"my-vpc\" cidr = \"10.0.0.0/16\" ... }`."}
]


# Convert docs to LangChain Document objects
documents = [Document(**d) for d in docs]

# --- Configuration ---
# Embedding Model (Good choice for CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model (Using TinyLlama as per previous discussion)
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_TASK = "text-generation" # Task for causal LMs like TinyLlama

# LLM Pipeline Settings
MAX_NEW_TOKENS = 512
DO_SAMPLE = True
TEMPERATURE = 0.4 # Lowered temperature for more focused output
TOP_P = 0.9 # Keep top_p for some variability if needed

# Retriever Settings
RETRIEVER_K = 2 # Reduced K to provide less (potentially noisy) context
# --- End Configuration ---


print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print("Building FAISS vector store...")
vectorstore = FAISS.from_documents(documents, embed_model)

print(f"Loading LLM model: {LLM_MODEL_NAME} for task: {LLM_TASK}")
# Using pipeline for simplicity - ensures model and tokenizer are loaded
# For CausalLM like TinyLlama, text-generation is appropriate
pipe = pipeline(
    LLM_TASK,
    model=LLM_MODEL_NAME,
    device=-1, # -1 for CPU
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    # trust_remote_code=True, # May be needed for some models like phi-2, not usually for TinyLlama
)
# Wrap the pipeline in the Langchain LLM object
llm = HuggingFacePipeline(pipeline=pipe)


# Define the stricter prompt for the 'stuff' chain type
prompt_template = """
***INSTRUCTIONS***
You are a helpful Cloud Platform assistant. Your task is to answer the user's QUESTION based ONLY on the RELEVANT information within the provided CONTEXT.
1. Read the QUESTION carefully.
2. Find the specific sentences or parts of the CONTEXT that directly answer the QUESTION.
3. Synthesize ONLY that relevant information into a clear, concise answer in natural English.
4. If the answer includes a command, briefly explain its purpose based on the context.
5. CRITICALLY IMPORTANT: Ignore and DO NOT mention any information from the CONTEXT that is NOT directly related to the specific QUESTION asked.
6. If the provided CONTEXT does not contain an answer to the QUESTION, state that the information is not available in the context. Do not add unrelated details.

***CONTEXT***
{context}

***QUESTION***
{question}

***ANSWER***"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Build a RAG chain using the 'stuff' strategy with reduced K
retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
print(f"Initializing RAG chain with retriever K={RETRIEVER_K}...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # Use stuff for simplicity
    retriever=retriever,
    return_source_documents=False, # Set to True for debugging to see retrieved docs
    chain_type_kwargs={"prompt": QA_PROMPT} # Use the stricter prompt
)
print("RAG chain ready.")

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handles chat requests."""
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        print(f"Received query: {query}")
        # Use invoke for the chain call
        result = qa_chain.invoke({"query": query})
        print(f"Raw result from chain: {result}") # Log the raw output structure

        # The raw text response is typically in the 'result' key
        raw_answer = result.get('result', '')

        # --- Post-processing Step ---
        # Define the marker that indicates the start of the actual answer
        answer_marker = "***ANSWER***"

        # Find the position of the marker
        marker_pos = raw_answer.find(answer_marker)

        if marker_pos != -1:
            # If marker is found, take the text *after* the marker
            # Add len(answer_marker) to start slicing after the marker itself
            extracted_answer = raw_answer[marker_pos + len(answer_marker):].strip()
            print(f"Extracted answer after marker: {extracted_answer}")
        else:
            # If marker is not found (shouldn't happen with current prompt but good to handle)
            # fallback to using the raw answer, maybe log a warning
            print(f"Warning: Answer marker '{answer_marker}' not found in raw output.")
            extracted_answer = raw_answer.strip()

        # Check if the extracted answer is empty or just whitespace
        if not extracted_answer:
            extracted_answer = "Sorry, I could not generate a valid answer based on the provided context."
        # --- End Post-processing Step ---


        return jsonify({'answer': extracted_answer}) # Return the cleaned answer

    except Exception as e:
        print(f"Error processing query '{query}': {e}")
        # Log the full exception traceback for debugging server-side
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred while processing your request.'}), 500


if __name__ == '__main__':
    # Set host='0.0.0.0' to be accessible on the network
    # debug=True is useful for development but should be False in production
    # Use port from environment variable if available, otherwise default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on host 0.0.0.0 port {port}")
    # Turn off Flask reloader in production or if it causes issues with model loading
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
