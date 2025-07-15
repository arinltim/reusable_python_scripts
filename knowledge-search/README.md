## Step-by-Step Approach (Simplified)

- **Disabled SSL Verification**  
  Avoids issues with certificate validation when loading models or data.

- **Prepared Knowledge Base**  
  Used a list of curated documents related to DevOps and platform topics.

- **Used Sentence Transformer for Embeddings**  
  Chose `all-MiniLM-L6-v2` (efficient and CPU-friendly) to generate text embeddings.

- **Created FAISS Vector Store**  
  Stored the embeddings using FAISS for fast similarity search.

- **Loaded TinyLlama for Text Generation**  
  Utilized HuggingFace `pipeline` to run the `TinyLlama` model with `text-generation`.

- **Defined a Strict Prompt**  
  Crafted a prompt that forces the model to use only the provided context.

- **Built a RAG Chain with LangChain**  
  Used `RetrievalQA` with the FAISS retriever and HuggingFace LLM to form the question-answering system.

- **Developed a Flask Web App**  
  Created a basic Flask server with an `/api/chat` POST endpoint for interacting with the QA system.

- **Cleaned Output Responses**  
  Extracted only the final answer using an `***ANSWER***` marker.

- **Ran Entire Stack Locally**  
  Works on CPU, without internet access once models are cached.

