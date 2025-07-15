'''''
prerequisites to run shell inside colab
=====================================================

!pip install colab-xterm
%load_ext colabxterm

%xterm

run in terminal
----------------------
curl https://ollama.ai/install.sh | sh

ollama serve &

ollama pull deepseek-r1:7b
ollama pull deepseek-r1:1.5b

ollama pull gemma:2b

ollama rm <model name>
---------------------------


!ollama list

from web - https://ollama.com/library/deepseek-r1 - get command for model

!pip install -U langchain-ollama

cleanup (after all is done!!)
-----------------
!rm -rf /usr/local/bin/ollama

deepseek url - https://ollama.com/library/deepseek-r1:1.5b
'''''


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Load models: DeepSeek for generation/refinement, Gemma for critique
generator_model = OllamaLLM(model="deepseek-r1:1.5b")
critique_model = OllamaLLM(model="gemma:2b")

# Step 1: Generate Content
def generate_content_idea(topic):
    print("\n Ideation Agent (DeepSeek): Generating initial content idea...")
    prompt = ChatPromptTemplate.from_template("""
    You are an expert media content creator.
    Generate a creative, engaging, and original idea for the following topic:

    Topic: {topic}

    Content Idea: """)
    chain = prompt | generator_model
    return chain.invoke({"topic": topic}).strip()

# Step 2: Critique Content
def critique_content(idea):
    print("\n Critique Agent (Gemma): Analyzing the idea...")
    prompt = ChatPromptTemplate.from_template("""
    You are a critical media analyst.
    Review the following content idea and provide constructive feedback to improve its clarity, engagement, and accuracy:

    Content Idea: {idea}

    Feedback: """)
    chain = prompt | critique_model
    return chain.invoke({"idea": idea}).strip()

# Step 3: Refine Content
def refine_content(idea, feedback):
    print("\n Refinement Agent (DeepSeek): Improving the content...")
    prompt = ChatPromptTemplate.from_template("""
    You are an expert media editor.
    Your task is to refine the provided content idea using the given feedback. Improve clarity, engagement, and accuracy.

    Original Idea: {idea}
    Feedback: {feedback}

    Provide ONLY the improved idea without restating the original or feedback.

    Refined Idea: """)
    chain = prompt | generator_model
    return chain.invoke({"idea": idea, "feedback": feedback}).strip()

# Main Loop: Run Collaboration for 5 Rounds
def agentic_media_ai(topic, rounds=5):
    idea = generate_content_idea(topic)
    print("first round creation... \n")
    print(idea)
    for i in range(rounds):
        print(f"\n Round {i + 1}...")
        feedback = critique_content(idea)
        print("round-"+str(i)+" feedback... \n")
        print(feedback)
        idea = refine_content(idea, feedback)
        print("round-"+str(i)+" refinement... \n")
        print(idea)

    print("\n Final Output:")
    print(f"Final Refined Idea: {idea}")

# Run the Continuous Agentic Media AI for 3 Rounds
agentic_media_ai("The impact of social media on journalism", rounds=3)
