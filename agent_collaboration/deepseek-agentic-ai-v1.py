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

model = OllamaLLM(model="deepseek-r1:1.5b")

def generate_content_idea(topic):
    print("\n🔍 Ideation Agent: Generating initial content idea...")
    prompt = ChatPromptTemplate.from_template("""
    You are an expert media content creator.
    Generate a creative, engaging, and original idea for the following topic:

    Topic: {topic}

    Content Idea: """)

    chain = prompt | model
    idea = chain.invoke({"topic": topic})
    # return idea.content.strip()
    return idea.strip()

def critique_content(idea):
    print("\n🧐 Critique Agent: Analyzing the idea...")
    prompt = ChatPromptTemplate.from_template("""
    You are a critical media analyst.
    Review the following content idea and provide constructive feedback to improve its clarity, engagement, and accuracy:

    Content Idea: {idea}

    Feedback: """)

    chain = prompt | model
    feedback = chain.invoke({"idea": idea})
    return feedback.strip()

def refine_content(idea, feedback):
    print("\n🔧 Refinement Agent: Improving the content...")
    prompt = ChatPromptTemplate.from_template("""
    You are an expert media editor.
    Your task is to refine the provided content idea using the given feedback. Improve clarity, engagement and impact.

    Original Idea: {idea}
    Feedback: {feedback}

    Provide ONLY the improved idea without restating the original or feedback.

    Refined Idea: """)

    chain = prompt | model
    refined_idea = chain.invoke({"idea": idea, "feedback": feedback})
    return refined_idea.strip()

def agentic_media_ai(topic):
    original_idea = generate_content_idea(topic)
    feedback = critique_content(original_idea)
    refined_idea = refine_content(original_idea, feedback)

    print("\n📢 Final Output:")
    print(f"Original Idea: {original_idea}")
    print(f"Feedback: {feedback}")
    print(f"Refined Idea: {refined_idea}")

# Run the Agentic Media AI
agentic_media_ai("The impact of social media on journalism")