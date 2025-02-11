import os
import PyPDF2
from langchain_ollama import OllamaLLM
# Define directories
PDF_DIRECTORY = "research_papers"

# Initialize the DeepSeek-R1-14B model via Ollama
llm = OllamaLLM(model="huihui_ai/deepseek-r1-abliterated:14b")

def get_latest_pdf(directory):
    """Retrieve the latest added PDF file from the directory."""
    pdf_files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".pdf")],
        key=lambda f: os.path.getctime(os.path.join(directory, f)),
        reverse=True
    )
    return os.path.join(directory, pdf_files[0]) if pdf_files else None

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def analyze_text_with_llm(text):
    """Use DeepSeek-R1-14B to extract numerical data and key insights with structured formatting."""
    prompt = f"""
    You are an AI research assistant analyzing a scientific research paper.

    **Your Task:** Extract and summarize all **numerical data**, including:
    - Statistics (percentages, means, standard deviations)
    - Equations & mathematical expressions
    - Experimental results with specific values
    - Comparisons between numerical findings

    **Output Structure (Must Follow This Format):**
    - **Key Statistics:**
      - [Extracted statistic] → [Context from paper]
    - **Equations & Calculations:**
      - [Extracted equation] → [How it's used in the study]
    - **Experimental Findings:**
      - [Finding] → [What it proves]
    - **Summary:**
      - [Concise summary of numerical insights]

    **Extracted Text from Research Paper:**
    {text}

    Provide a well-structured output:
    """

    response = llm.invoke(prompt)
    return response.strip()


# Step 1: Get latest PDF
latest_pdf = get_latest_pdf(PDF_DIRECTORY)

if latest_pdf:
    extracted_text = extract_text_from_pdf(latest_pdf)
    print(f"\n✅ Extracted text from: {latest_pdf}\n")
    
    # Step 2: Analyze with DeepSeek-R1-14B
    numerical_insights = analyze_text_with_llm(extracted_text)
    
    print("\n📊 **Numerical Insights from DeepSeek-R1-14B:**\n")
    print(numerical_insights)
else:
    print("⚠️ No PDFs found in the directory.")
