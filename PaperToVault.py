import os
import sys
import json
import time
import requests
import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

# =================CONFIGURATION=================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "nemotron:latest"
INPUT_PDF_FOLDER = "./papers"  # Folder containing raw PDFs
OUTPUT_VAULT_FOLDER = "./obsidian_vault" # Destination Obsidian Vault
CONTEXT_WINDOW_LIMIT = 25000  # Approx character limit to send to LLM (adjust based on your VRAM)
# ===============================================

class ResearchAgent:
    def __init__(self):
        self.session = requests.Session()
        self.vault_path = Path(OUTPUT_VAULT_FOLDER)
        self.papers_path = self.vault_path / "Papers"
        self.moc_path = self.vault_path / "000_MOC_Research.md"
        
        # Initialize Vault Structure
        self.vault_path.mkdir(exist_ok=True)
        self.papers_path.mkdir(exist_ok=True)
        self._init_moc()

    def _init_moc(self):
        """Initialize the Map of Content file if it doesn't exist."""
        if not self.moc_path.exists():
            content = """---
tags: [moc, research]
---
# Research Map of Content

This is the central hub for all research notes generated from PDFs.

## Papers
"""
            with open(self.moc_path, "w", encoding="utf-8") as f:
                f.write(content)

    def extract_text(self, pdf_path):
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            # Basic cleanup
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None

    def call_ollama(self, prompt, system_prompt=""):
        """Send a request to Ollama API."""
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3, # Lower temp for factual accuracy
                "num_ctx": 4096     # Adjust based on your hardware
            }
        }
        
        try:
            response = self.session.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return None

    def generate_file_structure(self, paper_text, paper_title):
        """
        Phase 1: Ask LLM to plan the note structure (Nesting).
        Returns a JSON list of planned files.
        """
        system_prompt = """
        You are an expert Researcher in Large Language Models (LLMs) and Fine-tuning techniques.
        Your task is to analyze a research paper and plan a structured Obsidian vault structure.
        You must break complex papers into multiple nested Markdown files.
        
        Output ONLY valid JSON. No markdown code fences.
        Format:
        [
            {"filename": "000_Main_Summary.md", "topic": "High level overview, abstract, core contribution"},
            {"filename": "001_Architecture.md", "topic": "Model architecture, diagrams, layers"},
            {"filename": "002_Finetuning_Method.md", "topic": "Specific fine-tuning techniques used (LoRA, DPO, etc.)"},
            {"filename": "003_Experiments.md", "topic": "Datasets, metrics, results"},
            {"filename": "004_Critique_And_Notes.md", "topic": "Critical analysis, future work, personal researcher notes"}
        ]
        Adjust the list based on the paper's actual content. If a section is not relevant, omit it.
        """
        
        # Truncate text if too long for the planning phase
        truncated_text = paper_text[:CONTEXT_WINDOW_LIMIT] if len(paper_text) > CONTEXT_WINDOW_LIMIT else paper_text
        
        prompt = f"""
        Analyze the following research paper text titled '{paper_title}'.
        Propose a file structure to represent this knowledge in Obsidian.
        
        PAPER TEXT START:
        {truncated_text}
        PAPER TEXT END
        """
        
        response = self.call_ollama(prompt, system_prompt)
        if not response:
            return None
            
        # Clean response to ensure JSON parsing
        try:
            # Sometimes LLMs wrap JSON in ```json ... ```
            clean_json = response.replace("```json", "").replace("```", "").strip()
            structure = json.loads(clean_json)
            return structure
        except json.JSONDecodeError:
            print("Failed to parse JSON structure from LLM. Falling back to default.")
            return [
                {"filename": "000_Main_Summary.md", "topic": "Summary"},
                {"filename": "001_Details.md", "topic": "Details"}
            ]

    def generate_content(self, paper_text, paper_title, file_plan_item, all_filenames):
        """
        Phase 2: Generate content for a specific file in the plan.
        """
        system_prompt = f"""
        You are an expert Researcher in LLMs and Fine-tuning. 
        You are writing a specific note for an Obsidian Vault.
        
        GUIDELINES:
        1. Use YAML Frontmatter (tags, aliases).
        2. Use [[Wikilinks]] for cross-referencing. Link to other files in this paper using {all_filenames}.
        3. Use Mermaid.js code blocks for diagrams (flowcharts, architecture).
        4. Focus on technical depth: Hyperparameters, Loss Functions, Architecture details, Data processing.
        5. If explaining a concept, provide a concrete example.
        6. Do not use markdown code fences around the entire output. Write raw markdown.
        """
        
        prompt = f"""
        Paper Title: {paper_title}
        Current File Topic: {file_plan_item['topic']}
        Filename for this note: {file_plan_item['filename']}
        Other available notes to link: {', '.join(all_filenames)}
        
        PAPER TEXT CONTEXT:
        {paper_text[:CONTEXT_WINDOW_LIMIT]}
        
        Write the content for '{file_plan_item['filename']}'. 
        Ensure you link to the other notes using [[filename]] syntax where relevant to create a knowledge graph.
        Include a Mermaid diagram if the topic involves architecture or processes.
        """
        
        response = self.call_ollama(prompt, system_prompt)
        return response

    def update_moc(self, paper_title, folder_name):
        """Append the new paper to the main Map of Content."""
        link = f"[[{folder_name}/{folder_name} - 000_Main_Summary|{paper_title}]]"
        entry = f"- {link}\n"
        
        with open(self.moc_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def process_paper(self, pdf_path):
        """Main workflow for a single PDF."""
        print(f"\n{'='*40}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*40}")
        
        # 1. Extract Text
        text = self.extract_text(pdf_path)
        if not text:
            return
        
        # Sanitize title for folder name
        safe_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in pdf_path.stem])
        folder_name = safe_title[:50] # Limit folder name length
        paper_folder = self.papers_path / folder_name
        paper_folder.mkdir(exist_ok=True)
        
        # 2. Plan Structure
        print("-> Generating Note Structure...")
        structure = self.generate_file_structure(text, pdf_path.stem)
        if not structure:
            print("-> Failed to generate structure. Skipping.")
            return
            
        file_names = [item['filename'] for item in structure]
        
        # 3. Generate Content for each file
        for i, item in enumerate(structure):
            print(f"-> Writing {item['filename']} ({i+1}/{len(structure)})...")
            content = self.generate_content(text, pdf_path.stem, item, file_names)
            
            if content:
                file_path = paper_folder / item['filename']
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
            # Rate limit protection
            time.sleep(1) 
            
        # 4. Update Index
        self.update_moc(pdf_path.stem, folder_name)
        print(f"-> Completed {pdf_path.name}")

    def run(self):
        """Scan input folder and process all PDFs."""
        pdf_files = list(Path(INPUT_PDF_FOLDER).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {INPUT_PDF_FOLDER}")
            return

        print(f"Found {len(pdf_files)} papers. Starting processing...")
        
        for pdf in tqdm(pdf_files):
            self.process_paper(pdf)
            
        print("\nAll processing complete. Open your Obsidian Vault at: ", os.path.abspath(self.vault_path))

if __name__ == "__main__":
    # Create input folder if it doesn't exist
    Path(INPUT_PDF_FOLDER).mkdir(exist_ok=True)
    
    agent = ResearchAgent()
    agent.run()