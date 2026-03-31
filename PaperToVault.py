import os
import sys
import json
import time
import requests
import fitz  # PyMuPDF
import re
from pathlib import Path
from tqdm import tqdm

# =================CONFIGURATION=================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b" #"nemotron:latest"
INPUT_PDF_FOLDER = "./papers"
OUTPUT_VAULT_FOLDER = "./obsidian_vault"
CONTEXT_WINDOW_LIMIT = 25000
# ===============================================

class ResearchAgent:
    def __init__(self):
        self.session = requests.Session()
        self.vault_path = Path(OUTPUT_VAULT_FOLDER)
        self.papers_path = self.vault_path / "Papers"
        self.moc_path = self.vault_path / "000_MOC_Research.md"
        
        self.vault_path.mkdir(exist_ok=True)
        self.papers_path.mkdir(exist_ok=True)
        self._init_moc()

    def _init_moc(self):
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
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None

    def call_ollama(self, prompt, system_prompt=""):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": 4096
            }
        }
        
        try:
            response = self.session.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Ollama API Error: {e}")
            return None

    def _clean_and_parse_json(self, response_text):
        """
        Robust JSON parsing with multiple fallback strategies.
        """
        if not response_text:
            return None
        
        original_response = response_text
        print(f"    [DEBUG] Raw response length: {len(response_text)} chars")
        
        # Strategy 1: Remove markdown code fences
        cleaned = re.sub(r'```json\s*', '', response_text, flags=re.IGNORECASE)
        cleaned = re.sub(r'```\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        # Strategy 2: Try to find JSON array brackets
        json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
            print(f"    [DEBUG] Extracted JSON array from response")
        
        # Strategy 3: Fix common JSON issues
        # Replace single quotes with double quotes (for keys and string values)
        cleaned = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', cleaned)  # Keys
        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)  # Values
        
        # Remove trailing commas before ] or }
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        
        # Remove any comments (// style)
        cleaned = re.sub(r'//.*$', '', cleaned, flags=re.MULTILINE)
        
        print(f"    [DEBUG] Cleaned JSON preview: {cleaned[:200]}...")
        
        # Strategy 4: Try to parse
        try:
            parsed = json.loads(cleaned)
            print(f"    [DEBUG] JSON parsed successfully!")
            return parsed
        except json.JSONDecodeError as e:
            print(f"    [DEBUG] JSON parse error: {e}")
            print(f"    [DEBUG] Problematic JSON:\n{cleaned[:500]}")
        
        # Strategy 5: Last resort - try to extract individual objects and rebuild
        try:
            objects = re.findall(r'\{[^{}]*"filename"[^{}]*\}', cleaned, re.DOTALL)
            if objects:
                rebuilt = []
                for obj_str in objects:
                    # Clean individual object
                    obj_str = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', obj_str)
                    obj_str = re.sub(r":\s*'([^']*)'", r': "\1"', obj_str)
                    obj_str = re.sub(r',\s*}', '}', obj_str)
                    try:
                        obj = json.loads(obj_str)
                        rebuilt.append(obj)
                    except:
                        continue
                if rebuilt:
                    print(f"    [DEBUG] Rebuilt {len(rebuilt)} objects from partial parse")
                    return rebuilt
        except Exception as e:
            print(f"    [DEBUG] Rebuild failed: {e}")
        
        return None

    def generate_file_structure(self, paper_text, paper_title):
        system_prompt = """
        You are an expert Researcher in Large Language Models (LLMs) and Fine-tuning techniques.
        Your task is to analyze a research paper and plan a structured Obsidian vault structure.
        You must break complex papers into multiple nested Markdown files.
        
        IMPORTANT: Output ONLY valid JSON array. No markdown code fences. No explanatory text.
        Just the raw JSON array starting with [ and ending with ].
        
        Format exactly like this:
        [
            {"filename": "000_Main_Summary.md", "topic": "High level overview, abstract, core contribution"},
            {"filename": "001_Architecture.md", "topic": "Model architecture, diagrams, layers"},
            {"filename": "002_Finetuning_Method.md", "topic": "Specific fine-tuning techniques used"},
            {"filename": "003_Experiments.md", "topic": "Datasets, metrics, results"},
            {"filename": "004_Critique_And_Notes.md", "topic": "Critical analysis, future work"}
        ]
        
        Adjust based on the paper's actual content. Use double quotes only. No trailing commas.
        """
        
        truncated_text = paper_text[:CONTEXT_WINDOW_LIMIT] if len(paper_text) > CONTEXT_WINDOW_LIMIT else paper_text
        
        prompt = f"""
        Analyze the following research paper text titled '{paper_title}'.
        Propose a file structure to represent this knowledge in Obsidian.
        
        PAPER TEXT START:
        {truncated_text}
        PAPER TEXT END
        
        Remember: Output ONLY the JSON array, nothing else.
        """
        
        print("    -> Calling LLM for structure planning...")
        response = self.call_ollama(prompt, system_prompt)
        
        if not response:
            print("    -> No response from LLM")
            return None
        
        print("    -> Parsing JSON response...")
        structure = self._clean_and_parse_json(response)
        
        if not structure:
            print("    -> JSON parsing failed after all attempts")
            return None
        
        # Validate structure
        if not isinstance(structure, list):
            print("    -> Parsed JSON is not a list, converting...")
            structure = [structure]
        
        # Ensure each item has required fields
        validated = []
        for item in structure:
            if isinstance(item, dict) and 'filename' in item and 'topic' in item:
                # Ensure filename ends with .md
                if not item['filename'].endswith('.md'):
                    item['filename'] = item['filename'] + '.md'
                validated.append(item)
        
        if not validated:
            print("    -> No valid items in structure")
            return None
        
        print(f"    -> Successfully parsed {len(validated)} file plans")
        return validated

    def generate_content(self, paper_text, paper_title, file_plan_item, all_filenames):
        system_prompt = f"""
        You are an expert Researcher in LLMs and Fine-tuning. 
        You are writing a specific note for an Obsidian Vault.
        
        GUIDELINES:
        1. Use YAML Frontmatter (tags, aliases).
        2. Use [[Wikilinks]] for cross-referencing. Link to other files in this paper using {all_filenames}.
        3. Use Mermaid.js code blocks for diagrams (flowcharts, architecture).
        4. Focus on technical depth: Hyperparameters, Loss Functions, Architecture details, Data processing.
        5. If explaining a concept, provide a concrete example.
        6. Write raw markdown, no code fences around the entire output.
        7. prepare the content as if it's going to be read by an engineer who will implement the paper, a policymaker who will regulate based on the paper, and a student who is learning from the paper. Include sections or notes specifically for each of these audiences where relevant.
        """
        
        prompt = f"""
        Paper Title: {paper_title}
        Current File Topic: {file_plan_item['topic']}
        Filename for this note: {file_plan_item['filename']}
        Other available notes to link: {', '.join(all_filenames)}
        
        PAPER TEXT CONTEXT:
        {paper_text[:CONTEXT_WINDOW_LIMIT]}
        
        Write the content for '{file_plan_item['filename']}'. 
        Ensure you link to the other notes using [[filename]] syntax where relevant.
        Include a Mermaid diagram if the topic involves architecture or processes.
        """
        
        response = self.call_ollama(prompt, system_prompt)
        return response

    def update_moc(self, paper_title, folder_name):
        link = f"[[{folder_name}/{folder_name} - 000_Main_Summary|{paper_title}]]"
        entry = f"- {link}\n"
        
        with open(self.moc_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def process_paper(self, pdf_path):
        print(f"\n{'='*50}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*50}")
        
        text = self.extract_text(pdf_path)
        if not text:
            print("-> Failed to extract text. Skipping.")
            return
        
        safe_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in pdf_path.stem])
        folder_name = safe_title[:50]
        paper_folder = self.papers_path / folder_name
        paper_folder.mkdir(exist_ok=True)
        
        print("-> Generating Note Structure...")
        structure = self.generate_file_structure(text, pdf_path.stem)
        
        if not structure:
            print("-> Failed to generate structure. Using fallback.")
            structure = [
                {"filename": "000_Main_Summary.md", "topic": "Summary and Overview"},
                {"filename": "001_Methodology.md", "topic": "Methods and Techniques"},
                {"filename": "002_Experiments.md", "topic": "Experiments and Results"},
                {"filename": "003_Notes.md", "topic": "Critical Analysis and Notes"},
                {"filename": "004_EngineerNotes.md", "topic": "Engineer Notes"},
                {"filename": "005_PolicyMakersNotes.md", "topic": "Policy Makers Analysis and Notes"},
                {"filename": "006_StudentsNotes.md", "topic": "Students Analysis and Notes"},
                {"filename": "007_ExecutiveSummaryNotes.md", "topic": "Executive Summary and Notes"}
            ]
            
        file_names = [item['filename'] for item in structure]
        
        print(f"-> Generating {len(structure)} note files...")
        for i, item in enumerate(structure):
            print(f"   -> Writing {item['filename']} ({i+1}/{len(structure)})...")
            content = self.generate_content(text, pdf_path.stem, item, file_names)
            
            if content:
                file_path = paper_folder / item['filename']
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"      ✓ Saved {item['filename']}")
            else:
                print(f"      ✗ Failed to generate {item['filename']}")
            
            time.sleep(0.5)
            
        self.update_moc(pdf_path.stem, folder_name)
        print(f"-> ✓ Completed {pdf_path.name}")

    def run(self):
        pdf_files = list(Path(INPUT_PDF_FOLDER).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {INPUT_PDF_FOLDER}")
            print(f"Please add PDF files to the '{INPUT_PDF_FOLDER}' folder")
            return

        print(f"\n{'#'*50}")
        print(f"Found {len(pdf_files)} papers. Starting processing...")
        print(f"{'#'*50}\n")
        
        for pdf in tqdm(pdf_files, desc="Processing Papers"):
            self.process_paper(pdf)
            
        print(f"\n{'='*50}")
        print("✓ All processing complete!")
        print(f"✓ Open your Obsidian Vault at: {os.path.abspath(self.vault_path)}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    Path(INPUT_PDF_FOLDER).mkdir(exist_ok=True)
    
    print("Starting Research Paper to Obsidian Vault Converter")
    print(f"Model: {MODEL_NAME}")
    print(f"Input: {INPUT_PDF_FOLDER}")
    print(f"Output: {OUTPUT_VAULT_FOLDER}")
    print("-" * 50)
    
    agent = ResearchAgent()
    agent.run()