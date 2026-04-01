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
MODEL_NAME = "nemotron-cascade-2:latest" # "llama3:8b" #"nemotron:latest"
INPUT_PDF_FOLDER = "./papers"
OUTPUT_VAULT_FOLDER = "./obsidian_vault_output"
CONTEXT_WINDOW_LIMIT = 8000
OLLAMA_TIMEOUT = 300000
MAX_RETRIES = 10
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

    def call_ollama(self, prompt, system_prompt="", retry_count=0):
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower for more deterministic output
                "num_ctx": 4096,
                "num_predict": 1024
            }
        }
        
        attempt = retry_count + 1
        print(f"    [Attempt {attempt}/{MAX_RETRIES}] Calling Ollama...", end=" ", flush=True)
        
        try:
            response = self.session.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            response.raise_for_status()
            result = response.json()['response']
            print("✓ Done")
            return result
            
        except requests.exceptions.Timeout:
            print(f"✗ Timeout ({OLLAMA_TIMEOUT}s)")
            if attempt < MAX_RETRIES:
                print(f"    -> Retrying in 5 seconds...")
                time.sleep(5)
                return self.call_ollama(prompt, system_prompt, attempt)
            return None
            
        except Exception as e:
            print(f"✗ Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
                return self.call_ollama(prompt, system_prompt, attempt)
            return None

    def _validate_structure_schema(self, items):
        """
        CRITICAL FIX: Validate that each item has 'filename' and 'topic' fields.
        Returns only valid items.
        """
        valid_items = []
        for item in items:
            if isinstance(item, dict):
                # Check for required fields
                has_filename = 'filename' in item and item['filename']
                has_topic = 'topic' in item and item['topic']
                
                if has_filename and has_topic:
                    # Ensure filename ends with .md
                    fn = str(item['filename']).strip()
                    if not fn.endswith('.md'):
                        fn = fn + '.md'
                    valid_items.append({
                        "filename": fn,
                        "topic": str(item['topic']).strip()
                    })
                else:
                    # Log what fields we actually got
                    keys = list(item.keys()) if isinstance(item, dict) else 'unknown'
                    print(f"    [WARN] Invalid schema - got fields: {keys}, expected: filename, topic")
        
        return valid_items

    def _extract_json_objects(self, text):
        """Extract JSON and validate schema."""
        if not text:
            return None
        
        print(f"    [DEBUG] Response length: {len(text)} chars")
        
        results = []
        
        # Strategy 1: Look for complete JSON array
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            print(f"    [DEBUG] Found JSON array pattern")
            
            # Clean the JSON string
            json_str = re.sub(r'```json\s*', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'```\s*', '', json_str)
            json_str = json_str.strip()
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    results.extend(parsed)
                elif isinstance(parsed, dict):
                    results.append(parsed)
                print(f"    [DEBUG] JSON parsed successfully")
            except json.JSONDecodeError as e:
                print(f"    [DEBUG] JSON parse error: {e}")
                print(f"    [DEBUG] Problematic JSON:\n{json_str[:500]}")
        
        # Strategy 2: Look for individual objects with filename field
        if not results:
            obj_pattern = r'\{[^{}]*?"filename"[^{}]*?\}'
            objects = re.findall(obj_pattern, text, re.DOTALL | re.IGNORECASE)
            print(f"    [DEBUG] Found {len(objects)} objects with 'filename'")
            
            for obj_str in objects:
                obj_str = obj_str.replace("'", '"')
                obj_str = re.sub(r',\s*}', '}', obj_str)
                try:
                    parsed = json.loads(obj_str)
                    results.append(parsed)
                except:
                    continue
        
        # Strategy 3: Line-by-line extraction for simple patterns
        if not results:
            lines = text.split('\n')
            for line in lines:
                if 'filename' in line.lower() and '.md' in line:
                    filename_match = re.search(r'["\']([^"\']*\.md)["\']', line)
                    topic_match = re.search(r'["\']([^"\']+)["\']', line)
                    
                    if filename_match:
                        item = {"filename": filename_match.group(1), "topic": "Section"}
                        if topic_match and topic_match.group(1) != item["filename"]:
                            item["topic"] = topic_match.group(1)
                        results.append(item)
        
        # CRITICAL: Validate schema
        if results:
            valid = self._validate_structure_schema(results)
            if valid:
                print(f"    [DEBUG] ✓ Validated {len(valid)} items with correct schema")
                return valid
            else:
                print(f"    [DEBUG] ✗ No items passed schema validation")
        
        return None

    def generate_file_structure(self, paper_text, paper_title):
        """
        Generate file structure with EXTREMELY explicit format requirements.
        """
        # Use a very constrained system prompt with examples
        system_prompt = """
        You MUST output a JSON array with this EXACT schema:
        
        [
            {"filename": "000_Main_Summary.md", "topic": "High level overview, abstract, core contribution"},
            {"filename": "001_Architecture.md", "topic": "Model architecture, diagrams, layers"},
            {"filename": "002_Finetuning_Method.md", "topic": "Specific fine-tuning techniques used"},
            {"filename": "003_Experiments.md", "topic": "Datasets, metrics, results"},
            {"filename": "004_Critique_And_Notes.md", "topic": "Critical analysis, future work"},
            {"filename": "004_EngineerNotes.md", "topic": "Engineer Notes"},
            {"filename": "005_PolicyMakersNotes.md", "topic": "Policy Makers Analysis and Notes"},
            {"filename": "006_StudentsNotes.md", "topic": "Students Analysis and Notes"},
            {"filename": "007_ExecutiveSummaryNotes.md", "topic": "Executive Summary and Notes"}
        ]
        
        CRITICAL REQUIREMENTS:
        - Each object MUST have exactly two fields: "filename" and "topic"
        - "filename" MUST end with .md
        - "filename" MUST start with a number like 000_, 001_, 002_
        - Use DOUBLE quotes only, no single quotes
        - NO other fields allowed (no "step", no "objective", etc.)
        - NO markdown code fences
        - NO explanatory text
        - Output ONLY the JSON array, nothing else
        - Create 4-6 files maximum
        
        WRONG (do NOT output this):
        [{"step": "...", "objective": "..."}]
        
        RIGHT (output this format):
        [{"filename": "000_Summary.md", "topic": "..."}]
        """
        
        # Extract key sections to reduce context
        sections = self._extract_key_sections(paper_text)
        context = f"""
        Title: {paper_title}
        
        Abstract/Intro:
        {sections.get('abstract', 'N/A')[:1000]}
        
        Methods:
        {sections.get('methods', 'N/A')[:1000]}
        """
        
        prompt = f"""
        Analyze this research paper and create a file structure for Obsidian notes.
        
        {context}
        
        Output ONLY the JSON array with filename and topic fields:
        """
        
        print("    -> Generating structure (30-60s)...")
        response = self.call_ollama(prompt, system_prompt)
        
        if not response:
            return None
        
        print("    -> Parsing and validating JSON...")
        structure = self._extract_json_objects(response)
        
        if structure:
            print(f"    -> ✓ Structure validated: {len(structure)} files")
            for item in structure:
                print(f"       - {item['filename']}: {item['topic'][:50]}")
            return structure
        
        print("    -> ✗ Structure extraction failed")
        return None

    def _extract_key_sections(self, text):
        """Extract key sections from paper to reduce context."""
        sections = {'abstract': '', 'methods': '', 'results': ''}
        
        patterns = {
            'abstract': [r'abstract[:\s]*(.*?)(?=introduction|1\.|introduction)', r'introduction[:\s]*(.*?)(?=method|2\.|method)'],
            'methods': [r'method[:\s]*(.*?)(?=result|3\.|experiment)', r'methodology[:\s]*(.*?)(?=result|3\.|experiment)'],
            'results': [r'result[:\s]*(.*?)(?=conclusion|discussion|5\.|conclusion)']
        }
        
        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    sections[section] = match.group(1).strip()[:2000]
                    break
        
        if not any(sections.values()):
            sections['abstract'] = text[:5000]
        
        return sections

    def generate_content(self, paper_text, paper_title, file_plan_item, all_filenames):
        system_prompt = """
        You are an expert LLM researcher writing Obsidian notes.
        
        FORMAT:
        1. Start with YAML frontmatter: --- tags: [] ---
        2. Use [[filename]] for links to other notes
        3. Use ```mermaid for diagrams
        4. Technical depth on LLMs/fine-tuning
        5. Raw markdown output only
        6. Include sections for engineers, policymakers, students where relevant
        7. Focus on hyperparameters, loss functions, architecture, data processing details
        8. If explaining a concept, provide a concrete example.
        9. Use Mermaid.js code blocks for diagrams (flowcharts, architecture).
        10. Focus on technical depth: Hyperparameters, Loss Functions, Architecture details, Data processing.
        11. If explaining a concept, provide a concrete example.
        12. Write raw markdown, no code fences around the entire output.
        13. prepare the content as if it's going to be read by an engineer who will implement the paper, a policymaker who will regulate based on the paper, and a student who is learning from the paper. Include sections or notes specifically for each of these audiences where relevant.
        """
        
        section_text = self._get_relevant_section(paper_text, file_plan_item['filename'])
        
        prompt = f"""
        PAPER: {paper_title}
        FILE: {file_plan_item['filename']}
        TOPIC: {file_plan_item['topic']}
        LINKS: {', '.join(all_filenames)}
        
        RELEVANT CONTENT:
        {section_text}
        
        Write complete markdown for {file_plan_item['filename']}.
        Link to other files with [[filename]]. Include mermaid diagram.
        """
        
        response = self.call_ollama(prompt, system_prompt)
        return response

    def _get_relevant_section(self, text, filename):
        filename_lower = filename.lower()
        
        if 'summary' in filename_lower or '000' in filename_lower:
            return text[:8000]
        elif 'arch' in filename_lower or '001' in filename_lower:
            match = re.search(r'(architecture|model|design)[:\s]*(.*?)(?=method|experiment)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[4000:12000]
        elif 'method' in filename_lower or '002' in filename_lower:
            match = re.search(r'(method|methodology|approach)[:\s]*(.*?)(?=result|experiment)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[8000:16000]
        elif 'experiment' in filename_lower or '003' in filename_lower:
            match = re.search(r'(experiment|result|evaluation)[:\s]*(.*?)(?=conclusion)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[12000:20000]
        else:
            return text[-8000:]

    def update_moc(self, paper_title, folder_name):
        link = f"[[{folder_name}/{folder_name} - 000_Summary|{paper_title}]]"
        with open(self.moc_path, "a", encoding="utf-8") as f:
            f.write(f"- {link}\n")

    def process_paper(self, pdf_path):
        print(f"\n{'='*60}")
        print(f"PROCESSING: {pdf_path.name}")
        print(f"{'='*60}")
        
        print("\n→ Step 1: Extracting text from PDF...")
        text = self.extract_text(pdf_path)
        if not text:
            print("✗ Failed to extract text. Skipping.")
            return
        print(f"✓ Extracted {len(text)} characters")
        
        safe_title = "".join([c if c.isalnum() or c in " -_" else "_" for c in pdf_path.stem])
        folder_name = safe_title[:50]
        paper_folder = self.papers_path / folder_name
        paper_folder.mkdir(exist_ok=True)
        
        print("\n→ Step 2: Generating Note Structure...")
        structure = self.generate_file_structure(text, pdf_path.stem)
        
        if not structure:
            print("⚠ Using fallback structure...")
            structure = [
                {"filename": "000_Summary.md", "topic": "Executive Summary"},
                {"filename": "001_Architecture.md", "topic": "Model Architecture"},
                {"filename": "002_Methodology.md", "topic": "Methods & Fine-tuning"},
                {"filename": "003_Experiments.md", "topic": "Experiments & Results"},
                {"filename": "004_Analysis.md", "topic": "Analysis & Future Work"},
                {"filename": "005_Notes.md", "topic": "Critical Analysis and Notes"},
                {"filename": "006_EngineerNotes.md", "topic": "Engineer Notes"},
                {"filename": "007_PolicyMakersNotes.md", "topic": "Policy Makers Analysis and Notes"},
                {"filename": "008_StudentsNotes.md", "topic": "Students Analysis and Notes"},
                {"filename": "009_ExecutiveSummaryNotes.md", "topic": "Executive Summary and Notes"}
            ]
        
        file_names = [item['filename'] for item in structure]
        print(f"✓ Structure: {len(structure)} files")
        
        print(f"\n→ Step 3: Generating Content (30-60s per file)...")
        files_created = 0
        for i, item in enumerate(structure):
            print(f"   [{i+1}/{len(structure)}] {item['filename']}...", end=" ", flush=True)
            content = self.generate_content(text, pdf_path.stem, item, file_names)
            
            if content:
                file_path = paper_folder / item['filename']
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"✓ ({len(content)} chars)")
                files_created += 1
            else:
                print("✗")
            
            time.sleep(0.5)
        
        print(f"\n→ Step 4: Updating Index...")
        self.update_moc(pdf_path.stem, folder_name)
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETED: {pdf_path.name}")
        print(f"✓ Files: {files_created}/{len(structure)}")
        print(f"✓ Location: {paper_folder}")
        print(f"{'='*60}\n")

    def run(self):
        pdf_files = list(Path(INPUT_PDF_FOLDER).glob("*.pdf"))
        
        if not pdf_files:
            print(f"\n✗ No PDF files found in {INPUT_PDF_FOLDER}")
            return

        print(f"\n{'#'*60}")
        print(f"# RESEARCH PAPER TO OBSIDIAN VAULT CONVERTER")
        print(f"{'#'*60}")
        print(f"Model: {MODEL_NAME}")
        print(f"Timeout: {OLLAMA_TIMEOUT}s")
        print(f"Papers: {len(pdf_files)}")
        print(f"{'#'*60}\n")
        
        for pdf in tqdm(pdf_files, desc="Processing Papers"):
            self.process_paper(pdf)
            
        print(f"\n{'='*60}")
        print("✓✓✓ ALL PROCESSING COMPLETE! ✓✓✓")
        print(f"✓ Vault: {os.path.abspath(self.vault_path)}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    Path(INPUT_PDF_FOLDER).mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("RESEARCH PAPER TO OBSIDIAN VAULT CONVERTER")
    print("="*60)
    
    try:
        test = requests.get("http://localhost:11434/api/tags", timeout=10)
        print("✓ Ollama is running")
    except:
        print("✗ Ollama is NOT running! Start with: ollama serve")
        sys.exit(1)
    
    print("="*60 + "\n")
    agent = ResearchAgent()
    agent.run()