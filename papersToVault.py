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
MODEL_NAME = "nemotron:latest"
INPUT_PDF_FOLDER = "./papers"
OUTPUT_VAULT_FOLDER = "./obsidian_vault"
CONTEXT_WINDOW_LIMIT = 8000  # REDUCED from 25000
OLLAMA_TIMEOUT = 300  # INCREASED from 120 to 300 seconds
MAX_RETRIES = 3
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
        """Call Ollama with retry logic and progress indication."""
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 2048
            }
        }
        
        attempt = retry_count + 1
        print(f"    [Attempt {attempt}/{MAX_RETRIES}] Calling Ollama...", end=" ", flush=True)
        
        try:
            response = self.session.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=OLLAMA_TIMEOUT
            )
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
            
        except requests.exceptions.ConnectionError:
            print("✗ Connection Error - Is Ollama running?")
            return None
            
        except Exception as e:
            print(f"✗ Error: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
                return self.call_ollama(prompt, system_prompt, attempt)
            return None

    def _extract_json_objects(self, text):
        """Extract JSON from LLM response with multiple strategies."""
        if not text:
            return None
        
        results = []
        
        # Strategy 1: Look for complete JSON array
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            parsed = self._try_parse_json(json_str)
            if parsed and isinstance(parsed, list):
                results.extend(parsed)
        
        # Strategy 2: Look for individual objects
        obj_pattern = r'\{[^{}]*?"filename"[^{}]*?\}'
        objects = re.findall(obj_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for obj_str in objects:
            parsed = self._try_parse_json(obj_str)
            if parsed and isinstance(parsed, dict):
                results.append(parsed)
        
        # Strategy 3: Line-by-line extraction
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
        
        # Deduplicate
        final_results = []
        seen = set()
        for item in results:
            if isinstance(item, dict) and 'filename' in item:
                fn = item['filename']
                if fn not in seen and fn.endswith('.md'):
                    seen.add(fn)
                    item['topic'] = item.get('topic', f'Section: {fn}')
                    final_results.append(item)
        
        return final_results if final_results else None

    def _try_parse_json(self, json_str):
        """Parse JSON with cleanup."""
        if not json_str:
            return None
        
        json_str = re.sub(r'```json\s*', '', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'```\s*', '', json_str)
        json_str = json_str.strip()
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        
        try:
            return json.loads(json_str)
        except:
            return None

    def generate_file_structure(self, paper_text, paper_title):
        """Generate file structure with minimal context."""
        system_prompt = """
        Output ONLY a JSON array. Example:
        [{"filename": "000_Summary.md", "topic": "Summary"}, {"filename": "001_Methods.md", "topic": "Methods"}]
        
        Rules: double quotes only, no markdown, no extra text, 4-6 files max.
        """
        
        # Extract only key sections to reduce context
        sections = self._extract_key_sections(paper_text)
        context = f"""
        Title: {paper_title}
        
        Abstract/Intro:
        {sections.get('abstract', 'N/A')[:1500]}
        
        Key Sections:
        {sections.get('methods', 'N/A')[:1500]}
        """
        
        prompt = f"""
        Create file structure for this paper:
        
        {context}
        
        Output ONLY JSON array:
        """
        
        print("    -> Generating structure (this may take 30-60s)...")
        response = self.call_ollama(prompt, system_prompt)
        
        if not response:
            return None
        
        structure = self._extract_json_objects(response)
        
        if structure:
            validated = []
            for i, item in enumerate(structure[:6]):  # Max 6 files
                if isinstance(item, dict) and 'filename' in item:
                    fn = item['filename'].strip()
                    if not fn.endswith('.md'):
                        fn = fn + '.md'
                    if fn not in [v['filename'] for v in validated]:
                        validated.append({"filename": fn, "topic": item.get('topic', f'Section {i+1}')})
            return validated if validated else None
        
        return None

    def _extract_key_sections(self, text):
        """Extract key sections from paper to reduce context."""
        sections = {
            'abstract': '',
            'methods': '',
            'results': ''
        }
        
        # Look for common section headers
        patterns = {
            'abstract': [r'abstract[:\s]*(.*?)(?=introduction|1\.|introduction)', r'introduction[:\s]*(.*?)(?=method|2\.|method)'],
            'methods': [r'method[:\s]*(.*?)(?=result|3\.|experiment)', r'methodology[:\s]*(.*?)(?=result|3\.|experiment)'],
            'results': [r'result[:\s]*(.*?)(?=conclusion|discussion|5\.|conclusion)']
        }
        
        text_lower = text.lower()
        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    sections[section] = match.group(1).strip()[:2000]
                    break
        
        # If no sections found, use first 5000 chars
        if not any(sections.values()):
            sections['abstract'] = text[:5000]
        
        return sections

    def generate_content(self, paper_text, paper_title, file_plan_item, all_filenames):
        """Generate content for a single file."""
        system_prompt = """
        You are an expert LLM researcher writing Obsidian notes.
        
        FORMAT:
        1. YAML frontmatter: --- tags: [] ---
        2. Use [[filename]] for links
        3. Use ```mermaid for diagrams
        4. Technical depth on LLMs/fine-tuning
        5. Raw markdown output
        """
        
        # Get relevant section based on filename
        section_text = self._get_relevant_section(paper_text, file_plan_item['filename'])
        
        prompt = f"""
        PAPER: {paper_title}
        FILE: {file_plan_item['filename']}
        TOPIC: {file_plan_item['topic']}
        LINKS: {', '.join(all_filenames)}
        
        RELEVANT CONTENT:
        {section_text}
        
        Write complete markdown for {file_plan_item['filename']}.
        Link to other files with [[filename]]. Include mermaid diagram if applicable.
        """
        
        response = self.call_ollama(prompt, system_prompt)
        return response

    def _get_relevant_section(self, text, filename):
        """Get relevant text section based on filename."""
        filename_lower = filename.lower()
        
        if 'summary' in filename_lower or '000' in filename_lower:
            return text[:8000]
        elif 'arch' in filename_lower or '001' in filename_lower:
            match = re.search(r'(architecture|model|design)[:\s]*(.*?)(?=method|experiment|result)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[4000:12000]
        elif 'method' in filename_lower or '002' in filename_lower:
            match = re.search(r'(method|methodology|approach)[:\s]*(.*?)(?=result|experiment|evaluation)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[8000:16000]
        elif 'experiment' in filename_lower or '003' in filename_lower:
            match = re.search(r'(experiment|result|evaluation)[:\s]*(.*?)(?=conclusion|discussion)', text, re.IGNORECASE | re.DOTALL)
            return match.group(0)[:8000] if match else text[12000:20000]
        else:
            return text[-8000:]

    def update_moc(self, paper_title, folder_name):
        link = f"[[{folder_name}/{folder_name} - 000_Summary|{paper_title}]]"
        entry = f"- {link}\n"
        
        with open(self.moc_path, "a", encoding="utf-8") as f:
            f.write(entry)

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
                {"filename": "004_Analysis.md", "topic": "Analysis & Future Work"}
            ]
        
        file_names = [item['filename'] for item in structure]
        print(f"✓ Structure: {len(structure)} files")
        
        print(f"\n→ Step 3: Generating Content (expect 30-60s per file)...")
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
            print(f"Please add PDF files to the '{INPUT_PDF_FOLDER}' folder\n")
            return

        print(f"\n{'#'*60}")
        print(f"# RESEARCH PAPER TO OBSIDIAN VAULT CONVERTER")
        print(f"{'#'*60}")
        print(f"Model: {MODEL_NAME}")
        print(f"Timeout: {OLLAMA_TIMEOUT}s")
        print(f"Retries: {MAX_RETRIES}")
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
    
    # Check Ollama
    try:
        test = requests.get("http://localhost:11434/api/tags", timeout=10)
        print("✓ Ollama is running")
    except:
        print("✗ Ollama is NOT running!")
        print("  Start with: ollama serve")
        sys.exit(1)
    
    # Check model
    try:
        tags = test.json().get('models', [])
        model_names = [m['name'] for m in tags]
        if not any(MODEL_NAME in m for m in model_names):
            print(f"⚠ Model '{MODEL_NAME}' not found")
            print(f"  Pull with: ollama pull {MODEL_NAME}")
    except:
        pass
    
    print("="*60 + "\n")
    
    agent = ResearchAgent()
    agent.run()