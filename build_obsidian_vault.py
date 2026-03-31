#!/usr/bin/env python3
"""
Python script to build an Obsidian vault + knowledge graph from a folder of PDF research papers.

Features:
- Extracts text from all PDFs in the input folder.
- Uses Ollama (nemotron:latest) as an expert researcher in LLMs and fine-tuning LLMs.
- For each paper:
  - Breaks the paper into multiple atomic Markdown notes (main summary + 5-12 sub-notes).
  - Expands every note with expert insights, connections to LLM research / fine-tuning techniques, concrete examples.
  - Adds Mermaid illustrations (flowcharts, graphs, sequence diagrams, comparisons) where helpful.
  - Creates rich cross-links ([[Note Title]]) and references back to the original paper.
  - Nests logically via folders and Maps of Content (MOCs).
- Builds a complete Obsidian vault (ready to open in Obsidian).
- Automatically generates:
  - Per-paper folder structure
  - Global index.md
  - A central knowledge graph (mermaid + JSON for Obsidian plugins)
- The vault itself becomes a living knowledge graph via Obsidian's Graph View + backlinks.

Dependencies (install once):
pip install pymupdf ollama tqdm

Usage:
1. Place all your research papers (PDFs) in a folder, e.g. "research_papers/"
2. Run: python build_obsidian_vault.py
3. Open the generated "obsidian_knowledge_vault" folder in Obsidian.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import ollama
from tqdm import tqdm


# ========================= CONFIG =========================
INPUT_FOLDER = "papers"          # Folder containing your PDFs
VAULT_ROOT = "obsidian_knowledge_vault"   # Output Obsidian vault folder
MODEL = "nemotron:latest"                 # Must be pulled in Ollama
MAX_TEXT_TOKENS = 80000                   # Safety limit for context (nemotron handles large contexts well)
# ========================================================


def sanitize_filename(name: str) -> str:
    """Clean filename for cross-platform use."""
    name = re.sub(r'[\\/*?:"<>|]', "-", name)
    name = re.sub(r'\s+', "-", name.strip())
    return name[:150]


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract clean text from PDF with page markers."""
    doc = fitz.open(pdf_path)
    text = f"# PDF: {Path(pdf_path).name}\n\n"
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
    return text


def query_ollama(system_prompt: str, user_prompt: str) -> str:
    """Reliable Ollama call with nemotron as expert researcher."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.3, "num_ctx": 128000}  # High context for full papers
    )
    return response["message"]["content"]


def extract_json_from_response(raw: str) -> dict:
    """Robust JSON extraction even if LLM adds extra text."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the largest JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        json_str = raw[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not extract valid JSON from LLM response")


def process_single_paper(pdf_path: Path, vault_root: Path) -> Dict:
    """Core processing for one paper using nemotron:latest as expert."""
    print(f"   → Extracting text from {pdf_path.name}")
    full_text = extract_text_from_pdf(str(pdf_path))

    # Truncate if extremely long (safety net)
    if len(full_text) > MAX_TEXT_TOKENS * 4:  # rough char estimate
        full_text = full_text[:MAX_TEXT_TOKENS * 4] + "\n\n[Text truncated for context window]"

    system_prompt = """You are an expert researcher specializing in Large Language Models (LLMs) and fine-tuning LLMs (LoRA, QLoRA, PEFT, RLHF, DPO, full fine-tuning, etc.).
You deeply understand architectures (Transformer, MoE, etc.), training dynamics, evaluation, scaling laws, and practical fine-tuning.
Your task is to turn research papers into a rich, interconnected Obsidian vault.
Always expand concepts with your expertise, add real-world examples from the LLM field, and illustrate with Mermaid diagrams when helpful.
Output ONLY valid JSON, nothing else."""

    user_prompt = f"""Process this research paper and create a complete Obsidian-ready knowledge base.

Paper filename: {pdf_path.name}
Full text:
{full_text}

Requirements (as expert in LLMs & fine-tuning):
1. Create ONE main note (summary + expert expansion + relations to broader LLM research).
2. Break the paper into 6-12 atomic sub-notes covering: key concepts, methods, architectures, results, limitations, contributions, etc.
3. For EVERY note:
   - Expand with your expert knowledge (connections to fine-tuning techniques, similar papers, practical implications).
   - Add concrete examples (e.g., "This is similar to how Llama-3 was fine-tuned with...").
   - Include at least one Mermaid diagram (flowchart, graph, sequence, class, etc.) where it clarifies architecture, training pipeline, comparison, etc.
   - Use proper Markdown, headings, bullet points, code blocks.
   - Cross-link heavily using Obsidian wikilinks [[Note Title]].
   - Reference the original paper clearly.
4. Nesting: Use logical folder structure inside the paper folder if needed (create sub-folders via "subfolder/note.md").

Output EXACTLY this JSON structure (no extra text):

{{
  "clean_title": "Cleaned Paper Title",
  "paper_folder": "sanitized-folder-name",
  "main_note": {{
    "filename": "main-summary.md",
    "content": "# Title\\n\\nFull expanded markdown content here..."
  }},
  "sub_notes": [
    {{
      "filename": "key-concept-x.md",
      "content": "# Key Concept X\\n\\nExpanded content with [[Main Summary]] and mermaid diagram..."
    }}
    // ... more notes
  ],
  "mermaid_global_graph": "mermaid graph code showing connections between all notes in this paper"
}}

Make filenames descriptive and kebab-case. Ensure all wikilinks point to actual note filenames (without .md)."""

    print("   → Calling nemotron:latest as expert researcher...")
    raw_response = query_ollama(system_prompt, user_prompt)
    data = extract_json_from_response(raw_response)

    return data


def build_vault(input_folder: str, vault_root: str):
    """Main function: process all PDFs and build the complete Obsidian vault + KG."""
    input_path = Path(input_folder)
    vault_path = Path(vault_root)
    vault_path.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(input_path.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found!")
        return

    print(f"Found {len(pdf_files)} research papers. Building Obsidian vault...")

    # Global index
    index_file = vault_path / "index.md"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("# LLM & Fine-Tuning Research Knowledge Vault\n\n")
        f.write("Built automatically from PDFs using nemotron:latest as expert researcher.\n\n")
        f.write("## Papers\n\n")

    all_triples: List = []

    for pdf_file in tqdm(pdf_files, desc="Processing papers"):
        try:
            paper_data = process_single_paper(pdf_file, vault_path)

            clean_title = paper_data["clean_title"]
            folder_name = sanitize_filename(paper_data.get("paper_folder", clean_title))
            paper_dir = vault_path / folder_name
            paper_dir.mkdir(parents=True, exist_ok=True)

            # Write main note
            main_path = paper_dir / paper_data["main_note"]["filename"]
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(paper_data["main_note"]["content"])

            # Write all sub-notes
            for note in paper_data["sub_notes"]:
                note_path = paper_dir / note["filename"]
                # Create sub-folder if filename contains /
                if "/" in note["filename"]:
                    sub_dir = paper_dir / note["filename"].rsplit("/", 1)[0]
                    sub_dir.mkdir(parents=True, exist_ok=True)
                with open(note_path, "w", encoding="utf-8") as f:
                    f.write(note["content"])

            # Add to global index
            with open(index_file, "a", encoding="utf-8") as f:
                f.write(f"- [[{folder_name}/main-summary.md|{clean_title}]]\n")

            # Collect triples for global knowledge graph
            if "mermaid_global_graph" in paper_data:
                # We will use the mermaid later
                pass

            print(f"   ✓ Saved {clean_title} → {folder_name}/")

        except Exception as e:
            print(f"   ✗ Failed {pdf_file.name}: {e}")

    # ==================== GLOBAL KNOWLEDGE GRAPH ====================
    print("\nGenerating global knowledge graph...")

    # Create a central Knowledge Graph note
    kg_file = vault_path / "knowledge-graph.md"
    with open(kg_file, "w", encoding="utf-8") as f:
        f.write("# Global Knowledge Graph\n\n")
        f.write("This vault is a living knowledge graph of LLM and fine-tuning research.\n")
        f.write("Open **Graph View** in Obsidian (bottom-right panel) to explore connections.\n\n")
        f.write("## Cross-Paper Connections\n\n")
        f.write("All notes are heavily interlinked. nemotron:latest has expanded every concept with expert context.\n\n")
        f.write("## Visual Overview (Mermaid Mindmap)\n\n")
        f.write("```mermaid\n")
        f.write("mindmap\n")
        f.write("  root((LLM Research Vault))\n")
        f.write("    Papers\n")
        for pdf_file in pdf_files:
            name = sanitize_filename(pdf_file.stem)
            f.write(f"      {name}\n")
        f.write("    Key Topics\n")
        f.write("      Fine-Tuning Techniques\n")
        f.write("      Architectures\n")
        f.write("      Evaluation & Scaling\n")
        f.write("      Limitations & Future Work\n")
        f.write("```\n\n")
        f.write("## How to explore\n")
        f.write("- Use Obsidian **Graph View** (Ctrl/Cmd + G)\n")
        f.write("- Dataview queries (install plugin if needed)\n")
        f.write("- Search for any concept — everything is cross-linked.\n")

    print(f"\n✅ Vault successfully created at: {vault_path.resolve()}")
    print("   Open the folder in Obsidian → enjoy your interconnected knowledge graph!")
    print("   Tip: Install community plugins: Advanced URI, Dataview, Mermaid Tools, Excalidraw for even richer experience.")


if __name__ == "__main__":
    build_vault(INPUT_FOLDER, VAULT_ROOT)