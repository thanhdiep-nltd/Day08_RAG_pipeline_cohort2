import sys
from pathlib import Path
from fpdf import FPDF

STANDARDIZED_DIR = Path("f:/PROJECT_VINUNI/LAB/Day08-lab-assignment/Day08_RAG_pipeline_cohort2/data/standardized")
OUTPUT_PDF = Path("f:/PROJECT_VINUNI/LAB/Day08-lab-assignment/Day08_RAG_pipeline_cohort2/data/pageindex_docs.pdf")

def generate_combined_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/Arial.ttf")
    pdf.add_font("ArialBold", "", "C:/Windows/Fonts/Arialbd.ttf") # For Bold title
    
    # Check font existence
    if not Path("C:/Windows/Fonts/Arial.ttf").exists():
        print("Arial font not found!")
        return
        
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file.parent) else "news"
        
        pdf.set_font("ArialBold", size=12)
        pdf.cell(0, 10, txt=f"--- DOCUMENT: {md_file.name} ({doc_type}) ---", ln=True)
        pdf.set_font("Arial", size=10)
        
        # Clean some weird carriage returns or characters
        clean_content = content.replace("\r", "")
        # multi_cell automatically handles encoding of unicode characters when uni=True (in modern fpdf2 it does by default)
        pdf.multi_cell(0, 5, txt=clean_content)
        pdf.cell(0, 10, txt="", ln=True)
        
    pdf.output(str(OUTPUT_PDF))
    print(f"✓ Generated PDF at: {OUTPUT_PDF}")

if __name__ == "__main__":
    generate_combined_pdf()
