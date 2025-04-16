import os
import sys
import pytesseract
import cv2
import spacy
import pandas as pd
from pdf2image import convert_from_path
import tempfile
import re

# Check if Tesseract is available
try:
    pytesseract.get_tesseract_version()
    print("Tesseract is available")
except Exception as e:
    print("Tesseract is not available or not in PATH.")
    print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
    print(f"Error: {str(e)}")
    sys.exit(1)

# Check if spaCy model is available
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully")
except Exception as e:
    print("spaCy model not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    print(f"Error: {str(e)}")
    sys.exit(1)

def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh)
        return text
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using OCR"""
    try:
        all_text = ""
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                page_text = extract_text_from_image(image_path)
                all_text += f"\n\n--- PAGE {i+1} ---\n\n{page_text}"
        return all_text
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return ""

def extract_financial_entities(text):
    """Extract financial entities from text using spaCy"""
    doc = nlp(text)
    
    # Create entity ruler for financial terms
    # Fix for newer spaCy versions where pipeline returns tuples
    pipeline_component_names = [name for name, _ in nlp.pipeline]
    if "entity_ruler" not in pipeline_component_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        
        # Define financial entity patterns
        financial_patterns = [
            {"label": "AMOUNT", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["million", "billion", "trillion"]}, "OP": "?"}]},
            {"label": "DATE_PATTERN", "pattern": [{"SHAPE": "dd"}, {"ORTH": "/"}, {"SHAPE": "dd"}, {"ORTH": "/"}, {"SHAPE": "dddd"}]},
            {"label": "PERCENTAGE", "pattern": [{"LIKE_NUM": True}, {"ORTH": "%"}]},
            {"label": "RATE", "pattern": [{"LOWER": "rate"}]},
            {"label": "CURRENCY", "pattern": [{"ORTH": {"IN": ["$", "€", "£", "¥"]}}, {"LIKE_NUM": True}]},
            {"label": "TERM", "pattern": [{"LOWER": {"IN": ["term", "maturity", "duration"]}}]},
            {"label": "ISSUER", "pattern": [{"LOWER": "issuer"}]},
            {"label": "INVESTOR", "pattern": [{"LOWER": {"IN": ["investor", "holder", "buyer"]}}]}
        ]
        ruler.add_patterns(financial_patterns)
    
    entities = []
    
    # Extract named entities
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_
        })
    
    # Extract regex patterns
    amount_pattern = r'\$\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:million|billion|trillion))?'
    percentage_pattern = r'\d+(?:\.\d+)?\s*%'
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
    
    # Find amounts
    for match in re.finditer(amount_pattern, text):
        entities.append({
            "text": match.group(),
            "label": "MONEY"
        })
    
    # Find percentages
    for match in re.finditer(percentage_pattern, text):
        entities.append({
            "text": match.group(),
            "label": "PERCENTAGE"
        })
    
    # Find dates
    for match in re.finditer(date_pattern, text):
        entities.append({
            "text": match.group(),
            "label": "DATE"
        })
    
    return entities

def process_document(file_path):
    """Main function to process a document"""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return
    
    print(f"Processing file: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_text = ""
    
    # Extract text based on file type
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        print("Detected image file, extracting text with OCR...")
        extracted_text = extract_text_from_image(file_path)
    elif file_extension == '.pdf':
        print("Detected PDF file, extracting text with OCR...")
        extracted_text = extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        print("Detected text file, reading content...")
        with open(file_path, 'r', encoding='utf-8') as f:
            extracted_text = f.read()
    else:
        print(f"Unsupported file type: {file_extension}")
        print("Supported file types: .pdf, .jpg, .jpeg, .png, .tif, .tiff, .bmp, .txt")
        return
    
    # Extract financial entities
    print("Extracting financial entities...")
    entities = extract_financial_entities(extracted_text)
    
    # Display results
    print("\n" + "="*50)
    print("DOCUMENT ANALYSIS RESULTS")
    print("="*50)
    
    # Show extracted text (first 500 chars)
    print("\nEXTRACTED TEXT (first 500 chars):")
    print("-"*50)
    print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
    
    # Show financial entities
    print("\nFINANCIAL ENTITIES:")
    print("-"*50)
    
    if not entities:
        print("No financial entities found")
    else:
        # Group entities by label
        entities_by_label = {}
        for entity in entities:
            label = entity['label']
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append(entity['text'])
        
        # Print grouped entities (remove duplicates)
        for label, texts in entities_by_label.items():
            unique_texts = list(set(texts))
            print(f"{label}:")
            for text in unique_texts[:10]:  # Limit to 10 entities per type
                print(f"  - {text}")
            if len(unique_texts) > 10:
                print(f"  ... and {len(unique_texts) - 10} more")
    
    # Save results
    output_file = os.path.splitext(file_path)[0] + "_analysis.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DOCUMENT ANALYSIS RESULTS\n")
            f.write("="*50 + "\n\n")
            
            f.write("EXTRACTED TEXT:\n")
            f.write("-"*50 + "\n")
            f.write(extracted_text + "\n\n")
            
            f.write("FINANCIAL ENTITIES:\n")
            f.write("-"*50 + "\n")
            if not entities:
                f.write("No financial entities found\n")
            else:
                for label, texts in entities_by_label.items():
                    unique_texts = list(set(texts))
                    f.write(f"{label}:\n")
                    for text in unique_texts:
                        f.write(f"  - {text}\n")
        
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_processor.py <file_path>")
        print("Example: python simple_processor.py sample.pdf")
        sys.exit(1)
    
    process_document(sys.argv[1]) 