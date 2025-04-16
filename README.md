# Term Sheet Analysis Tool

A Python tool for extracting and analyzing financial information from term sheets and other documents using:

1. **Tesseract OCR** - For text extraction from images and PDFs
2. **spaCy** - For financial entity recognition 
3. **Camelot** - For table extraction from PDFs

## Features

- Extract text from PDFs, images, Word documents, and text files
- Identify financial entities such as:
  - Monetary amounts
  - Dates
  - Percentages
  - Financial terms
  - Currencies
  - Organizations
- Extract and analyze tables from PDFs and Word documents
- Save results in JSON format for further processing

## Installation

### Prerequisites

- Python 3.7+
- Tesseract OCR installed on your system

### Install Tesseract OCR

#### Windows:
1. Download the installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your PATH

#### macOS:
```bash
brew install tesseract
```

#### Linux:
```bash
sudo apt install tesseract-ocr
```

### Install Required Python Packages

```bash
pip install -r requirements.txt
```

### Install spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```bash
python app/main.py <file_path>
```

### Options

- `--json`: Output results in JSON format
- `--save <path>`: Save results to specified file path
- `--help`: Show help message

### Examples

```bash
# Process a PDF file
python app/main.py sample.pdf

# Process an image file and output in JSON format
python app/main.py contract.jpg --json

# Process a PDF and save results to a file
python app/main.py termsheet.pdf --save results.json
```

## Supported File Types

- **PDF Documents** (.pdf)
- **Images** (.jpg, .jpeg, .png, .bmp, .tiff, .tif)
- **Word Documents** (.docx)
- **Text Files** (.txt)

## Output Format

The tool provides the following information:

1. **Extracted Text**: The full text content extracted from the document
2. **Financial Entities**: List of identified financial entities with their types
3. **Tables**: Structured data extracted from tables in the document

## Example

```
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  Term Sheet Analysis Tool                                  ║
║  - OCR Text Extraction                                     ║
║  - Financial Entity Recognition                            ║
║  - Table Extraction                                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

Processing file: sample_termsheet.pdf

--- FILE TYPE: PDF ---

--- EXTRACTED TEXT (first 500 chars) ---
TERM SHEET
Issuer: ABC Corporation
Date: 01/15/2023
Amount: $10,000,000
Interest Rate: 5.25%
Maturity: 5 years from date of issuance
...

--- FINANCIAL ENTITIES ---
Found 8 financial entities

ORGANIZATION:
  - ABC Corporation

MONEY:
  - $10,000,000

PERCENTAGE:
  - 5.25%

DATE:
  - 01/15/2023

--- TABLES ---
Found 2 tables

Table 1:
   Term             Value
0  Issuer           ABC Corporation
1  Principal Amount $10,000,000
2  Interest Rate    5.25%
3  Maturity         5 years

Processing complete!
```

## Limitations

- OCR accuracy depends on document quality
- Complex tables may not be extracted correctly
- PDF extraction requires clear, non-scanned documents for best results

## Troubleshooting

If you encounter issues with table extraction:
- Ensure your PDF has actual tables (not images of tables)
- For image-based PDFs, only text will be extracted through OCR

If entity recognition misses expected entities:
- Custom financial patterns can be added in `document_processor.py` 