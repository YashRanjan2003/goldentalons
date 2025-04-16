import os
import sys
import json
from document_processor import DocumentProcessor

def print_banner():
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║  Term Sheet Analysis Tool                                  ║
║  - OCR Text Extraction                                     ║
║  - Financial Entity Recognition                            ║
║  - Table Extraction                                        ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)

def print_usage():
    print("""
Usage:
    python main.py <file_path> [options]

Options:
    --json          Output results in JSON format
    --save <path>   Save results to specified file path
    --help          Show this help message

Examples:
    python main.py sample.pdf
    python main.py contract.jpg --json
    python main.py termsheet.pdf --save results.json
    """)

def main():
    print_banner()
    
    # Parse command line arguments
    if len(sys.argv) < 2 or '--help' in sys.argv:
        print_usage()
        sys.exit(0)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    
    # Check for options
    output_json = '--json' in sys.argv
    save_path = None
    if '--save' in sys.argv:
        save_index = sys.argv.index('--save')
        if save_index + 1 < len(sys.argv):
            save_path = sys.argv[save_index + 1]
    
    # Process the document
    print(f"Processing file: {file_path}")
    processor = DocumentProcessor()
    result = processor.process_file(file_path)
    
    # Handle output
    if output_json:
        # Print JSON output
        print(json.dumps(result, indent=2))
    else:
        # Print formatted output
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\n--- FILE TYPE: {result['file_type'].upper()} ---")
            
            print("\n--- EXTRACTED TEXT (first 500 chars) ---")
            print(result['extracted_text'][:500] + "..." if len(result['extracted_text']) > 500 else result['extracted_text'])
            
            print("\n--- FINANCIAL ENTITIES ---")
            if not result['entities']:
                print("No financial entities found")
            else:
                print(f"Found {len(result['entities'])} financial entities")
                # Group entities by label
                entities_by_label = {}
                for entity in result['entities']:
                    label = entity['label']
                    if label not in entities_by_label:
                        entities_by_label[label] = []
                    entities_by_label[label].append(entity['text'])
                
                # Print grouped entities
                for label, entities in entities_by_label.items():
                    print(f"\n{label}:")
                    # Print unique entities
                    unique_entities = list(set(entities))
                    for entity in unique_entities[:10]:  # Limit to 10 per type
                        print(f"  - {entity}")
                    if len(unique_entities) > 10:
                        print(f"  ... and {len(unique_entities) - 10} more")
            
            print("\n--- TABLES ---")
            if len(result['tables']) == 0:
                print("No tables found")
            else:
                print(f"Found {len(result['tables'])} tables")
                for i, table in enumerate(result['tables'][:2]):  # Show just first 2 tables
                    print(f"\nTable {i+1}:")
                    try:
                        import pandas as pd
                        print(pd.DataFrame(table['data']).head().to_string())
                    except:
                        print(table['data'][:5])  # Fallback if pandas fails
                    
                    if i >= 1 and len(result['tables']) > 2:
                        print(f"\n... and {len(result['tables']) - 2} more tables (use --json for complete data)")
    
    # Save to file if requested
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to {save_path}")
        except Exception as e:
            print(f"\nError saving results: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 