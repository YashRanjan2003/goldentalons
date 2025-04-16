import os
from document_processor import DocumentProcessor

def test_document_processor():
    """
    Simple test function to demonstrate the document processor functionality.
    This can be used for testing without having real documents.
    """
    # Create a mock document processor
    processor = DocumentProcessor()
    
    # Test with a text string that contains financial information
    test_text = """
    TERM SHEET
    
    Issuer: ABC Corporation
    Date: 01/15/2023
    Amount: $10,000,000
    Interest Rate: 5.25%
    Term: 5 years
    Maturity Date: 01/15/2028
    
    The investor will purchase senior secured notes from ABC Corporation
    with a principal amount of $10 million. The notes will have a coupon rate
    of 5.25% per annum, payable quarterly on the 15th of January, April, July, and October.
    
    Payment Schedule:
    
    Year | Principal Payment | Interest Payment | Total Payment
    -----|-------------------|------------------|---------------
    2023 | $0                | $525,000         | $525,000
    2024 | $0                | $525,000         | $525,000
    2025 | $0                | $525,000         | $525,000
    2026 | $0                | $525,000         | $525,000
    2027 | $0                | $525,000         | $525,000
    2028 | $10,000,000       | $525,000         | $10,525,000
    """
    
    # Extract entities from the text
    entities = processor.extract_financial_entities(test_text)
    
    # Print the results
    print("\n=== TEST DOCUMENT PROCESSOR ===\n")
    print("Sample Term Sheet Text:")
    print("-" * 40)
    print(test_text)
    print("-" * 40)
    
    print("\nExtracted Financial Entities:")
    print("-" * 40)
    if not entities:
        print("No entities found")
    else:
        # Group entities by label
        entities_by_label = {}
        for entity in entities:
            label = entity['label']
            if label not in entities_by_label:
                entities_by_label[label] = []
            entities_by_label[label].append(entity['text'])
        
        # Print grouped entities
        for label, entities in entities_by_label.items():
            print(f"{label}:")
            # Print unique entities
            unique_entities = list(set(entities))
            for entity in unique_entities:
                print(f"  - {entity}")
    
    print("\nTest completed successfully!")
    return entities

if __name__ == "__main__":
    test_document_processor() 