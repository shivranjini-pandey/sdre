"""
Load sample documents for testing and demo.
"""

import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal, init_db
from app.ingest import get_ingester

def load_sample_docs():
    """Load sample documents."""
    # Initialize DB
    init_db()
    db = SessionLocal()
    
    try:
        ingester = get_ingester(db)
        
        # Create sample documents
        sample_docs = [
            {
                "title": "Employment Contract",
                "content": """
EMPLOYMENT CONTRACT

This Employment Contract ("Contract") is entered into as of January 1, 2024,
between ABC Corporation ("Company") and John Doe ("Employee").

1. POSITION AND DUTIES
Employee shall serve as Senior Software Engineer and shall have the responsibilities
normally associated with such position.

2. COMPENSATION
Employee's annual salary shall be $150,000, payable in accordance with Company's
standard payroll practices.

3. TERMINATION
Either party may terminate this Contract with 30 days written notice.
In case of termination without cause, Employee is entitled to severance pay
equal to one month's salary per year of service.

4. NON-COMPETE
Employee agrees not to engage in any competitive business within the same industry
for a period of 12 months following termination.

5. CONFIDENTIALITY
All company information and trade secrets remain confidential and belong to Company.
""",
            },
            {
                "title": "Insurance Policy",
                "content": """
INSURANCE POLICY DOCUMENT

Policy Number: POL-2024-001234
Effective Date: January 1, 2024
Expiration Date: December 31, 2024

COVERAGE DETAILS

1. COVERED PERILS
This policy covers:
- Fire and smoke damage
- Water damage from burst pipes (but NOT flooding)
- Theft and burglary
- Wind and hail damage

2. EXCLUSIONS
The following are NOT covered:
- Flood or water from external sources
- War or civil unrest
- Earthquake damage
- Damage from lack of maintenance

3. DEDUCTIBLE
The standard deductible is $500 for most claims.
For water damage claims, deductible is $1,000.

4. POLICY LIMITS
Maximum coverage: $250,000
Personal property: $50,000
Additional living expenses: $25,000

5. CLAIMS PROCESS
To file a claim, contact our claims department within 30 days.
Required documentation includes photos, receipts, and proof of loss.

6. RENEWAL
This policy automatically renews annually unless cancelled.
""",
            },
        ]
        
        # Save to temp files and ingest
        for i, doc in enumerate(sample_docs):
            file_path = f"/tmp/sample_doc_{i}.txt"
            with open(file_path, "w") as f:
                f.write(doc["content"])
            
            ingester.ingest_file(file_path, title=doc["title"])
            print(f"✓ Loaded: {doc['title']}")
        
        print("\n✓ Sample documents loaded successfully")
    
    finally:
        db.close()

if __name__ == "__main__":
    load_sample_docs()
