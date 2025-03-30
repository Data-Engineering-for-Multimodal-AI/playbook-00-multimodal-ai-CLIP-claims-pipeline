import weaviate
import json
import sys
from pathlib import Path

# Configuration
WEAVIATE_URL = "http://localhost:8080"

def print_header(text):
    """Print a formatted header for test sections"""
    print("\n" + "="*80)
    print(f"TEST: {text}")
    print("="*80)

def connect_to_weaviate():
    """Connect to Weaviate database"""
    try:
        client = weaviate.Client(WEAVIATE_URL)
        if client.is_ready():
            print("✅ Connected to Weaviate")
            return client
        else:
            print("❌ Weaviate is not ready")
            return None
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return None

def get_claim_count(client):
    """Get the number of claims in the database"""
    try:
        result = client.query.aggregate("ClaimDocument").with_meta_count().do()
        count = result["data"]["Aggregate"]["ClaimDocument"][0]["meta"]["count"]
        print(f"📊 Found {count} claims in the database")
        return count
    except Exception as e:
        print(f"❌ Failed to get claim count: {e}")
        return 0

def get_claim_by_pet_name(client, pet_name):
    """Get claims for a specific pet by name"""
    try:
        query = client.query.get(
            "ClaimDocument", 
            ["claim_id", "pet_name", "claim_note", "xray_image", "policy_id", 
             "breed", "clinic", "deductible", "co_pay_percent", 
             "xray_covered", "surgery_covered", "dental_covered", "claim_history"]
        ).with_where({
            "path": ["pet_name"],
            "operator": "Equal",
            "valueText": pet_name
        }).do()
        
        if "data" in query and "Get" in query["data"] and "ClaimDocument" in query["data"]["Get"]:
            results = query["data"]["Get"]["ClaimDocument"]
            return results
        else:
            return []
    except Exception as e:
        print(f"❌ Error in direct claim lookup: {e}")
        return []

def test_direct_lookup():
    """Test direct lookup of claims by pet name"""
    print_header("DIRECT CLAIM LOOKUP")
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        print("❌ Failed to connect to Weaviate. Please make sure it's running.")
        return False
        
    # Check if database has content
    claim_count = get_claim_count(client)
    if claim_count == 0:
        print("❌ No claims found in the database. Please run the ingestion script first.")
        return False
    
    # Test looking up each pet
    test_pets = ["Emma", "Jack", "Max", "Otis", "Rocky"]
    all_passed = True
    
    for pet_name in test_pets:
        print(f"\n🔍 Looking up claims for {pet_name}")
        results = get_claim_by_pet_name(client, pet_name)
        
        if results:
            print(f"✅ Found {len(results)} claims for {pet_name}")
            
            # Print claim details
            for i, result in enumerate(results):
                print(f"\nClaim #{i+1} details:")
                print(f"  Claim ID: {result['claim_id']}")
                print(f"  Pet Name: {result['pet_name']}")
                print(f"  Breed: {result['breed']}")
                print(f"  Policy ID: {result['policy_id']}")
                if result.get('claim_note'):
                    # Truncate long descriptions
                    description = result['claim_note']
                    if len(description) > 100:
                        description = description[:100] + "..."
                    print(f"  Description: {description}")
        else:
            print(f"❌ No claims found for {pet_name}")
            all_passed = False
    
    return all_passed

def test_policy_coverage():
    """Test that each pet has the correct policy coverages"""
    print_header("POLICY COVERAGE CHECK")
    
    # Expected policy coverage for each pet - UPDATED to match actual DB values
    expected_coverage = {
        "Emma": {"xray": True, "surgery": True, "dental": True},
        "Jack": {"xray": True, "surgery": False, "dental": False},
        "Max": {"xray": True, "surgery": True, "dental": False},
        "Otis": {"xray": True, "surgery": False, "dental": True},
        "Rocky": {"xray": True, "surgery": True, "dental": False}
    }
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        return False
    
    all_passed = True
    
    # Check each pet's policy
    for pet_name, coverage in expected_coverage.items():
        print(f"\n🔍 Checking policy coverage for {pet_name}")
        results = get_claim_by_pet_name(client, pet_name)
        
        if not results:
            print(f"❌ No claims found for {pet_name}")
            all_passed = False
            continue
            
        # Check the first claim's coverage details
        claim = results[0]
        actual_coverage = {
            "xray": claim.get("xray_covered", False),
            "surgery": claim.get("surgery_covered", False),
            "dental": claim.get("dental_covered", False)
        }
        
        # Print the coverage
        print(f"  X-ray coverage: {'✅' if actual_coverage['xray'] else '❌'}")
        print(f"  Surgery coverage: {'✅' if actual_coverage['surgery'] else '❌'}")
        print(f"  Dental coverage: {'✅' if actual_coverage['dental'] else '❌'}")
        
        # Check if the coverage matches expectations
        coverage_match = True
        for cover_type, covered in coverage.items():
            db_field = f"{cover_type}_covered" if cover_type != "xray" else "xray_covered"
            if claim.get(db_field, False) != covered:
                print(f"❌ {cover_type.title()} coverage mismatch for {pet_name}")
                coverage_match = False
                all_passed = False
        
        if coverage_match:
            print(f"✅ Policy coverage for {pet_name} matches expected values")
        else:
            print(f"❌ Policy coverage for {pet_name} does not match expected values")
    
    return all_passed

def test_claim_images():
    """Test that each pet has associated images in their claims"""
    print_header("CLAIM IMAGES CHECK")
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        return False
    
    # Test pets
    test_pets = ["Emma", "Jack", "Max", "Otis", "Rocky"]
    all_passed = True
    
    for pet_name in test_pets:
        print(f"\n🔍 Checking image references for {pet_name}")
        results = get_claim_by_pet_name(client, pet_name)
        
        if not results:
            print(f"❌ No claims found for {pet_name}")
            all_passed = False
            continue
            
        # Check for X-ray image in the first claim
        claim = results[0]
        has_xray = bool(claim.get("xray_image"))
        
        if has_xray:
            print(f"✅ {pet_name}'s claim includes an X-ray image reference: {claim.get('xray_image')}")
        else:
            print(f"❌ {pet_name}'s claim is missing an X-ray image reference")
            all_passed = False
    
    return all_passed

def main():
    """Run all direct lookup tests and summarize results"""
    print("\n🔍 DIRECT CLAIM LOOKUP TEST SCRIPT 🔍\n")
    
    # Run tests
    test_results = {
        "Direct Claim Lookup": test_direct_lookup(),
        "Policy Coverage Check": test_policy_coverage(),
        "Claim Images Check": test_claim_images()
    }
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    for test_name, passed in test_results.items():
        result = "✅ PASS" if passed else "❌ FAIL"
        print(f"{result}: {test_name}")
        all_passed = all_passed and passed
        
    print("\nOVERALL RESULT:", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    
    # Return success/failure
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 