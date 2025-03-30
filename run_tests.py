#!/usr/bin/env python3
import subprocess
import sys
import os
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"🧪 {text} 🧪")
    print("="*80)

def run_test(test_script):
    """Run a test script and return success/failure"""
    print_header(f"Running {test_script}")
    
    try:
        # Run the test script and capture output
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Print the output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        # Return True if the script exited with code 0
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running {test_script}: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisite services are running"""
    print_header("Checking prerequisites")
    
    # Check if Weaviate is running
    try:
        import requests
        response = requests.get("http://localhost:8080/v1/.well-known/ready")
        if response.status_code == 200:
            print("✅ Weaviate is running")
            weaviate_running = True
        else:
            print(f"❌ Weaviate returned status code {response.status_code}")
            weaviate_running = False
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        weaviate_running = False
        
    # Check if the embedding API is running
    try:
        response = requests.get("http://localhost:8001/")
        if response.status_code == 200:
            print("✅ Embedding API is running")
            api_running = True
        else:
            print(f"❌ Embedding API returned status code {response.status_code}")
            api_running = False
    except Exception as e:
        print(f"❌ Failed to connect to Embedding API: {e}")
        api_running = False
    
    # Return overall status
    prerequisites_met = weaviate_running and api_running
    
    if not prerequisites_met:
        print("\n❌ Some prerequisites are not met. Please start the required services.")
        print("   - To start Weaviate: docker-compose up -d")
        print("   - To start Embedding API: python app.py")
    
    return prerequisites_met

def main():
    """Run all test scripts and summarize results"""
    print("\n🧪🧪🧪 MULTIMODAL SEARCH TESTING SUITE 🧪🧪🧪\n")
    
    # Check if prerequisites are met
    if not check_prerequisites():
        print("\nTests cannot be run until all prerequisites are met.")
        return 1
    
    # Define test scripts to run
    test_scripts = [
        "test_api_embedding.py",  # Test the embedding API
        "test_direct_claim_lookup.py",  # Test direct claim lookup
        "test_multimodal_search.py"  # Test multimodal search functionality
    ]
    
    # Run each test script and collect results
    test_results = {}
    
    for script in test_scripts:
        test_results[script] = run_test(script)
        # Add a short pause between tests
        time.sleep(2)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    all_passed = True
    for script, passed in test_results.items():
        result = "✅ PASS" if passed else "❌ FAIL"
        print(f"{result}: {script}")
        all_passed = all_passed and passed
    
    print("\nOVERALL RESULT:", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    
    # Return success/failure
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 