#!/usr/bin/env python3
"""
Test runner script for LOOK-DGC
Provides convenient commands for running different types of tests
"""

import os
import sys
import argparse
import subprocess

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def install_test_deps():
    """Install test dependencies"""
    print("Installing test dependencies...")
    return run_command("pip install -r test_requirements.txt", cwd="tests")

def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    print("Running unit tests...")
    cmd = "python -m pytest unit/"
    if verbose:
        cmd += " -v"
    if coverage:
        cmd += " --cov=../gui --cov-report=html --cov-report=term"
    return run_command(cmd, cwd="tests")

def run_integration_tests(verbose=False):
    """Run integration tests"""
    print("Running integration tests...")
    cmd = "python -m pytest integration/"
    if verbose:
        cmd += " -v"
    return run_command(cmd, cwd="tests")

def run_all_tests(verbose=False, coverage=False):
    """Run all tests"""
    print("Running all tests...")
    cmd = "python -m pytest"
    if verbose:
        cmd += " -v"
    if coverage:
        cmd += " --cov=../gui --cov-report=html --cov-report=term"
    return run_command(cmd, cwd="tests")

def main():
    parser = argparse.ArgumentParser(description="LOOK-DGC Test Runner")
    parser.add_argument("--install", action="store_true", 
                       help="Install test dependencies")
    parser.add_argument("--unit", action="store_true", 
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests only")
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--setup", action="store_true",
                       help="Install dependencies and run all tests")
    
    args = parser.parse_args()
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # If no specific test type requested, run all
    if not any([args.unit, args.integration, args.all]):
        args.all = True
    
    success = True
    
    if args.install or args.setup:
        if not install_test_deps():
            success = False
            if not args.setup:  # Only exit if not doing full setup
                return
    
    if args.setup:
        args.all = True
        args.verbose = True
        args.coverage = True
    
    if args.unit:
        if not run_unit_tests(args.verbose, args.coverage):
            success = False
    
    if args.integration:
        if not run_integration_tests(args.verbose):
            success = False
    
    if args.all:
        if not run_all_tests(args.verbose, args.coverage):
            success = False
    
    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report available in tests/htmlcov/")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()