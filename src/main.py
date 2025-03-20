"""
Main script to run the Agentic Reasoning System
"""
import os
import argparse
from dotenv import load_dotenv
from src.models.reasoning_system import ReasoningSystem
from src.config import MAX_ITERATIONS

# Load environment variables
load_dotenv()

def main():
    """
    Main function to run the reasoning system
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Agentic Reasoning System")
    parser.add_argument("--prompt", type=str, help="Initial prompt to seed the reasoning")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--load", type=str, help="Load state from file")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    args = parser.parse_args()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables.")
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not found in environment variables.")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize system
    if args.load:
        print(f"Loading state from {args.load}...")
        system = ReasoningSystem.load_state(args.load)
    else:
        system = ReasoningSystem(args.prompt)
    
    # Run iterations
    system.run_iterations(min(args.iterations, MAX_ITERATIONS), not args.quiet)
    
    # Save state
    output_files = system.save_state(args.output)
    
    print("\nRun completed!")
    print(f"Results saved to:")
    for key, path in output_files.items():
        print(f"- {key}: {path}")

if __name__ == "__main__":
    main() 