import argparse
import os
import subprocess
import sys

def extract_figures(pdf_path, output_dir):
    """
    Uses the pdffigures2 command-line tool to extract images from a PDF.

    :param pdf_path: Path to the input PDF file.
    :param output_dir: Directory to save the extracted images and data.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {pdf_path}...")
    
    # Construct the command to run pdffigures2
    # The tool will save images and a JSON file with metadata in the output_dir
    command = [
        "pdffigures2",
        pdf_path,
        "-d",
        output_dir + os.path.sep, # Add separator for clarity
    ]
    
    try:
        # Run the command
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        print("Successfully extracted figures.")
        print("Output from pdffigures2:")
        print(process.stdout)

    except FileNotFoundError:
        print("Error: 'pdffigures2' command not found.", file=sys.stderr)
        print("Please ensure pdffigures2 is installed and in your system's PATH.", file=sys.stderr)
        print("See: https://github.com/allenai/pdffigures2", file=sys.stderr)
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing pdffigures2 for {pdf_path}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(
        description="Extract images from a PDF file using pdffigures2.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument("output_dir", help="Directory to save the extracted images.")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.pdf_path):
        print(f"Error: The file '{args.pdf_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    extract_figures(args.pdf_path, args.output_dir)

if __name__ == "__main__":
    main()

