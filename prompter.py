import os


def format_py_files():
    # Get all .py files in the current directory
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and os.path.isfile(f)]

    # Create or overwrite the output file
    with open('formatted_py_files.txt', 'w') as output_file:
        for py_file in py_files:
            # Write the filename
            output_file.write(f"{py_file}\n")

            # Read and write the contents of the .py file
            with open(py_file, 'r') as file:
                output_file.write(file.read())

            # Add a newline between files for better readability
            output_file.write('\n\n')

    print("Formatting complete. Check 'formatted_py_files.txt' for the result.")


if __name__ == "__main__":
    format_py_files()