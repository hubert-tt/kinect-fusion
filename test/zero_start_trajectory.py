import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python zero_start_trajectory.py <path_to_file>")
        return

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return

    with open(input_path, 'r') as f:
        lines = f.readlines()

    header = []
    data_lines = []

    for line in lines:
        if line.startswith('#'):
            header.append(line)
        else:
            data_lines.append(line)

    if not data_lines:
        print("No data to process.")
        return

    first_timestamp = float(data_lines[0].split()[0])

    output_lines = []
    for line in data_lines:
        parts = line.strip().split()
        timestamp = float(parts[0]) - first_timestamp
        new_line = f"{timestamp:.4f} " + " ".join(parts[1:]) + "\n"
        output_lines.append(new_line)

    output_filename = 'zerod_' + os.path.basename(input_path)
    output_path = os.path.join(os.path.dirname(input_path), output_filename)

    with open(output_path, 'w') as f:
        f.writelines(header)
        f.writelines(output_lines)

    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()
