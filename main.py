#!/usr/bin/env python3

import argparse
import json
import sys
import os
import platform
from PIL import Image
import tkinter as tk
from PIL import ImageTk
import numpy as np
from tkinter import filedialog
import cv2  # OpenCV library for advanced image processing
from collections import Counter
from math import sqrt

class PixelCodeRunner:
    def __init__(self, args):
        self.args = args
        self.code_output = ''
        self.char_lab_list = []  # List of tuples: (char, LAB)
        self.color_mapping = {}
        self.mapping_path = args.mapping if args.mapping else 'color_mapping.json'
        self.color_threshold = int(args.color_threshold) if args.color_threshold else 10  # Increased default threshold
        self.dynamic_threshold = args.dynamic_threshold

    def load_mapping(self):
        try:
            with open(self.mapping_path, 'r') as f:
                mapping_data = json.load(f)
            self.color_mapping = mapping_data['characters']
            # Validate color mapping
            self.validate_color_mapping()
            # Convert hex colors to LAB and store as a list
            for char, hex_color in self.color_mapping.items():
                rgb = self.hex_to_rgb(hex_color.upper())
                lab = self.rgb_to_lab(rgb)
                self.char_lab_list.append((char, tuple(lab)))
        except Exception as e:
            print(f"Error loading color mapping: {e}")
            sys.exit(1)

    def validate_color_mapping(self):
        """
        Ensures that each character has a unique and sufficiently distinct color.
        """
        colors = list(self.color_mapping.values())
        unique_colors = set(colors)
        if len(colors) != len(unique_colors):
            print("Error: Duplicate hex colors found in color mapping. Each character must have a unique hex color.")
            #sys.exit(1)
        # Optionally, check for minimum color distance
        min_distance = 20  # Minimum distance in LAB space
        chars = list(self.color_mapping.keys())
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                c1 = self.hex_to_rgb(colors[i].upper())
                c2 = self.hex_to_rgb(colors[j].upper())
                lab1 = self.rgb_to_lab(c1)
                lab2 = self.rgb_to_lab(c2)
                distance = self.color_distance(lab1, lab2)
                if distance < min_distance:
                    print(f"Warning: Colors for characters '{chars[i]}' and '{chars[j]}' are too similar (Distance: {distance:.2f}). Consider using more distinct colors.")
        print("Color mapping validation complete.")

    def rgb_to_lab(self, rgb):
        """
        Converts an RGB tuple to LAB color space using OpenCV.
        """
        rgb_array = np.uint8([[rgb]])  # Shape: (1,1,3)
        lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
        return lab[0][0]

    def lab_to_rgb(self, lab):
        """
        Converts a LAB tuple to RGB color space using OpenCV.
        """
        lab_array = np.uint8([[lab]])  # Shape: (1,1,3)
        rgb = cv2.cvtColor(lab_array, cv2.COLOR_LAB2RGB)
        return tuple(rgb[0][0])

    def rgb_to_hex(self, rgb):
        return '#{:02X}{:02X}{:02X}'.format(*rgb)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def color_distance(self, c1, c2):
        """
        Calculates the Euclidean distance between two LAB colors.
        """
        return sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

    def find_closest_color(self, target_color_lab):
        """
        Finds the closest matching character from the mapping based on LAB color distance.
        """
        closest_char = None
        min_distance = float('inf')
        for char, lab_color in self.char_lab_list:
            distance = self.color_distance(target_color_lab, lab_color)
            if distance < min_distance:
                min_distance = distance
                closest_char = char
        if self.dynamic_threshold:
            # If dynamic threshold is enabled, return the closest character regardless of distance
            return closest_char
        elif min_distance <= self.color_threshold:
            return closest_char
        else:
            return None

    def detect_cell_size(self, img):
        if self.args.cell_size:
            cell_size = int(self.args.cell_size)
            print(f"Using user-specified cell size: {cell_size}x{cell_size}")
            return cell_size

        print("Attempting to detect cell size automatically...")

        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Use Canny edge detector with lower thresholds to detect weaker edges
        edges = cv2.Canny(blurred, threshold1=20, threshold2=60, apertureSize=3, L2gradient=True)

        # Display the edges detected
        cv2.imshow('Edges Detected', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Use Hough Line Transform to detect lines in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=5, maxLineGap=2)

        if lines is None:
            print("No internal grid lines detected. Assuming a 1x1 grid.")
            cell_size = min(img.width, img.height)
            return cell_size

        # Collect line positions
        vertical_lines = []
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15 or abs(angle) > 165:  # Near horizontal
                horizontal_lines.extend([y1, y2])
            elif 75 < abs(angle) < 105:  # Near vertical
                vertical_lines.extend([x1, x2])

        if not vertical_lines or not horizontal_lines:
            print("Insufficient internal grid lines detected. Assuming a 1x1 grid.")
            cell_size = min(img.width, img.height)
            return cell_size

        # Remove duplicates and sort
        vertical_lines = sorted(set(vertical_lines))
        horizontal_lines = sorted(set(horizontal_lines))

        # Group lines that are close to each other to handle minor variations
        vertical_lines = self.group_lines(vertical_lines)
        horizontal_lines = self.group_lines(horizontal_lines)

        # Compute differences to find cell size
        vertical_diffs = np.diff(vertical_lines)
        horizontal_diffs = np.diff(horizontal_lines)

        if len(vertical_diffs) == 0 or len(horizontal_diffs) == 0:
            print("Could not compute cell size from detected grid lines. Assuming a 1x1 grid.")
            cell_size = min(img.width, img.height)
            return cell_size

        # Use median of differences as cell size
        cell_size_v = int(np.median(vertical_diffs))
        cell_size_h = int(np.median(horizontal_diffs))

        # Force square cells
        cell_size = int((cell_size_v + cell_size_h) / 2)
        print(f"Detected cell size (forced square): {cell_size}x{cell_size}")

        # Overlay detected grid lines on the image for verification
        grid_display = img_cv.copy()
        for x in vertical_lines:
            cv2.line(grid_display, (x, 0), (x, img.height), (0, 255, 0), 1)  # Green vertical lines
        for y in horizontal_lines:
            cv2.line(grid_display, (0, y), (img.width, y), (255, 0, 0), 1)  # Blue horizontal lines
        cv2.imshow('Detected Grid Lines', grid_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return cell_size

    def group_lines(self, lines, gap=5):
        """
        Groups lines that are within 'gap' pixels of each other.
        Returns the average position of each group.
        """
        if not lines:
            return []
        grouped_lines = []
        current_group = [lines[0]]
        for line in lines[1:]:
            if line - current_group[-1] <= gap:
                current_group.append(line)
            else:
                average_line = int(np.mean(current_group))
                grouped_lines.append(average_line)
                current_group = [line]
        # Append the last group
        average_line = int(np.mean(current_group))
        grouped_lines.append(average_line)
        return grouped_lines

    def get_dominant_color(self, img):
        """
        Gets the dominant color of the image by finding the most common color.
        """
        pixels = np.array(img).reshape(-1, 3)
        # Convert to list of tuples
        pixels = [tuple(pixel) for pixel in pixels]
        if not pixels:
            return (255, 255, 255)  # Default to white
        # Find the most common color
        most_common_color = Counter(pixels).most_common(1)[0][0]
        return most_common_color

    def select_image_file(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.update()
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        root.destroy()
        return file_path

    def decode_image(self):
        # If image path is not provided, prompt the user to select a file
        if not self.args.image:
            image_path = self.select_image_file()
            if not image_path:
                print("No image file selected.")
                sys.exit(1)
        else:
            image_path = self.args.image

        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image: {e}")
            sys.exit(1)

        # Detect cell size
        cell_size = self.detect_cell_size(img)
        img_width, img_height = img.size

        if cell_size == 0:
            print("Detected cell size is zero. Exiting.")
            sys.exit(1)

        cols = img_width // cell_size
        rows = img_height // cell_size

        print(f"Number of columns: {cols}, Number of rows: {rows}")

        self.code_output = ''

        for row in range(rows):
            line_chars = ''
            for col in range(cols):
                left = col * cell_size
                upper = row * cell_size
                right = left + cell_size
                lower = upper + cell_size

                # Handle edge cases where the grid might not perfectly divide the image
                if right > img_width:
                    right = img_width
                if lower > img_height:
                    lower = img_height

                # Crop the cell
                cell = img.crop((left, upper, right, lower))

                # Get the dominant color of the cell
                cell_color = self.get_dominant_color(cell)
                cell_color_lab = self.rgb_to_lab(cell_color)
                character = self.find_closest_color(cell_color_lab)

                # Debug: Print cell position, color, and matched character
                # Uncomment the next line for debugging
                # print(f"Row {row}, Col {col}: Color {self.rgb_to_hex(cell_color)} -> {character}")

                # Ignore background pixels (assuming white and black backgrounds)
                if character is None:
                    # Optionally, ignore certain colors or log them
                    bg_colors = [tuple(self.hex_to_rgb(c)) for c in ['#FFFFFF', '#000000']]
                    if cell_color in bg_colors:
                        continue
                    else:
                        print(f"Warning: No matching character for color {self.rgb_to_hex(cell_color)} at Row {row}, Col {col}. Skipping.")
                        continue

                line_chars += character
            if line_chars.strip() != '':
                self.code_output += line_chars + '\n'  # Add newline after each line

        # Remove any extra newlines that may have been added
        self.code_output = self.code_output.rstrip('\n')

        if not self.code_output:
            print("No characters were decoded from the image.")
            sys.exit(1)

        # Output the reconstructed code
        print("Reconstructed Code:")
        print("-" * 40)
        print(self.code_output)
        print("-" * 40)

        # Save the code to a file for review
        temp_code_file = 'reconstructed_code.py'
        with open(temp_code_file, 'w') as f:
            f.write(self.code_output)

        # Ask the user for confirmation before executing
        while True:
            execute = input("Do you want to execute the reconstructed code? (y/n): ").strip().lower()
            if execute == 'y':
                break
            elif execute == 'n':
                print("Execution aborted.")
                os.remove(temp_code_file)
                sys.exit(0)

        # Execute the code
        try:
            exec(compile(self.code_output, '<string>', 'exec'))
        except Exception as e:
            print(f"Error executing code: {e}")
        finally:
            os.remove(temp_code_file)

    def encode_code(self):
        # Determine the source of the code: file or string
        if self.args.encode:
            code_source = 'file'
            code_path = self.args.encode
            try:
                with open(code_path, 'r') as f:
                    code_text = f.read()
            except Exception as e:
                print(f"Error reading code file: {e}")
                sys.exit(1)
        elif self.args.string:
            code_source = 'string'
            code_text = ' '.join(self.args.string)
        else:
            print("No code input provided. Use -e to encode a file or -s to encode a string.")
            sys.exit(1)

        output_image = self.args.output if self.args.output else 'encoded_image.png'

        # Ensure that newline characters are preserved
        if '\n' not in self.color_mapping:
            # Assign a unique color to newline character if not present
            self.color_mapping['\n'] = '#010101'  # A color close to black but not black
            lab = self.rgb_to_lab(self.hex_to_rgb('#010101'))
            self.char_lab_list.append(('\n', tuple(lab)))

        # Remove any characters not in the mapping
        valid_characters = set(self.color_mapping.keys())
        filtered_code = ''.join(c if c in valid_characters else '' for c in code_text)

        # Split code into lines
        code_lines = filtered_code.split('\n')

        # Determine image dimensions
        max_line_length = max(len(line) for line in code_lines) if code_lines else 1
        width = max_line_length if max_line_length > 0 else 1
        height = len(code_lines) if len(code_lines) > 0 else 1

        # Create a new image
        img = Image.new('RGB', (width, height), color='white')  # Default color is white

        # Encode the code into the image
        for y, line in enumerate(code_lines):
            for x, char in enumerate(line):
                hex_color = self.color_mapping.get(char)
                if hex_color:
                    rgb_color = self.hex_to_rgb(hex_color)
                    img.putpixel((x, y), rgb_color)
                else:
                    # Skip characters not in mapping
                    pass

        # Save the image
        try:
            img.save(output_image)
            print(f"Code has been encoded into the image '{output_image}'.")
        except Exception as e:
            print(f"Error saving image: {e}")
            sys.exit(1)

        # Display the image after encoding
        self.display_image(img)

    def display_image(self, img):
        # Determine the operating system
        if self.args.os:
            os_name = self.args.os.lower()
        else:
            os_name = platform.system().lower()

        try:
            if 'windows' in os_name:
                # Use Tkinter to display the image on Windows
                root = tk.Tk()
                root.title("Encoded Image")
                # Resize the image for better visibility
                scaling_factor = 10
                resized_img = img.resize(
                    (img.width * scaling_factor, img.height * scaling_factor),
                    Image.NEAREST
                )
                img_tk = ImageTk.PhotoImage(resized_img)
                label = tk.Label(root, image=img_tk)
                label.pack()
                root.mainloop()
            else:
                # For Linux and macOS, open the image normally
                output_image = self.args.output if self.args.output else 'encoded_image.png'
                opener = 'xdg-open' if 'linux' in os_name else 'open'
                os.system(f"{opener} '{output_image}'")
        except Exception as e:
            print(f"Error displaying image: {e}")

    def run(self):
        self.load_mapping()
        if self.args.action == 'decode':
            self.decode_image()
        elif self.args.action == 'encode':
            self.encode_code()
        else:
            print("No action specified. Use 'encode' or 'decode' as the action.")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Encode or decode code in images.')
    parser.add_argument('action', choices=['encode', 'decode'],
                        help='Action to perform: encode or decode.')
    parser.add_argument('-i', '--image',
                        help='Path to the image file to decode and execute.')
    parser.add_argument('-e', '--encode',
                        help='Path to the code file (.py) to encode into an image.')
    parser.add_argument('-s', '--string', nargs='+',
                        help='String of text to encode into an image.')
    parser.add_argument('-o', '--output',
                        help='Output image file name when encoding.')
    parser.add_argument('-m', '--mapping',
                        help='Path to the color mapping JSON file.')
    parser.add_argument('--os',
                        help='Specify the operating system (windows, linux, mac).')
    parser.add_argument('--cell-size',
                        help='Specify the cell size (square) for decoding.')
    parser.add_argument('--color-threshold', default=5,
                        help='Color distance threshold for matching (default: 10).')
    parser.add_argument('--dynamic-threshold', action='store_true',
                        help='Enable dynamic color threshold based on color distances.')
    args = parser.parse_args()

    runner = PixelCodeRunner(args)
    runner.run()


if __name__ == '__main__':
    main()