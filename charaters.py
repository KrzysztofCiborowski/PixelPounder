import json
from PIL import Image

# Function to convert RGB to Hex
def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

# Load the color mapping from JSON file
with open('color_mapping.json', 'r') as f:
    color_mapping = json.load(f)

# Create reverse mapping: Hex Color -> Character
reverse_mapping = {value.upper(): key for key, value in color_mapping['characters'].items()}

# Open the image file
img = Image.open('hello.png').convert('RGB')
width, height = img.size

code_output = ''

# Process each pixel in the image
for y in range(height):
    for x in range(width):
        pixel_color = img.getpixel((x, y))
        hex_color = rgb_to_hex(pixel_color).upper()
        character = reverse_mapping.get(hex_color)
        if character is not None:
            code_output += character
        else:
            # Handle unknown colors (you can choose to skip or implement a tolerance)
            pass
    code_output += '\n'  # Newline at the end of each row

# Output the reconstructed code
print(code_output)

# Optionally, save the code to a file
with open('reconstructed_code.py', 'w') as f:
    f.write(code_output)