import json
import random


def generate_random_hex_color():
    """
    Generates a random hex color.

    Returns:
        str: A hex color string in the format '#RRGGBB'.
    """
    return '#{:06X}'.format(random.randint(0, 0xFFFFFF))


def generate_unique_random_colors(characters):
    """
    Generates a dictionary mapping each character to a unique random hex color.

    Args:
        characters (list): List of characters to map.

    Returns:
        dict: Dictionary mapping each character to a unique hex color.
    """
    color_mapping = {}
    used_colors = set()

    for char in characters:
        while True:
            color = generate_random_hex_color()
            if color not in used_colors:
                used_colors.add(color)
                color_mapping[char] = color
                break
    return color_mapping


def save_color_mapping(color_mapping, filename='color_mapping.json'):
    """
    Saves the color mapping to a JSON file.

    Args:
        color_mapping (dict): Dictionary mapping characters to hex colors.
        filename (str): Name of the JSON file to save.
    """
    data = {"characters": color_mapping}
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Color mapping successfully saved to '{filename}'.")
    except Exception as e:
        print(f"Error saving color mapping: {e}")


def main():
    # Define the list of characters to map
    characters = [
        " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",",
        "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F",
        "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
        "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "{", "|", "}", "~"
    ]

    # Generate the color mapping
    color_mapping = generate_unique_random_colors(characters)

    # Save the mapping to a JSON file
    save_color_mapping(color_mapping)


if __name__ == "__main__":
    main()
