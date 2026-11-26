"""Image rendering for the frozen encoder."""

from PIL import Image


def render_image_white(path):
    with Image.open(path) as source:
        rgba = source.convert('RGBA')
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert('RGB')
