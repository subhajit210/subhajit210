
import math
import sys
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps

try:
    import numpy as np
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


# config

INPUT_IMAGE = Path(r"../files/reference_dragon_knight.png")
OUTPUT_SVG  = Path(r"../files/dragon_knight_static.svg")

COLS = 260
CHAR_ASPECT = 0.48

FONT_SIZE = 8
CHAR_WIDTH = 4.8   # px per character (monospace)
LINE_HEIGHT = 9.2   # px per row

DARK_THRESHOLD = 12

# Carefully ordered by visual density (lightest to heaviest)
RAMP = " .`'^\":;~-_+<>i!lI?/\\|()1{}[]rcvunxzjftLCJUYkXZO0Qoahbdpqwm*WMB8&#%$@"

# Fire-specific characters for bright orange/yellow areas
FIRE_CHARS = "*%#@WMB8&"


def load_and_prepare(path, cols, char_aspect):
    """Load image with enhancements."""
    img = Image.open(path).convert("RGB")
    
    # Boost contrast and sharpness
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Color(img).enhance(1.15)
    
    orig_w, orig_h = img.size
    rows = int(cols * (orig_h / orig_w) * char_aspect)
    
    # Resize with supersampling for quality
    # First resize to 2x target, then downsample
    img_2x = img.resize((cols * 2, rows * 2), Image.Resampling.LANCZOS)
    img_final = img_2x.resize((cols, rows), Image.Resampling.LANCZOS)
    
    return img_final, cols, rows


def compute_edges(lum):
    """Sobel edge detection."""
    padded = np.pad(lum, 1, mode='edge')
    
    gx = (-padded[:-2, :-2] + padded[:-2, 2:]
          - 2*padded[1:-1, :-2] + 2*padded[1:-1, 2:]
          - padded[2:, :-2] + padded[2:, 2:])
    
    gy = (-padded[:-2, :-2] - 2*padded[:-2, 1:-1] - padded[:-2, 2:]
          + padded[2:, :-2] + 2*padded[2:, 1:-1] + padded[2:, 2:])
    
    return np.sqrt(gx**2 + gy**2), np.arctan2(gy, gx)


def is_fire_color(r, g, b):
    """Detect if a pixel is in the fire/flame color range."""
    return r > 150 and g > 80 and g < r and b < g * 0.7


def pick_char(lum_val, edge_mag, edge_dir, r, g, b, ramp):
    """Pick the best ASCII character."""
    if lum_val < DARK_THRESHOLD:
        return ' '  # space for background
    
    # Strong edge? Use directional char
    if edge_mag > 120:
        deg = math.degrees(edge_dir) % 180
        if edge_mag > 200:
            if deg < 22.5 or deg > 157.5:
                return '-'
            elif 67.5 < deg < 112.5:
                return '|'
            elif 22.5 <= deg <= 67.5:
                return '\\'
            else:
                return '/'
    
    # Normalize luminance
    norm = min(lum_val / 255.0, 1.0)
    gamma = norm ** 0.7
    
    # For fire regions, use denser/brighter characters
    if is_fire_color(int(r), int(g), int(b)):
        fire_idx = int(gamma * (len(FIRE_CHARS) - 1))
        fire_idx = max(0, min(fire_idx, len(FIRE_CHARS) - 1))
        # Blend: use fire chars for bright areas, ramp for dimmer
        if gamma > 0.4:
            return FIRE_CHARS[fire_idx]
    
    idx = int(gamma * (len(ramp) - 1))
    idx = max(0, min(idx, len(ramp) - 1))
    return ramp[idx]


def boost_color(r, g, b):
    """Boost color for dark background visibility."""
    # Increase saturation and brightness
    boost = 1.2
    r_out = min(255, int(r * boost))
    g_out = min(255, int(g * boost))
    b_out = min(255, int(b * boost))
    
    # Minimum brightness for visibility
    mx = max(r_out, g_out, b_out)
    if mx < 30:
        return None
    
    return (r_out, g_out, b_out)


def escape_xml(ch):
    """Escape for XML."""
    if ch == '&': return '&amp;'
    if ch == '<': return '&lt;'
    if ch == '>': return '&gt;'
    if ch == '"': return '&quot;'
    if ch == "'": return '&apos;'
    return ch


def generate_svg(img_array, lum, edge_mag, edge_dir, cols, rows, ramp):
    """Generate SVG using row-based text for better rendering."""
    
    # Normalize edges
    emax = edge_mag.max()
    if emax > 0:
        edge_norm = edge_mag / emax * 255
    else:
        edge_norm = edge_mag
    
    svg_w = int(cols * CHAR_WIDTH) + 20
    svg_h = int(rows * LINE_HEIGHT) + 16
    
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="{svg_w}" height="{svg_h}">')
    parts.append(f'  <rect width="{svg_w}" height="{svg_h}" fill="#0d1117"/>')
    parts.append(f'  <style>')
    parts.append(f'    text {{')
    parts.append(f'      font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", "Courier New", monospace;')
    parts.append(f'      font-size: {FONT_SIZE}px;')
    parts.append(f'      white-space: pre;')
    parts.append(f'    }}')
    parts.append(f'  </style>')
    
    char_count = 0
    
    for row in range(rows):
        y = int(row * LINE_HEIGHT + FONT_SIZE) + 4
        
        # Build row character by character
        row_has_content = False
        
        for col in range(cols):
            r = img_array[row, col, 0]
            g = img_array[row, col, 1]
            b = img_array[row, col, 2]
            l = lum[row, col]
            em = edge_norm[row, col]
            ed = edge_dir[row, col]
            
            ch = pick_char(l, em, ed, r, g, b, ramp)
            
            if ch == ' ':
                continue
            
            color = boost_color(r, g, b)
            if color is None:
                continue
            
            row_has_content = True
            x = int(col * CHAR_WIDTH) + 8
            
            cr, cg, cb = color
            hex_color = f"#{cr:02x}{cg:02x}{cb:02x}"
            ch_esc = escape_xml(ch)
            
            parts.append(f'  <text x="{x}" y="{y}" fill="{hex_color}">{ch_esc}</text>')
            char_count += 1
    
    parts.append('</svg>')
    
    print(f"  -> {char_count} chars, {rows}x{cols} grid, {svg_w}x{svg_h}px")
    return '\n'.join(parts)


def main():
    print("=" * 60)
    print("  Ultra-Premium ASCII Art Generator v4")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_IMAGE
    output_path = script_dir / OUTPUT_SVG
    
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)
    
    print(f"\n1. Loading & enhancing image...")
    img, cols, rows = load_and_prepare(str(input_path), COLS, CHAR_ASPECT)
    print(f"   Grid: {cols}x{rows}")
    
    print(f"\n2. Computing luminance & edges...")
    arr = np.array(img, dtype=np.float64)
    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    edge_mag, edge_dir = compute_edges(lum)
    
    print(f"\n3. Building SVG...")
    svg = generate_svg(arr, lum, edge_mag, edge_dir, cols, rows, RAMP)
    
    print(f"\n4. Saving...")
    output_path.write_text(svg, encoding='utf-8')
    
    size_kb = output_path.stat().st_size / 1024
    print(f"   Output: {output_path.name} ({size_kb:.1f} KB)")
    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
