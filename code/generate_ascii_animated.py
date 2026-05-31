
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
OUTPUT_SVG  = Path(r"../files/dragon_knight_ani.svg")

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

# Animation group counts (only for fire zones)
NUM_FLAME_GROUPS = 16
NUM_SCATTER_GROUPS = 8


# image processing

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


# animation

def classify_fire_zone(row, rows, col, cols, r, g, b, lum_val):
    """
    Classify fire-colored characters only. Returns:
      'flame'   — fire in the main beam
      'scatter' — fire at the shield impact edge (rightmost part of the beam)
      None      — not fire-colored or outside the fire beam (render static)
    """
    if lum_val < DARK_THRESHOLD:
        return None

    norm_x = col / cols
    norm_y = row / rows
    
    # Do not animate the dragon (left) or its upper wings (top)
    if norm_x < 0.34 or norm_y < 0.38:
        return None
        
    # Do not animate the knight or the shield (on the right)
    if norm_x > 0.75:
        return None

    if not (is_fire_color(int(r), int(g), int(b)) and lum_val > 30):
        return None

    # The impact area right before the shield
    if norm_x > 0.70:
        return 'scatter'
    return 'flame'


def compute_anim_class(zone, row, col, cols):
    """Return the CSS class for a fire character's animation group."""
    if zone == 'flame':
        fire_start = 0.33
        fire_end = 0.75
        norm_x = col / cols
        fire_pos = (norm_x - fire_start) / (fire_end - fire_start)
        fire_pos = max(0.0, min(1.0, fire_pos))

        # Add row-based turbulence so the wave isn't a flat vertical line
        turb = ((row * 7 + col * 13) % 16) / 16.0 * 0.2
        delay_norm = fire_pos * 0.8 + turb
        bucket = int(delay_norm * NUM_FLAME_GROUPS) % NUM_FLAME_GROUPS
        return f'fl{bucket}'

    elif zone == 'scatter':
        bucket = ((col * 7 + row * 13) % NUM_SCATTER_GROUPS)
        return f'sp{bucket}'

    return None


def generate_css_animations():
    """Generate CSS @keyframes and animation classes for fire zones only."""
    lines = []

    lines.append("""
    /* ═══ Fire Animation Keyframes ═══ */

    /* Flame flicker: opacity + brightness pulsing */
    @keyframes flicker {
      0%   { opacity: 0.82; filter: brightness(1.0); }
      14%  { opacity: 1.0;  filter: brightness(1.2); }
      28%  { opacity: 0.72; filter: brightness(0.9); }
      42%  { opacity: 0.95; filter: brightness(1.12); }
      56%  { opacity: 0.78; filter: brightness(0.92); }
      70%  { opacity: 1.0;  filter: brightness(1.15); }
      84%  { opacity: 0.68; filter: brightness(0.88); }
      100% { opacity: 0.82; filter: brightness(1.0); }
    }

    /* Flame drift: subtle positional jitter */
    @keyframes drift {
      0%, 100% { transform: translate(0, 0); }
      25%  { transform: translate(0.8px, -0.5px); }
      50%  { transform: translate(-0.5px, 0.4px); }
      75%  { transform: translate(0.6px, -0.7px); }
    }

    /* Shield sparks: very subtle movement and brightness pulses so it stays clear */
    @keyframes spark {
      0%   { opacity: 1.0;  transform: translate(0, 0);          filter: brightness(1.0); }
      33%  { opacity: 0.85; transform: translate(0.5px, -0.5px); filter: brightness(1.4); }
      66%  { opacity: 0.95; transform: translate(-0.3px, 0.3px); filter: brightness(1.2); }
      100% { opacity: 1.0;  transform: translate(0, 0);          filter: brightness(1.0); }
    }
    """)

    # Flame classes: varying duration + staggered delay for wave effect
    for i in range(NUM_FLAME_GROUPS):
        delay = i * 0.06
        dur_f = 0.70 + (i % 3) * 0.10   # 0.70 / 0.80 / 0.90
        dur_d = 1.10 + (i % 3) * 0.15   # 1.10 / 1.25 / 1.40
        lines.append(
            f"    .fl{i} {{ animation: flicker {dur_f:.2f}s {delay:.2f}s ease-in-out infinite,"
            f" drift {dur_d:.2f}s {delay:.2f}s ease-in-out infinite; }}"
        )

    # Scatter classes: faster, more chaotic
    for i in range(NUM_SCATTER_GROUPS):
        delay = i * 0.05
        dur = 0.50 + (i % 3) * 0.10   # 0.50 / 0.60 / 0.70
        lines.append(
            f"    .sp{i} {{ animation: spark {dur:.2f}s {delay:.2f}s ease-in-out infinite; }}"
        )

    return "\n".join(lines)


# svg generation

def generate_svg(img_array, lum, edge_mag, edge_dir, cols, rows, ramp):
    """Generate SVG with fire-only CSS animations. Non-fire chars are static (v4 quality)."""

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
    parts.append(generate_css_animations())
    parts.append(f'  </style>')

    char_count = 0
    animated_count = 0
    static_count = 0

    for row in range(rows):
        y = int(row * LINE_HEIGHT + FONT_SIZE) + 4

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

            x = int(col * CHAR_WIDTH) + 8
            cr, cg, cb = color
            hex_color = f"#{cr:02x}{cg:02x}{cb:02x}"
            ch_esc = escape_xml(ch)

            # Only fire-colored chars get animation; everything else is static
            zone = classify_fire_zone(row, rows, col, cols, r, g, b, l)
            anim_class = compute_anim_class(zone, row, col, cols) if zone else None

            if anim_class:
                parts.append(f'  <text x="{x}" y="{y}" fill="{hex_color}" class="{anim_class}">{ch_esc}</text>')
                animated_count += 1
            else:
                parts.append(f'  <text x="{x}" y="{y}" fill="{hex_color}">{ch_esc}</text>')
                static_count += 1

            char_count += 1

    parts.append('</svg>')

    print(f"  -> {char_count} total chars ({animated_count} animated, {static_count} static)")
    print(f"  -> Grid: {rows}x{cols}, Canvas: {svg_w}x{svg_h}px")
    return '\n'.join(parts)


# main

def main():
    print("=" * 60)
    print("  Ultra-Premium ANIMATED ASCII Art Generator v5")
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
    
    # Hide the Gemini watermark in the bottom right corner by blending with adjacent rocks
    r_start, c_start = int(rows * 0.85), int(cols * 0.90)
    for r in range(r_start, rows):
        for c in range(c_start, cols):
            l_val = 0.299 * arr[r, c, 0] + 0.587 * arr[r, c, 1] + 0.114 * arr[r, c, 2]
            if l_val > 35:  # The white logo pixels
                # Mirror the rock texture from the left side of the watermark zone
                mirror_c = c_start - (c - c_start) - 5
                if mirror_c >= 0:
                    arr[r, c] = arr[r, mirror_c]

    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    edge_mag, edge_dir = compute_edges(lum)

    print(f"\n3. Building animated SVG...")
    svg = generate_svg(arr, lum, edge_mag, edge_dir, cols, rows, RAMP)

    print(f"\n4. Saving...")
    output_path.write_text(svg, encoding='utf-8')

    size_kb = output_path.stat().st_size / 1024
    print(f"   Output: {output_path.name} ({size_kb:.1f} KB)")
    print(f"\n{'=' * 60}")
    print("  Done! Open in a browser to see fire animations.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
