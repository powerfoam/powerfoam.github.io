import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties

np.random.seed(42)

def create_power_foam_points(text="Power Foam", num_points=10000, noise_std=0.5):
    # Create a hidden figure to render the text to an image
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Try using a bold sans-serif font
    fp = FontProperties(family='sans-serif', weight='bold')
    
    # Render the text black on white
    ax.text(0.5, 0.5, text, fontsize=120, ha='center', va='center', fontproperties=fp)
    ax.axis('off')
    
    # Save the text image to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1, transparent=False)
    buf.seek(0)
    plt.close(fig)
    
    # Read the image back in
    img = mpimg.imread(buf)
    
    # Check if image is RGBA and drop alpha channel for grayscale calculation if needed
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Use RGB for intensity calculation
        gray = np.mean(img[:, :, :3], axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img
        
    # Text is black, background is white. Find coordinates of dark pixels.
    y_idx, x_idx = np.where(gray < 0.5)
    
    if len(x_idx) == 0:
        return np.array([]), np.array([])
        
    min_x, max_x = x_idx.min(), x_idx.max()
    min_y, max_y = y_idx.min(), y_idx.max()
    
    from visualize_power_diagram import generate_blue_noise_points
    
    # Generate blue noise points over the bounding box of the text
    # Radius of 4.5 gives a good density for a 1200x300 image
    print("Generating blue noise points...")
    all_points = generate_blue_noise_points(min_x, max_x, min_y, max_y, r=7.5)
    
    # Filter points that fall inside the dark text regions
    x_coords = []
    y_coords = []
    
    for p in all_points:
        px, py = int(p[0]), int(p[1])
        if 0 <= py < gray.shape[0] and 0 <= px < gray.shape[1]:
            if gray[py, px] < 0.5:
                x_coords.append(p[0])
                # Y coordinates go from top to bottom in images, so we need to negative them for the scatter plot
                y_coords.append(-p[1])
                
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
        
    # Add optional small random noise to simulate a foam/particle look if requested
    if noise_std > 0:
        x_coords = x_coords + np.random.normal(0, noise_std, size=x_coords.shape)
        y_coords = y_coords + np.random.normal(0, noise_std, size=y_coords.shape)
    
    # Normalize coordinates somewhat for easier plotting, putting center at (0, 0)
    if len(x_coords) > 0:
        x_coords = x_coords - np.mean(x_coords)
        y_coords = y_coords - np.mean(y_coords)
    
    print(f"Sampled {len(x_coords)} blue noise points inside text.")
    return x_coords, y_coords


def main():
    print("Generating points for 'Power Foam'...")
    x, y = create_power_foam_points(text="Power Foam", num_points=100, noise_std=0.1)
    
    print("Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Map the x coordinates to a custom colormap for a nice gradient effect
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    
    # Original colors: (16, 0, 126) and (145, 21, 21)
    # Convert to HSV, use 30% saturation, 100% brightness (value), keep original hue
    h1, _, _ = colorsys.rgb_to_hsv(16/255.0, 0.0, 126/255.0)
    color_start = colorsys.hsv_to_rgb(h1, 0.5, 0.9)
    
    h2, _, _ = colorsys.rgb_to_hsv(145/255.0, 21/255.0, 21/255.0)
    color_end = colorsys.hsv_to_rgb(h2, 0.5, 0.9)
    print(color_start, color_end)
    
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", [color_start, color_end])
    
    norm_x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    colors = custom_cmap(norm_x)
    
    # Add variable radii for a foam-like effect
    # The coordinates are based on image pixels, bounding box is roughly 1000x300
    # Average distance between 500 points is around 15-25 units
    radii = np.random.uniform(6, 10, size=len(x))
    
    print("Computing power diagram...")
    from visualize_power_diagram import compute_power_diagram_edges
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Arc
    
    points = np.column_stack((x, y))
    processed_edges, alpha_edges, centroids, boundary_arcs, site_neighbors = compute_power_diagram_edges(points, radii)
    
    print("Drawing power cells...")
    # Plotting circles as opaque patches so they don't blend in overlap regions
    for pt, r, c in zip(points, radii, colors):
        circle = plt.Circle(pt, r, color=c, alpha=1.0, ec='none')
        ax.add_patch(circle)
        
    # Plot restricted power diagram edges
    restricted_lines = []
    for edge in processed_edges:
        if edge["type"] == "restricted":
            restricted_lines.append([edge["p1"], edge["p2"]])
            
    if restricted_lines:
        lc_res = LineCollection(
            restricted_lines,
            colors="white",
            linewidths=1.2,
            alpha=0.9
        )
        ax.add_collection(lc_res)

    # Plot union boundary arcs
    for arc in boundary_arcs:
        p = Arc(
            xy=arc["center"],
            width=2 * arc["radius"],
            height=2 * arc["radius"],
            angle=0,
            theta1=arc["start_angle"],
            theta2=arc["end_angle"],
            color="white",
            linewidth=1.2,
            alpha=0.9
        )
        ax.add_patch(p)
    
    # Make the plot clean by removing axes
    ax.set_xlim(x.min() - 30, x.max() + 30)
    ax.set_ylim(y.min() - 30, y.max() + 30)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    
    plt.title("Power Foam", fontsize=20, fontname='sans-serif', weight='bold', color='white', y=-0.1)
    
    # Add a dark background option for contrast (optional)
    # plt.gcf().patch.set_facecolor('#1a1a1a')
    
    output_filename = "power_foam_teaser.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1, transparent=True)
    print(f"Teaser figure saved successfully to {output_filename}")


if __name__ == "__main__":
    main()
