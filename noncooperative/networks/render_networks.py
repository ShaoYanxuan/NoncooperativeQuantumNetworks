"""
python3 -m main_paper.networks.render_networks
"""


from graph_tool.all import load_graph, graph_draw, sfdp_layout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# --- Parameters ---
NUM_NODES = 8+4*6
GRAPH_FOLDER = Path(f"main_paper/networks/{NUM_NODES}_nodes")
OUTPUT_PDF = f"sample_graphs_{NUM_NODES}_nodes.pdf"
RANGE = range(0, 100)

with PdfPages(OUTPUT_PDF) as pdf:
    for i in RANGE:
        graph_file = GRAPH_FOLDER / f"{i}.xml.gz"
        if not graph_file.exists():
            continue

        g = load_graph(str(graph_file))

        # Compute layout manually
        pos = sfdp_layout(g)

        # Create matplotlib figure
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        # Draw using the layout
        graph_draw(
            g,
            pos=pos,
            vertex_text=g.vertex_index,
            output_size=(300, 300),
            mplfig=ax,
        )
        ax.set_title(f"Graph {i}")
        pdf.savefig(fig)
        plt.close(fig)

print(f"Saved to {OUTPUT_PDF}")
