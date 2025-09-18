"""
✅
Visualize an allocator's quantum network with edge fidelities and vertex indices.
"""

from graph_tool.all import graph_draw
from qroute.routing.allocator import _make_edge_key
from qroute.routing.entanglement_manipulation import fidelity_from_werner_param


def plot_allocator_network(allocator, output_filename="network_plot.pdf"):
    g = allocator.graph
    edge_text = g.new_edge_property("string")
    for e in g.edges():
        key = _make_edge_key(e)
        edge_text[e] = "{:.3f}".format(
            fidelity_from_werner_param(allocator.edge_p0[key])
        )

    graph_draw(
        g,
        vprops={"text": g.vertex_index},
        eprops={"text": edge_text, "font_size": 30},
        output_size=(1200, 1200),
        output=output_filename,
    )

    if output_filename:
        print(f"[+] Network plot saved to {output_filename}")
