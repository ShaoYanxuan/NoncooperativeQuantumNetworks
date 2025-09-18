def savefig_smart(fig, path, dpi=600):
    from pathlib import Path
    import matplotlib.pyplot as plt

    # Ensure parent directory exists
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save and close figure
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved: {path}")
