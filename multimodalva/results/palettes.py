"""
Color palettes for MultimodalVA result visualizations.

Designed to follow Lancet / Nature / NEJM house styles with colour-blind
friendly choices. All constants are module-level so they can be imported
directly and overridden at call sites.

Public names (exported from results/__init__.py):
    TOPK_BAR_COLORS   — categorical palette for bar charts (top-k accuracy, etc.)
    HEATMAP_SEQ       — monotone sequential (counts, probabilities, confusion matrix)
    HEATMAP_DIV       — diverging around zero (diff heatmaps, residuals)
    HEATMAP_CLINICAL  — high-contrast clinical sequential (Lancet-style accuracy heatmaps)
"""

# ---------------------------------------------------------------------------
# Categorical palette — bar / line charts
# ---------------------------------------------------------------------------
# Inspired by ggsci lancet_lanonc, nejm, and npg palettes.
# Ordered for maximum contrast between adjacent bars.
# Colour-blind friendly: avoids pure red/green confusion pairs by separating
# them with blue/teal/neutral entries.

TOPK_BAR_COLORS: list[str] = [
    "#00468B",  # deep navy          (Lancet primary anchor)
    "#ED0000",  # controlled red     (Lancet accent; not oversaturated)
    "#42B540",  # muted green        (Nature NPG secondary category)
    "#0099B4",  # teal               (NEJM clinical tone)
    "#7E6148",  # earthy brown       (softer than saddlebrown)
    "#925E9F",  # muted purple       (balanced mid-tone)
    "#FDAF91",  # soft coral/salmon  (highlight; not dominant)
    "#ADB6B6",  # neutral grey       (baseline / reference)
    "#1B1919",  # near-black         (strong contrast)
    "#3B4992",  # indigo-blue        (navy variation)
    "#EE7733",  # muted orange       (less aggressive than darkorange)
    "#66A61E",  # olive green
    "#11A579",  # green-teal hybrid
    "#DDCC77",  # desaturated yellow (print-friendly)
]

# ---------------------------------------------------------------------------
# Sequential heatmap — monotone luminance increase
# ---------------------------------------------------------------------------
# Use for: counts, probabilities, intensities, confusion matrices.
# Blues progression (light → dark); perceptually uniform.

HEATMAP_SEQ: list[str] = [
    "#F7FBFF",
    "#DEEBF7",
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#2171B5",
    "#08519C",
    "#08306B",
]

# ---------------------------------------------------------------------------
# Diverging heatmap — centred at zero / neutral midpoint
# ---------------------------------------------------------------------------
# Use for: performance differences vs baseline, log fold-change, residuals,
#          cause_accuracy_diff_heatmap().

HEATMAP_DIV: list[str] = [
    "#053061",  # deep navy    (negative extreme)
    "#2166AC",
    "#67A9CF",
    "#F7F7F7",  # near-white   (zero / midpoint)
    "#FDDBC7",
    "#EF8A62",
    "#B2182B",  # deep red     (positive extreme)
]

# ---------------------------------------------------------------------------
# Clinical sequential — high-contrast Lancet-style
# ---------------------------------------------------------------------------
# Use for: cause-specific accuracy heatmaps where clinical readability matters.
# Lighter base with strong saturated end; works well on white paper / slides.

HEATMAP_CLINICAL: list[str] = [
    "#F5F5F5",
    "#D0D6E2",
    "#A6BDD7",
    "#74A9CF",
    "#2B8CBE",
    "#045A8D",
]
