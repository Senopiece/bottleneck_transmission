import numpy as np
import networkx as nx
import plotly.graph_objects as go


# ----------------------------
#  Matrix + graph construction
# ----------------------------


def random_gf2_matrix(n: int) -> np.ndarray:
    """Generate a random full-rank binary matrix over GF(2)."""
    while True:
        A = np.random.randint(0, 2, (n, n), dtype=int)
        # if np.linalg.matrix_rank(A % 2) == n:
        return A % 2


def build_state_graph(A: np.ndarray) -> nx.DiGraph:
    """Build directed graph of 2^n binary states under A*s mod 2."""
    n = A.shape[0]
    G = nx.DiGraph()
    for i in range(2**n):
        s = np.array(list(map(int, np.binary_repr(i, width=n))))
        t = (A @ s) % 2
        j = int("".join(map(str, t)), 2)
        G.add_edge(i, j)  # self-transitions included if A*s == s
    return G


# ----------------------------
#  Layout utilities
# ----------------------------


def repel_positions(pos, iterations=200, lr=0.02, repulsion=0.02):
    """Simple repulsion adjustment to spread overlapping nodes."""
    nodes = list(pos.keys())
    coords = np.array([pos[n] for n in nodes])
    for _ in range(iterations):
        delta = coords[:, None, :] - coords[None, :, :]
        dist2 = np.sum(delta**2, axis=-1) + 1e-6
        forces = (repulsion / dist2[..., None]) * delta
        coords += lr * np.sum(forces, axis=1)
    return {n: coords[i] for i, n in enumerate(nodes)}


def layout_components(G: nx.DiGraph) -> dict:
    """Compute a global layout where each component is repelled from others."""
    components = list(nx.weakly_connected_components(G))
    all_pos = {}

    # Base circle positions for components
    R = 3.0
    for ci, comp in enumerate(components):
        sub = G.subgraph(comp)
        # Local layout within component
        local_pos = nx.kamada_kawai_layout(sub)
        local_pos = repel_positions(local_pos, iterations=300, lr=0.02, repulsion=0.03)

        # Scale down component size
        local_scale = 0.4 + 0.1 * np.log(len(comp) + 1)
        local_pos = {n: local_scale * np.array(p) for n, p in local_pos.items()}

        # Place component on large circle
        angle = 2 * np.pi * ci / len(components)
        shift = np.array([R * np.cos(angle), R * np.sin(angle)])
        for n, p in local_pos.items():
            all_pos[n] = p + shift

    # Global repulsion for components themselves
    all_pos = repel_positions(all_pos, iterations=300, lr=0.02, repulsion=0.01)
    return all_pos


# ----------------------------
#  Visualization
# ----------------------------


def draw_state_graph_plotly(G: nx.DiGraph, A: np.ndarray):
    """Interactive Plotly visualization with arrows and component repulsion."""
    pos = layout_components(G)

    edge_x, edge_y, arrow_x, arrow_y = [], [], [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # line for edge
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        # arrowhead
        if u != v:
            dx, dy = x1 - x0, y1 - y0
            norm = np.hypot(dx, dy) + 1e-6
            ux, uy = dx / norm, dy / norm
            ax, ay = x1 - 0.04 * ux, y1 - 0.04 * uy
            perp = np.array([[-uy, ux], [uy, -ux]])
            for p in perp:
                arrow_x += [ax, ax - 0.015 * (ux + p[0]), None]
                arrow_y += [ay, ay - 0.015 * (uy + p[1]), None]
        else:
            # self-loop
            theta = np.linspace(0, 2 * np.pi, 20)
            r = 0.05
            edge_x += list(x0 + r * np.cos(theta)) + [None]
            edge_y += list(y0 + r * np.sin(theta)) + [None]

    node_x = [pos[k][0] for k in G.nodes()]
    node_y = [pos[k][1] for k in G.nodes()]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.6)", width=1),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arrow_x,
            y=arrow_y,
            mode="lines",
            line=dict(color="rgba(80,80,80,0.8)", width=1),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=5, color="deepskyblue"),
            hoverinfo="none",
        )
    )

    fig.update_layout(
        title=f"State Transition Graph (components repelled) — n={A.shape[0]}",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        width=800,
        height=800,
    )
    fig.show()


# ----------------------------
#  Main
# ----------------------------


def main():
    n = 5
    A = random_gf2_matrix(n)
    print("Random A =\n", A)
    G = build_state_graph(A)
    draw_state_graph_plotly(G, A)


if __name__ == "__main__":
    main()
