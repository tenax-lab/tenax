# Tensor Network GUI Design Notes

A GUI to visually plot tensor networks and generate Tenax code, inspired by [TensorTrace](https://www.tensortrace.com/) and [GuiTeNet](https://github.com/GuiTeNet/guitenet).

---

## Existing Tools Landscape

| Tool | Tech | Strengths | Limitations |
|------|------|-----------|-------------|
| [TensorTrace](https://www.tensortrace.com/) | Desktop app (Unity) | Custom drawing GUI, optimal contraction order, code gen (MATLAB/Python/Julia via `ncon`) | Desktop-only, no web, outputs `ncon` not library-native code |
| [GuiTeNet](https://github.com/GuiTeNet/guitenet) | Web (JS) | Browser-based, visual graph editing | Research prototype, limited backends |
| [Cytnx Network](https://github.com/Cytnx-dev/Cytnx) | Python CLI | `print_diagram()` ASCII art, `.net` file support | Not visual/interactive, no drag-and-drop |

Tenax already has `NetworkBlueprint` / `.net` files and a `TensorNetwork` graph container -- that's a strong foundation for code generation. The missing piece is the **visual editor**.

---

## Decisions

- **Platform**: Lightweight **web app** (no desktop, no Jupyter widget for MVP)
- **Repo**: Separate repository (e.g., `tn-editor`), not inside Tenax
  - Different release cycles — GUI ships independently from the library
  - Different CI — npm/Vite/Playwright vs pytest/ruff
  - Different contributors — frontend devs don't need the full Tenax tree
  - Tenax is a pip dependency of the backend, not a monorepo sibling

### Frontend: Svelte + Svelte Flow

**Svelte** over React for lightweight:
- ~5x smaller bundle (no virtual DOM runtime)
- Less boilerplate, reactive by default
- Svelte 5 runes API is clean and performant

**Svelte Flow** for the graph canvas:
- Tensor = custom node with dynamic handles (one per leg)
- Connecting legs = drag edge between handles
- Free legs = dangling handles (output indices)
- Built-in minimap, controls, pan/zoom

### Backend: Two Options

Both options expose the same four operations wrapping Tenax:
- **optimize** -- find optimal contraction order + FLOP cost via `opt_einsum`
- **codegen** -- convert network JSON to Tenax Python code
- **export** -- generate `.net` file or LaTeX/TikZ
- **validate** -- check charge conservation for symmetric networks

#### Option A: FastAPI (REST)

Thin **FastAPI** server with REST endpoints (`/api/optimize`, `/api/codegen`, etc.):
- Async, minimal overhead
- Direct access to Tenax (no serialization/pyodide hacks)
- Auto-generated OpenAPI docs for free
- Single `pip install` with `uvicorn`
- Straightforward for GUI-only use case

```
Browser (Svelte)                    Server (FastAPI)
+----------------------------+      +------------------------+
| +--------+ +-------------+ |      |                        |
| | Tensor | | Svelte Flow | | JSON | /api/optimize          |
| | Palette| | Canvas      |------>| /api/codegen           |
| +--------+ +------+------+ |      | /api/export            |
|            | Props Panel  | |<-----| /api/validate          |
|            +------+------+ |      |                        |
|                   |         |      |  Tenax (opt_einsum,  |
|  +--------+ +----------+   |      |   NetworkBlueprint)    |
|  | Code   | | Cost     |   |      +------------------------+
|  | Mirror | | Display  |   |
|  +--------+ +----------+   |
+----------------------------+
```

#### Option B: MCP Server

Expose the same operations as **MCP tools** (`optimize_contraction`, `generate_code`,
`export_netfile`, `validate_network`) via Streamable HTTP transport:
- **Same server, two clients** -- both the GUI and AI assistants (Claude Code,
  or a chat panel embedded in the GUI) connect to the same MCP server
- **AI-assisted editing** -- user can ask "add a Heisenberg MPO for L=10" and
  the AI calls MCP tools to modify the network on the canvas
- **Single protocol** -- no separate REST API to maintain
- Trade-off: MCP is heavier than plain REST for simple tool calls; the win is
  native AI integration

```
Browser (Svelte)              MCP Server (Python)
+----------------------+      +---------------------+
| Svelte Flow Canvas   |      | Tools:              |
| + AI Chat Panel      | HTTP | - optimize          |
|                      |----->| - codegen           |
|                      |      | - export            |
|                      |<-----| - validate          |
+----------------------+      |                     |
                              | Tenax (opt_einsum,  |
Claude Code / any      |      | NetworkBlueprint)   |
MCP client             |----->|                     |
                              +---------------------+
```

#### Recommendation

Start with **Option A** (FastAPI) for MVP -- simpler to build and debug. The core
logic (codegen, optimization, export) lives in shared modules either way. Option B
can be added later by wrapping the same modules as MCP tools, and both can coexist
(FastAPI serving the GUI, MCP serving AI clients).

### Data Flow

1. User draws network on canvas (Svelte Flow nodes + edges)
2. Frontend holds network state as JSON (see schema below)
3. On change, call `optimize` for live cost estimate
4. "Generate Code" button calls `codegen`, result shown in CodeMirror
5. Code panel is read-only display (no bidirectional sync in MVP)

### 4. Internal JSON Schema (the "source of truth")

A network drawn in the GUI maps to a JSON like:

```json
{
  "tensors": {
    "A": {
      "rank": 3,
      "legs": [
        {"label": "v0_1", "dim": 32, "flow": "out", "type": "virtual"},
        {"label": "p0",   "dim": 2,  "flow": "out", "type": "physical"},
        {"label": "v1_2", "dim": 32, "flow": "in",  "type": "virtual"}
      ],
      "position": [100, 200],
      "shape": "circle"
    }
  },
  "connections": [
    {"from": ["A", "v0_1"], "to": ["B", "v0_1"]}
  ],
  "output_labels": ["p0", "p1"],
  "metadata": {
    "symmetry": "U1",
    "algorithm_hint": "DMRG"
  }
}
```

This maps 1:1 to the existing `NetworkBlueprint` / `.net` format and to `TensorNetwork.connect()` calls.

### 5. Code Generation Targets

The GUI should generate **idiomatic Tenax code**, not generic einsum. Three levels:

**Level 1 -- Raw contraction:**
```python
from tenax import DenseTensor, TensorIndex, contract, FlowDirection

idx_bond = TensorIndex(symmetry=U1Symmetry(), charges=..., flow=FlowDirection.OUT, label="bond")
A = DenseTensor(data=jnp.zeros((2, 32)), indices=(idx_phys, idx_bond))
B = DenseTensor(data=jnp.zeros((32, 2)), indices=(idx_bond2, idx_phys2))
result = contract(A, B)
```

**Level 2 -- Blueprint / .net file:**
```python
from tenax import NetworkBlueprint
bp = NetworkBlueprint("""
A: i, j, k
B: k, l, m
TOUT: i, j, l, m
""")
```

**Level 3 -- Algorithm template:**
```python
from tenax import AutoMPO, build_random_mps, dmrg, DMRGConfig

auto = AutoMPO(L=10, d=2)
auto += (1.0, "Sz", 0, "Sz", 1)
# ... user-drawn Hamiltonian terms
mpo = auto.to_mpo()
mps = build_random_mps(L=10, d=2, chi=32)
result = dmrg(mpo, mps, DMRGConfig(max_bond_dim=64))
```

### 6. Differentiating Features

1. **Bidirectional sync** -- edit code or diagram, both stay in sync (like Svelte playground)
2. **Contraction cost estimator** -- show FLOP count and optimal order live as you draw
3. **Algorithm templates** -- drag in "DMRG setup" and get MPS + MPO + environments pre-wired
4. **Symmetry-aware** -- legs display charge sectors, flow arrows (IN/OUT), conservation validation in real-time
5. **Export to LaTeX/TikZ** -- for papers (TensorTrace does this well, worth matching)
6. **Jupyter embedding** -- `widget = TNEditor(); display(widget)` then `widget.to_network()` returns a `TensorNetwork`

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| UI framework | **Svelte 5** | Smallest bundle, reactive by default |
| Graph editor | **Svelte Flow** | Node graph with handles, built for this |
| Code display | **CodeMirror 6** | Python syntax highlighting |
| Backend | **FastAPI + uvicorn** | Async, direct Tenax access, auto docs |
| Contraction opt | **opt_einsum** (via Tenax) | Server-side path finding |
| Build/bundle | **Vite** | Fast HMR, small production builds |
| Diagram export | **SVG -> LaTeX** | Native SVG from Svelte Flow |

---

## Project Structure

```
tn-editor/          # standalone repo
  frontend/
    src/
      lib/
        components/
          TensorNode.svelte      # custom Svelte Flow node
          LegHandle.svelte       # handle per tensor leg
          PropertiesPanel.svelte  # edit selected tensor
          TensorPalette.svelte   # drag to add tensors
          CodePanel.svelte       # CodeMirror output
          CostDisplay.svelte     # FLOP count badge
        stores/
          network.ts             # Svelte store: nodes, edges, JSON
          api.ts                 # fetch helpers for backend
        types.ts                 # TypeScript types for network JSON
      App.svelte
      main.ts
    package.json
    vite.config.ts
    svelte.config.js
  backend/
    app.py                       # FastAPI app
    codegen.py                   # JSON -> Tenax Python code
    optimizer.py                 # opt_einsum wrapper
    exporter.py                  # .net file + LaTeX export
    schemas.py                   # Pydantic models (match JSON schema)
  pyproject.toml                 # backend deps (fastapi, uvicorn, tenax as pip dep)
  README.md
```

---

## MVP Scope

**In scope (v0.1):**
- Draw tensors (custom node, configurable rank/legs)
- Connect legs by dragging edges
- Free legs shown as dangling stubs
- Properties panel: edit tensor name, leg labels, dimensions
- Live contraction cost estimate (FLOP count)
- Code generation: `.net` file format + `contract()` API
- Copy/download generated code

**Deferred (v0.2+):**
- Algorithm templates (DMRG, iPEPS pre-wired setups)
- Symmetry-aware mode (charge sectors, flow arrows)
- LaTeX/TikZ export
- Bidirectional code <-> diagram sync
- Jupyter widget embedding via anywidget

---

## Future Considerations

- **Jupyter embedding**: The Svelte component can be wrapped with `anywidget` later
- **Static deploy**: Code gen could move client-side via pyodide if serverless hosting is needed
- **Algorithm templates**: Pre-built Svelte Flow node groups (MPS chain, MPO row, PEPS grid)

---

## References

- [TensorTrace paper (arXiv 1911.02558)](https://arxiv.org/abs/1911.02558)
- [TensorTrace website](https://www.tensortrace.com/)
- [GuiTeNet (GitHub)](https://github.com/GuiTeNet/guitenet)
- [GuiTeNet paper](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.304)
- [Cytnx library](https://github.com/Cytnx-dev/Cytnx)
