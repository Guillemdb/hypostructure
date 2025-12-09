# Regularity Diagnostic Framework - LaTeX Version

```latex
\documentclass[11pt]{article}
\usepackage[paperwidth=50cm,paperheight=180cm,margin=1cm]{geometry}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning, calc, fit}
\usepackage{xcolor}

% Define colors matching the original diagram
\definecolor{axiomblue}{HTML}{3b82f6}
\definecolor{barrierorange}{HTML}{f59e0b}
\definecolor{failurered}{HTML}{ef4444}
\definecolor{successpurple}{HTML}{8b5cf6}
\definecolor{successgreen}{HTML}{22c55e}

\begin{document}
\pagestyle{empty}

\begin{tikzpicture}[
    % Node styles
    startstop/.style={
        rectangle, rounded corners=8pt,
        minimum width=4cm, minimum height=1.2cm,
        text centered, draw=successpurple!80!black, line width=2pt,
        fill=successpurple, text=white, font=\bfseries\small,
        align=center
    },
    axiom/.style={
        diamond, aspect=2.5,
        minimum width=5cm, minimum height=2cm,
        text centered, draw=axiomblue!80!black, line width=1.5pt,
        fill=axiomblue, text=white, font=\small,
        align=center, inner sep=2pt
    },
    barrier/.style={
        diamond, aspect=2.5,
        minimum width=4.5cm, minimum height=1.8cm,
        text centered, draw=barrierorange!80!black, line width=1.5pt,
        fill=barrierorange, text=black, font=\small,
        align=center, inner sep=2pt
    },
    failure/.style={
        rectangle, rounded corners=3pt,
        minimum width=3.5cm, minimum height=1cm,
        text centered, draw=failurered!80!black, line width=1.5pt,
        fill=failurered, text=white, font=\small\bfseries,
        align=center
    },
    success/.style={
        rectangle, rounded corners=8pt,
        minimum width=4cm, minimum height=1.2cm,
        text centered, draw=successgreen!80!black, line width=3pt,
        fill=successgreen, text=black, font=\bfseries\small,
        align=center
    },
    intermediate/.style={
        rectangle, rounded corners=5pt,
        minimum width=4cm, minimum height=1cm,
        text centered, draw=successpurple!80!black, line width=1.5pt,
        fill=successpurple, text=white, font=\small\bfseries,
        align=center
    },
    finalgate/.style={
        diamond, aspect=2.5,
        minimum width=5cm, minimum height=2.5cm,
        text centered, draw=successpurple!60!black, line width=3pt,
        fill=successpurple, text=white, font=\small,
        align=center, inner sep=2pt
    },
    arrow/.style={
        -stealth,
        thick, draw=black!70
    },
    labelstyle/.style={
        font=\scriptsize, fill=white, inner sep=2pt
    }
]

% ============ LEVEL 0: START ============
\node[startstop] (start) {\textbf{START DIAGNOSTIC}};

% ============ LEVEL 1: CONSERVATION ============
\node[axiom, below=1.5cm of start] (energy) {
    \textbf{1. AXIOM D (Dissipation)}\\
    Is Global Energy Finite?
};

\node[barrier, below left=2cm and 3cm of energy] (barriersat) {
    \textbf{SATURATION BARRIER}\\
    Is Drift Controlled?\\
    \textit{[Thm 6.2]}
};

\node[failure, below=1.5cm of barriersat] (modece) {
    \textbf{Mode C.E}\\
    Energy Blow-Up
};

\node[axiom, below=3cm of energy] (zeno) {
    \textbf{2. AXIOM REC (Recovery)}\\
    Are Discrete Events Finite?
};

\node[barrier, below left=2cm and 3cm of zeno] (barriercausal) {
    \textbf{CAUSAL CENSOR}\\
    Is Depth Finite?\\
    \textit{[Thm 85]}
};

\node[failure, below=1.5cm of barriercausal] (modecc) {
    \textbf{Mode C.C}\\
    Event Accumulation
};

% ============ LEVEL 2: DUALITY ============
\node[axiom, below=3cm of zeno] (compact) {
    \textbf{3. AXIOM C (Compactness)}\\
    Does Energy Concentrate?
};

\node[barrier, below left=2cm and 3cm of compact] (barrierscat) {
    \textbf{SCATTERING BARRIER}\\
    Is Interaction Finite?\\
    \textit{[Thm 47]}
};

\node[success, below left=1.5cm and -1cm of barrierscat] (modedd) {
    \textbf{Mode D.D}: Dispersion\\
    \textit{(Global Existence)}
};

\node[failure, below right=1.5cm and -1cm of barrierscat] (modecd_alt) {
    \textbf{Mode C.D}\\
    Geometric Collapse\\
    \textit{(Via Escape)}
};

\node[intermediate, below right=2cm and 3cm of compact] (profile) {
    \textbf{Canonical Profile $V$ Emerges}
};

% ============ LEVEL 3: SYMMETRY ============
\node[axiom, below=2cm of profile] (scale) {
    \textbf{4. AXIOM SC (Scaling)}\\
    Is it Subcritical?
};

\node[barrier, below left=2cm and 2cm of scale] (barriertypeii) {
    \textbf{TYPE II BARRIER}\\
    Is Renorm Cost Infinite?\\
    \textit{[Thm 56]}
};

\node[failure, below=1.5cm of barriertypeii] (modese) {
    \textbf{Mode S.E}\\
    Supercritical Cascade
};

\node[axiom, below=3cm of scale] (param) {
    \textbf{5. AXIOM SC (Stability)}\\
    Are Constants Stable?
};

\node[barrier, below left=2cm and 2cm of param] (barriervac) {
    \textbf{VACUUM BARRIER}\\
    Is Phase Stable?\\
    \textit{[Thm 9.150]}
};

\node[failure, below=1.5cm of barriervac] (modesc) {
    \textbf{Mode S.C}\\
    Parameter Instability
};

% ============ LEVEL 4: GEOMETRY ============
\node[axiom, below=3cm of param] (geom) {
    \textbf{6. AXIOM CAP (Capacity)}\\
    Is Dimension $>$ Critical?
};

\node[barrier, below left=2cm and 2cm of geom] (barriercap) {
    \textbf{CAPACITY BARRIER}\\
    Is Measure Zero?\\
    \textit{[Thm 58]}
};

\node[failure, below=1.5cm of barriercap] (modecd) {
    \textbf{Mode C.D}\\
    Geometric Collapse
};

% ============ LEVEL 5: STIFFNESS ============
\node[axiom, below=3cm of geom] (stiffness) {
    \textbf{7. AXIOM LS (Stiffness)}\\
    Is Hessian Positive?
};

\node[barrier, below left=2cm and 2cm of stiffness] (barriergap) {
    \textbf{SPECTRAL BARRIER}\\
    Is there a Gap?\\
    \textit{[Thm 83]}
};

% Bifurcation branch
\node[axiom, below right=3cm and 6cm of barriergap] (bifurcate) {
    \textbf{BIFURCATION CHECK}\\
    \textbf{(Axiom LS)}\\
    Is State Unstable?\\
    \textit{[Thm 125]}
};

\node[failure, below left=2cm and 1cm of bifurcate] (modesd) {
    \textbf{Mode S.D}\\
    Stiffness Breakdown
};

\node[axiom, below right=2cm and 1cm of bifurcate] (symcheck) {
    \textbf{SYMMETRY CHECK}\\
    Is Vacuum Degenerate?\\
    \textit{(Does Group $G$ exist?)}
};

% Path A: Symmetry Breaking
\node[axiom, below left=2cm and 2cm of symcheck] (checksc) {
    \textbf{AXIOM SC (Param)}\\
    Are Constants Stable?\\
    \textit{[Thm 51]}
};

\node[intermediate, below=1.5cm of checksc] (actionssb) {
    \textbf{ACTION: SYM. BREAKING}\\
    Generates Mass Gap
};

\node[failure, below right=1.5cm and 3cm of checksc] (modesc_rest) {
    \textbf{Mode S.C}\\
    Parameter Instability\\
    \textit{(Vacuum Decay)}
};

% Path B: Surgery
\node[axiom, below right=2cm and 2cm of symcheck] (checktb) {
    \textbf{AXIOM TB (Action)}\\
    Is Cost Finite?\\
    \textit{[Thm 32.2]}
};

\node[intermediate, below=1.5cm of checktb] (actionsurgery) {
    \textbf{ACTION: SURGERY}\\
    Dissipates Singularity
};

\node[failure, below right=1.5cm and 2cm of checktb] (modete_rest) {
    \textbf{Mode T.E}\\
    Topological Twist\\
    \textit{(Metastasis)}
};

% ============ LEVEL 6: TOPOLOGY ============
\node[axiom, below=3cm of stiffness] (topo) {
    \textbf{8. AXIOM TB (Topology)}\\
    Is Sector Accessible?
};

\node[barrier, below left=2cm and 2cm of topo] (barrieraction) {
    \textbf{ACTION BARRIER}\\
    Is Energy $<$ Gap?\\
    \textit{[Thm 59]}
};

\node[failure, below=1.5cm of barrieraction] (modete) {
    \textbf{Mode T.E}\\
    Topological Twist
};

\node[axiom, below=3cm of topo] (tame) {
    \textbf{9. AXIOM TB (Tameness)}\\
    Is Topology Simple?
};

\node[barrier, below left=2cm and 2cm of tame] (barrierOmin) {
    \textbf{O-MINIMAL BARRIER}\\
    Is it Definable?\\
    \textit{[Thm 46]}
};

\node[failure, below=1.5cm of barrierOmin] (modetc) {
    \textbf{Mode T.C}\\
    Labyrinthine
};

\node[axiom, below=3cm of tame] (ergo) {
    \textbf{10. AXIOM TB (Mixing)}\\
    Does it Mix?
};

\node[barrier, below left=2cm and 2cm of ergo] (barriermix) {
    \textbf{MIXING BARRIER}\\
    Is Trap Escapable?\\
    \textit{[Thm 132]}
};

\node[failure, below=1.5cm of barriermix] (modetd) {
    \textbf{Mode T.D}\\
    Glassy Freeze
};

% ============ LEVEL 7: COMPLEXITY ============
\node[axiom, below=3cm of ergo] (complex) {
    \textbf{11. AXIOM REP (Dictionary)}\\
    Is it Computable?
};

\node[barrier, below left=2cm and 2cm of complex] (barrierepi) {
    \textbf{EPISTEMIC BARRIER}\\
    Is Description Finite?\\
    \textit{[Thm 101]}
};

\node[failure, below=1.5cm of barrierepi] (modedc) {
    \textbf{Mode D.C}\\
    Semantic Horizon
};

\node[axiom, below=3cm of complex] (oscillate) {
    \textbf{12. AXIOM GC (Gradient)}\\
    Does it Oscillate?
};

\node[barrier, below left=2cm and 2cm of oscillate] (barrierfreq) {
    \textbf{FREQUENCY BARRIER}\\
    Is Integral Finite?\\
    \textit{[Thm 48]}
};

\node[failure, below=1.5cm of barrierfreq] (modede) {
    \textbf{Mode D.E}\\
    Oscillatory
};

% ============ LEVEL 8: BOUNDARY ============
\node[axiom, below=3cm of oscillate] (boundary) {
    \textbf{13. BOUNDARY CHECK}\\
    Is System Open?
};

\node[axiom, below left=2cm and 4cm of boundary] (overload) {
    \textbf{14. BOUNDARY: CONTROL}\\
    Is Input Bounded?
};

\node[barrier, below=1.5cm of overload] (barrierbode) {
    \textbf{BODE BARRIER}\\
    Is Sensitivity Bounded?\\
    \textit{[Thm 13]}
};

\node[failure, below=1.5cm of barrierbode] (modebe) {
    \textbf{Mode B.E}\\
    Injection
};

\node[axiom, below=4cm of overload] (starve) {
    \textbf{15. BOUNDARY: SUPPLY}\\
    Is Input Sufficient?
};

\node[barrier, below=1.5cm of starve] (barrierinput) {
    \textbf{INPUT BARRIER}\\
    Is Reserve Sufficient?\\
    \textit{[Prop 18]}
};

\node[failure, below=1.5cm of barrierinput] (modebd) {
    \textbf{Mode B.D}\\
    Starvation
};

\node[axiom, below=4cm of starve] (align) {
    \textbf{16. BOUNDARY: GAUGE}\\
    Is it Aligned?
};

\node[barrier, below=1.5cm of align] (barriervariety) {
    \textbf{VARIETY BARRIER}\\
    Does Control Match\\Disturbance?\\
    \textit{[Thm 89]}
};

\node[failure, below=1.5cm of barriervariety] (modebc) {
    \textbf{Mode B.C}\\
    Misalignment
};

% ============ LEVEL 9: THE FINAL GATE ============
\node[finalgate, below=5cm of boundary] (exclusion) {
    \textbf{17. THE CATEGORICAL LOCK}\\
    Is $\mathrm{Hom}(\mathrm{Bad}, S) = \emptyset$?\\
    \textit{[Metatheorem 76]}
};

\node[success, below left=2cm and 2cm of exclusion] (victory) {
    \textbf{GLOBAL REGULARITY}\\
    \textit{(Structural Exclusion Confirmed)}
};

\node[failure, below right=2cm and 2cm of exclusion] (modecat) {
    \textbf{FATAL ERROR}\\
    Structural Inconsistency
};

% ============ ARROWS ============
% Main flow
\draw[arrow] (start) -- (energy);
\draw[arrow] (energy) -- node[labelstyle, right] {Yes} (zeno);
\draw[arrow] (energy) -- node[labelstyle, above left] {No} (barriersat);
\draw[arrow] (barriersat) -- node[labelstyle, left] {No (Breached)} (modece);
\draw[arrow] (barriersat) -- node[labelstyle, below right] {Yes (Blocked)} (zeno);

\draw[arrow] (zeno) -- node[labelstyle, right] {Yes} (compact);
\draw[arrow] (zeno) -- node[labelstyle, above left] {No} (barriercausal);
\draw[arrow] (barriercausal) -- node[labelstyle, left] {No (Breached)} (modecc);
\draw[arrow] (barriercausal) -- node[labelstyle, below right] {Yes (Blocked)} (compact);

\draw[arrow] (compact) -- node[labelstyle, above left] {No (Scatters)} (barrierscat);
\draw[arrow] (compact) -- node[labelstyle, above right] {Yes} (profile);
\draw[arrow] (barrierscat) -- node[labelstyle, above left] {Yes (Benign)} (modedd);
\draw[arrow] (barrierscat) -- node[labelstyle, above right] {No (Pathological)} (modecd_alt);

\draw[arrow] (profile) -- (scale);
\draw[arrow] (scale) -- node[labelstyle, above left] {No (Supercritical)} (barriertypeii);
\draw[arrow] (scale) -- node[labelstyle, right] {Yes (Safe)} (param);
\draw[arrow] (barriertypeii) -- node[labelstyle, left] {No (Breached)} (modese);
\draw[arrow] (barriertypeii) -- node[labelstyle, below right] {Yes (Blocked)} (param);

\draw[arrow] (param) -- node[labelstyle, above left] {No} (barriervac);
\draw[arrow] (param) -- node[labelstyle, right] {Yes} (geom);
\draw[arrow] (barriervac) -- node[labelstyle, left] {No (Breached)} (modesc);
\draw[arrow] (barriervac) -- node[labelstyle, below right] {Yes (Blocked)} (geom);

\draw[arrow] (geom) -- node[labelstyle, above left] {No (Too Thin)} (barriercap);
\draw[arrow] (geom) -- node[labelstyle, right] {Yes (Safe)} (stiffness);
\draw[arrow] (barriercap) -- node[labelstyle, left] {No (Breached)} (modecd);
\draw[arrow] (barriercap) -- node[labelstyle, below right] {Yes (Blocked)} (stiffness);

\draw[arrow] (stiffness) -- node[labelstyle, above left] {No (Flat)} (barriergap);
\draw[arrow] (stiffness) -- node[labelstyle, right] {Yes (Safe)} (topo);
\draw[arrow] (barriergap) -- node[labelstyle, above right] {No (Stagnation)} (bifurcate);
\draw[arrow] (barriergap) -- node[labelstyle, below left] {Yes (Blocked)} (topo);

% Bifurcation flow
\draw[arrow] (bifurcate) -- node[labelstyle, above left] {No (Stable)} (modesd);
\draw[arrow] (bifurcate) -- node[labelstyle, above right] {Yes (Unstable)} (symcheck);
\draw[arrow] (symcheck) -- node[labelstyle, above left] {Yes (Symmetric)} (checksc);
\draw[arrow] (symcheck) -- node[labelstyle, above right] {No (Asymmetric)} (checktb);
\draw[arrow] (checksc) -- node[labelstyle, left] {Yes} (actionssb);
\draw[arrow] (checksc) -- node[labelstyle, above right] {No} (modesc_rest);
\draw[arrow] (checktb) -- node[labelstyle, left] {Yes} (actionsurgery);
\draw[arrow] (checktb) -- node[labelstyle, above right] {No} (modete_rest);
% Snap-back paths: Actions are continuation mechanisms, not termination
% Symmetry Breaking must re-verify stiffness of new vacuum state
\draw[arrow, bend left=60] (actionssb) to node[labelstyle, right] {Re-verify Stability} (stiffness);
% Surgery must re-verify topology is still tame after modification
\draw[arrow, bend right=60] (actionsurgery) to node[labelstyle, right] {Re-verify Topology} (tame);

% Topology flow
\draw[arrow] (topo) -- node[labelstyle, above left] {No (Protected)} (barrieraction);
\draw[arrow] (topo) -- node[labelstyle, right] {Yes (Safe)} (tame);
\draw[arrow] (barrieraction) -- node[labelstyle, left] {No (Breached)} (modete);
\draw[arrow] (barrieraction) -- node[labelstyle, below right] {Yes (Blocked)} (tame);

\draw[arrow] (tame) -- node[labelstyle, above left] {No} (barrierOmin);
\draw[arrow] (tame) -- node[labelstyle, right] {Yes} (ergo);
\draw[arrow] (barrierOmin) -- node[labelstyle, left] {No (Breached)} (modetc);
\draw[arrow] (barrierOmin) -- node[labelstyle, below right] {Yes (Blocked)} (ergo);

\draw[arrow] (ergo) -- node[labelstyle, above left] {No} (barriermix);
\draw[arrow] (ergo) -- node[labelstyle, right] {Yes} (complex);
\draw[arrow] (barriermix) -- node[labelstyle, left] {No (Breached)} (modetd);
\draw[arrow] (barriermix) -- node[labelstyle, below right] {Yes (Blocked)} (complex);

% Complexity flow
\draw[arrow] (complex) -- node[labelstyle, above left] {No} (barrierepi);
\draw[arrow] (complex) -- node[labelstyle, right] {Yes} (oscillate);
\draw[arrow] (barrierepi) -- node[labelstyle, left] {No (Breached)} (modedc);
\draw[arrow] (barrierepi) -- node[labelstyle, below right] {Yes (Blocked)} (oscillate);

\draw[arrow] (oscillate) -- node[labelstyle, above left] {Yes} (barrierfreq);
\draw[arrow] (oscillate) -- node[labelstyle, right] {No} (boundary);
\draw[arrow] (barrierfreq) -- node[labelstyle, left] {No (Breached)} (modede);
\draw[arrow] (barrierfreq) -- node[labelstyle, below right] {Yes (Blocked)} (boundary);

% Boundary flow
\draw[arrow] (boundary) -- node[labelstyle, above left] {Yes} (overload);
\draw[arrow] (boundary) -- node[labelstyle, right] {No} (exclusion);
\draw[arrow] (overload) -- node[labelstyle, left] {Yes} (barrierbode);
\draw[arrow] (overload) -- node[labelstyle, right] {No} (starve);
\draw[arrow] (barrierbode) -- node[labelstyle, left] {No (Breached)} (modebe);
\draw[arrow] (barrierbode) -- node[labelstyle, right] {Yes (Blocked)} (exclusion);

\draw[arrow] (starve) -- node[labelstyle, left] {Yes} (barrierinput);
\draw[arrow] (starve) -- node[labelstyle, right] {No} (align);
\draw[arrow] (barrierinput) -- node[labelstyle, left] {No (Breached)} (modebd);
\draw[arrow] (barrierinput) -- node[labelstyle, right] {Yes (Blocked)} (exclusion);

\draw[arrow] (align) -- node[labelstyle, left] {No} (barriervariety);
\draw[arrow] (align) -- node[labelstyle, right] {Yes} (exclusion);
\draw[arrow] (barriervariety) -- node[labelstyle, left] {No (Breached)} (modebc);
\draw[arrow] (barriervariety) -- node[labelstyle, right] {Yes (Blocked)} (exclusion);

% Final gate
\draw[arrow] (exclusion) -- node[labelstyle, above left] {Yes (Blocked)} (victory);
\draw[arrow] (exclusion) -- node[labelstyle, above right] {No (Morphism Exists)} (modecat);

\end{tikzpicture}

\end{document}
```

## Compilation Instructions

To compile this LaTeX document to PDF:

```bash
# Using pdflatex
pdflatex diagram_latex.tex

# Or using lualatex (recommended for large diagrams)
lualatex diagram_latex.tex
```

## Notes

- The diagram uses a custom paper size (50cm x 180cm) to accommodate the full flowchart
- Colors match the original Mermaid diagram
- Node styles:
  - **Blue diamonds**: Axiom checks
  - **Orange diamonds**: Barriers
  - **Red rectangles**: Failure modes
  - **Green rounded rectangles**: Success states
  - **Purple rectangles**: Intermediate nodes and actions
  - **Purple diamond (thick border)**: Final categorical lock
