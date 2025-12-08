
```mermaid
graph TD
    Start(["<b>REGULARITY ANALYSIS</b>"]) --> EnergyCheck{"<b>1. A PRIORI ESTIMATES</b><br>Is Energy Bounded?"}

    %% --- LEVEL 1: CONSERVATION ---
    EnergyCheck -- "No" --> BarrierSat{"<b>CONSERVATION LAWS</b><br>Is Drift Controlled?"}
    BarrierSat -- "Yes #40;Blocked#41;" --> ZenoCheck
    BarrierSat -- "No #40;Breached#41;" --> ModeCE["<b>ENERGY BLOW-UP</b>"]

    EnergyCheck -- "Yes" --> ZenoCheck{"<b>2. ZENO EXCLUSION</b><br>Are Events Well-Founded?"}
    ZenoCheck -- "No" --> BarrierCausal{"<b>WELL-FOUNDEDNESS</b><br>Is Depth Finite?"}
    BarrierCausal -- "No #40;Breached#41;" --> ModeCC["<b>ZENO SINGULARITY</b>"]
    BarrierCausal -- "Yes #40;Blocked#41;" --> CompactCheck

    ZenoCheck -- "Yes" --> CompactCheck{"<b>3. CONCENTRATION-COMPACTNESS</b><br>Does Energy Concentrate?"}

    %% --- LEVEL 2: DUALITY ---
    CompactCheck -- "No #40;Scatters#41;" --> BarrierScat{"<b>DISPERSIVE ESTIMATES</b><br>Is Interaction Finite?"}
    BarrierScat -- "Yes #40;Benign#41;" --> ModeDD["<b>GLOBAL EXISTENCE</b><br><i>#40;Scattering#41;</i>"]
    BarrierScat -- "No #40;Pathological#41;" --> ModeCD_Alt["<b>CONCENTRATION SINGULARITY</b><br><i>#40;Via Escape#41;</i>"]

    CompactCheck -- "Yes" --> Profile["<b>PROFILE DECOMPOSITION</b>"]

    %% --- LEVEL 3: CRITICALITY ---
    Profile --> ScaleCheck{"<b>4. CRITICALITY</b><br>Is Scaling Subcritical?"}

    ScaleCheck -- "No #40;Supercritical#41;" --> BarrierTypeII{"<b>SUBCRITICAL WELL-POSEDNESS</b><br>Is Renorm Cost Infinite?"}
    BarrierTypeII -- "No #40;Breached#41;" --> ModeSE["<b>TYPE I BLOW-UP</b>"]
    BarrierTypeII -- "Yes #40;Blocked#41;" --> ParamCheck

    ScaleCheck -- "Yes #40;Safe#41;" --> ParamCheck{"<b>5. STRUCTURAL STABILITY</b><br>Are Parameters Stable?"}
    ParamCheck -- "No" --> BarrierVac{"<b>PHASE STABILITY</b><br>Is Phase Stable?"}
    BarrierVac -- "No #40;Breached#41;" --> ModeSC["<b>STRUCTURAL INSTABILITY</b>"]
    BarrierVac -- "Yes #40;Blocked#41;" --> GeomCheck

    ParamCheck -- "Yes" --> GeomCheck{"<b>6. PARTIAL REGULARITY</b><br>Is dim#40;Sing#41; < Critical?"}

    %% --- LEVEL 4: GEOMETRY ---
    GeomCheck -- "No #40;Too Thin#41;" --> BarrierCap{"<b>ε-REGULARITY</b><br>Is Measure Zero?"}
    BarrierCap -- "No #40;Breached#41;" --> ModeCD["<b>CONCENTRATION SINGULARITY</b>"]
    BarrierCap -- "Yes #40;Blocked#41;" --> StiffnessCheck

    GeomCheck -- "Yes #40;Safe#41;" --> StiffnessCheck{"<b>7. SPECTRAL ANALYSIS</b><br>Is Hessian Positive Definite?"}

    %% --- LEVEL 5: STABILITY ---
    StiffnessCheck -- "No #40;Flat#41;" --> BarrierGap{"<b>ASYMPTOTIC STABILITY</b><br>Is there a Spectral Gap?"}
    BarrierGap -- "Yes #40;Blocked#41;" --> TopoCheck
    BarrierGap -- "No #40;Stagnation#41;" --> BifurcateCheck{"<b>STABILITY OF PROFILE</b><br>Is Profile Unstable?"}

    %% --- LEVEL 5b: INSTABILITY RESOLUTION (Deterministic) ---
    BifurcateCheck -- "No #40;Stable#41;" --> ModeSD["<b>METASTABILITY</b><br><i>#40;Center Manifold#41;</i>"]
    BifurcateCheck -- "Yes #40;Unstable#41;" --> SymCheck{"<b>SYMMETRY ANALYSIS</b><br>Is Vacuum Degenerate?<br><i>#40;Does Group G exist?#41;</i>"}

    %% Path A: Symmetry Breaking
    SymCheck -- "Yes #40;Symmetric#41;" --> CheckSC{"<b>VACUUM STABILITY</b><br>Can Symmetry Break?"}
    CheckSC -- "Yes" --> ActionSSB["<b>SPONTANEOUS SYMMETRY BREAKING</b><br>Generates Mass Gap"]
    ActionSSB --> BarrierExclusion
    CheckSC -- "No" --> ModeSC_Rest["<b>STRUCTURAL INSTABILITY</b><br><i>#40;Vacuum Decay#41;</i>"]

    %% Path B: Surgery
    SymCheck -- "No #40;Asymmetric#41;" --> CheckTB{"<b>ACTION COST</b><br>Is Surgery Affordable?"}
    CheckTB -- "Yes" --> ActionSurgery["<b>SINGULARITY RESOLUTION</b><br>Topology Change"]
    ActionSurgery --> BarrierExclusion
    CheckTB -- "No" --> ModeTE_Rest["<b>TOPOLOGICAL DEFECT</b><br><i>#40;Persistent#41;</i>"]

    StiffnessCheck -- "Yes #40;Safe#41;" --> TopoCheck{"<b>8. TOPOLOGICAL INVARIANTS</b><br>Is Sector Accessible?"}

    %% --- LEVEL 6: TOPOLOGY ---
    TopoCheck -- "No #40;Protected#41;" --> BarrierAction{"<b>ENERGY GAP</b><br>Is Energy < Gap?"}
    BarrierAction -- "No #40;Breached#41;" --> ModeTE["<b>TOPOLOGICAL DEFECT</b>"]
    BarrierAction -- "Yes #40;Blocked#41;" --> TameCheck

    TopoCheck -- "Yes #40;Safe#41;" --> TameCheck{"<b>9. DEFINABILITY</b><br>Is Topology Tame?"}

    TameCheck -- "No" --> BarrierOmin{"<b>DEFINABILITY CRITERION</b><br>Is it Definable?"}
    BarrierOmin -- "No #40;Breached#41;" --> ModeTC["<b>WILD TOPOLOGY</b>"]
    BarrierOmin -- "Yes #40;Blocked#41;" --> ErgoCheck

    TameCheck -- "Yes" --> ErgoCheck{"<b>10. ERGODICITY</b><br>Does System Mix?"}

    ErgoCheck -- "No" --> BarrierMix{"<b>ESCAPE RATE</b><br>Is Trap Escapable?"}
    BarrierMix -- "No #40;Breached#41;" --> ModeTD["<b>DYNAMICAL TRAPPING</b>"]
    BarrierMix -- "Yes #40;Blocked#41;" --> ComplexCheck

    ErgoCheck -- "Yes" --> ComplexCheck{"<b>11. MODULI STRUCTURE</b><br>Is Solution Computable?"}

    %% --- LEVEL 7: COMPLEXITY ---
    ComplexCheck -- "No" --> BarrierEpi{"<b>FINITE DESCRIPTION</b><br>Is Description Finite?"}
    BarrierEpi -- "No #40;Breached#41;" --> ModeDC["<b>UNDECIDABILITY</b>"]
    BarrierEpi -- "Yes #40;Blocked#41;" --> OscillateCheck

    ComplexCheck -- "Yes" --> OscillateCheck{"<b>12. OSCILLATION</b><br>Does Solution Oscillate?"}

    OscillateCheck -- "Yes" --> BarrierFreq{"<b>OSCILLATION BOUND</b><br>Is Integral Finite?"}
    BarrierFreq -- "No #40;Breached#41;" --> ModeDE["<b>OSCILLATORY BLOW-UP</b>"]
    BarrierFreq -- "Yes #40;Blocked#41;" --> BoundaryCheck

    OscillateCheck -- "No" --> BoundaryCheck{"<b>13. BOUNDARY CONDITIONS</b><br>Is System Open?"}

    %% --- LEVEL 8: BOUNDARY ---
    BoundaryCheck -- "Yes" --> OverloadCheck{"<b>14. INPUT CONTROL</b><br>Is Input Bounded?"}

    OverloadCheck -- "Yes" --> BarrierBode{"<b>SENSITIVITY BOUND</b><br>Is Sensitivity Bounded?"}
    BarrierBode -- "No #40;Breached#41;" --> ModeBE["<b>INPUT SATURATION</b>"]

    OverloadCheck -- "No" --> StarveCheck{"<b>15. RESOURCE SUPPLY</b><br>Is Input Sufficient?"}
    StarveCheck -- "Yes" --> BarrierInput{"<b>RESOURCE BOUND</b><br>Is Reserve Sufficient?"}
    BarrierInput -- "No #40;Breached#41;" --> ModeBD["<b>RESOURCE DEPLETION</b>"]

    StarveCheck -- "No" --> AlignCheck{"<b>16. CONTROL ALIGNMENT</b><br>Is it Aligned?"}
    AlignCheck -- "No" --> BarrierVariety{"<b>REQUISITE VARIETY</b><br>Does Control Match Disturbance?"}
    BarrierVariety -- "No #40;Breached#41;" --> ModeBC["<b>CONTROL MISMATCH</b>"]

    %% --- LEVEL 9: STRUCTURAL EXCLUSION ---
    %% All successful paths funnel here
    BoundaryCheck -- "No" --> BarrierExclusion
    BarrierBode -- "Yes #40;Blocked#41;" --> BarrierExclusion
    BarrierInput -- "Yes #40;Blocked#41;" --> BarrierExclusion
    BarrierVariety -- "Yes #40;Blocked#41;" --> BarrierExclusion
    AlignCheck -- "Yes" --> BarrierExclusion

    BarrierExclusion{"<b>17. STRUCTURAL EXCLUSION</b><br>Is Hom#40;Sing, S#41; Empty?"}

    BarrierExclusion -- "Yes #40;Blocked#41;" --> VICTORY(["<b>GLOBAL REGULARITY</b><br><i>#40;All Obstructions Excluded#41;</i>"])
    BarrierExclusion -- "No #40;Morphism Exists#41;" --> ModeCat["<b>LOGICAL INCONSISTENCY</b>"]

    %% ====== STYLES ======
    %% Success states - Green
    style VICTORY fill:#22c55e,stroke:#16a34a,color:#000000,stroke-width:4px
    style ModeDD fill:#22c55e,stroke:#16a34a,color:#000000

    %% Failure modes - Red
    style ModeCE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCD_Alt fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeSD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeDE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBE fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBD fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeBC fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeCat fill:#ef4444,stroke:#dc2626,color:#ffffff

    %% Barriers - Orange/Amber
    style BarrierSat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCausal fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierScat fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierTypeII fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVac fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierCap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierGap fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierAction fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierOmin fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierMix fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierEpi fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierFreq fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierBode fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierInput fill:#f59e0b,stroke:#d97706,color:#000000
    style BarrierVariety fill:#f59e0b,stroke:#d97706,color:#000000

    %% The Final Gate - Purple with thick border
    style BarrierExclusion fill:#8b5cf6,stroke:#7c3aed,color:#ffffff,stroke-width:4px

    %% Axiom Checks - Blue
    style EnergyCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ZenoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CompactCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ScaleCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ParamCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style GeomCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StiffnessCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TopoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style TameCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ErgoCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style ComplexCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OscillateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style BoundaryCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style OverloadCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style StarveCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style AlignCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Restoration checks - Blue (standard axiom checks)
    style BifurcateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style SymCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckSC fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckTB fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Intermediate nodes - Purple
    style Start fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style Profile fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration mechanisms - Purple (escape mechanisms)
    style ActionSSB fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style ActionSurgery fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration failure modes - Red
    style ModeSC_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
```
