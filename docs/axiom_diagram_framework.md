
```mermaid
graph TD
    Start(["<b>START DIAGNOSTIC</b>"])

    %% ==================== AXIOM D: DISSIPATION ====================
    subgraph AXIOM_D["<b>D: DISSIPATION</b>"]
        EnergyCheck{"<b>1. AXIOM D #40;Dissipation#41;</b><br>Is Global Energy Finite?"}
        BarrierSat{"<b>SATURATION BARRIER</b><br>Is Drift Controlled?<br>"}
        ModeCE["<b>Mode C.E</b>: Energy Blow-Up"]
        SurgCE["<b>SURGERY:</b><br>Ghost/Cap"]

        EnergyCheck -- "No" --> BarrierSat
        BarrierSat -- "No #40;Breached#41;" --> ModeCE
        ModeCE --> SurgCE
    end

    Start --> EnergyCheck

    %% ==================== AXIOM REC: RECOVERY ====================
    subgraph AXIOM_REC["<b>REC: RECOVERY</b>"]
        ZenoCheck{"<b>2. AXIOM REC #40;Recovery#41;</b><br>Are Discrete Events Finite?"}
        BarrierCausal{"<b>CAUSAL CENSOR</b><br>Is Depth Finite?<br>"}
        ModeCC["<b>Mode C.C</b>: Event Accumulation"]
        SurgCC["<b>SURGERY:</b><br>Discrete Saturation"]

        ZenoCheck -- "No" --> BarrierCausal
        BarrierCausal -- "No #40;Breached#41;" --> ModeCC
        ModeCC --> SurgCC
    end

    BarrierSat -- "Yes #40;Blocked#41;" --> ZenoCheck
    SurgCE -.-> ZenoCheck
    EnergyCheck -- "Yes" --> ZenoCheck

    %% ==================== AXIOM C: COMPACTNESS ====================
    subgraph AXIOM_C["<b>C: COMPACTNESS</b>"]
        CompactCheck{"<b>3. AXIOM C #40;Compactness#41;</b><br>Does Energy Concentrate?"}
        BarrierScat{"<b>SCATTERING BARRIER</b><br>Is Interaction Finite?<br>"}
        ModeDD["<b>Mode D.D</b>: Dispersion<br><i>#40;Global Existence#41;</i>"]
        ModeCD_Alt["<b>Mode C.D</b>: Geometric Collapse<br><i>#40;Via Escape#41;</i>"]
        SurgCD_Alt["<b>SURGERY:</b><br>Concentration-Compactness"]
        Profile["<b>Canonical Profile V Emerges</b>"]

        CompactCheck -- "No #40;Scatters#41;" --> BarrierScat
        BarrierScat -- "Yes #40;Benign#41;" --> ModeDD
        BarrierScat -- "No #40;Pathological#41;" --> ModeCD_Alt
        ModeCD_Alt --> SurgCD_Alt
        CompactCheck -- "Yes" --> Profile
    end

    BarrierCausal -- "Yes #40;Blocked#41;" --> CompactCheck
    SurgCC -.-> CompactCheck
    ZenoCheck -- "Yes" --> CompactCheck
    SurgCD_Alt -.-> Profile

    %% ==================== AXIOM SC: SCALING ====================
    subgraph AXIOM_SC["<b>SC: SCALING</b>"]
        ScaleCheck{"<b>4. AXIOM SC #40;Scaling#41;</b><br>Is it Subcritical?"}
        BarrierTypeII{"<b>TYPE II BARRIER</b><br>Is Renorm Cost Infinite?<br>"}
        ModeSE["<b>Mode S.E</b>: Supercritical Cascade"]
        SurgSE["<b>SURGERY:</b><br>Regularity Lift"]
        ParamCheck{"<b>5. AXIOM SC #40;Stability#41;</b><br>Are Constants Stable?"}
        BarrierVac{"<b>VACUUM BARRIER</b><br>Is Phase Stable?<br>"}
        ModeSC["<b>Mode S.C</b>: Parameter Instability"]
        SurgSC["<b>SURGERY:</b><br>Convex Integration"]

        ScaleCheck -- "No #40;Supercritical#41;" --> BarrierTypeII
        BarrierTypeII -- "No #40;Breached#41;" --> ModeSE
        ModeSE --> SurgSE
        ScaleCheck -- "Yes #40;Safe#41;" --> ParamCheck
        ParamCheck -- "No" --> BarrierVac
        BarrierVac -- "No #40;Breached#41;" --> ModeSC
        ModeSC --> SurgSC
    end

    Profile --> ScaleCheck
    BarrierTypeII -- "Yes #40;Blocked#41;" --> ParamCheck
    SurgSE -.-> ParamCheck

    %% ==================== AXIOM CAP: CAPACITY ====================
    subgraph AXIOM_CAP["<b>CAP: CAPACITY</b>"]
        GeomCheck{"<b>6. AXIOM CAP #40;Capacity#41;</b><br>Is Dimension > Critical?"}
        BarrierCap{"<b>CAPACITY BARRIER</b><br>Is Measure Zero?<br>"}
        ModeCD["<b>Mode C.D</b>: Geometric Collapse"]
        SurgCD["<b>SURGERY:</b><br>Aux/Structural"]

        GeomCheck -- "No #40;Too Thin#41;" --> BarrierCap
        BarrierCap -- "No #40;Breached#41;" --> ModeCD
        ModeCD --> SurgCD
    end

    BarrierVac -- "Yes #40;Blocked#41;" --> GeomCheck
    SurgSC -.-> GeomCheck
    ParamCheck -- "Yes" --> GeomCheck

    %% ==================== AXIOM LS: STIFFNESS ====================
    subgraph AXIOM_LS["<b>LS: STIFFNESS</b>"]
        StiffnessCheck{"<b>7. AXIOM LS #40;Stiffness#41;</b><br>Is Hessian Positive?"}
        BarrierGap{"<b>SPECTRAL BARRIER</b><br>Is there a Gap?<br>"}
        BifurcateCheck{"<b>7a. BIFURCATION CHECK #40;Axiom LS#41;</b><br>Is State Unstable?<br>"}
        ModeSD["<b>Mode S.D</b>: Stiffness Breakdown"]
        SurgSD["<b>SURGERY:</b><br>Ghost Extension"]
        SymCheck{"<b>7b. SYMMETRY CHECK</b><br>Is Vacuum Degenerate?<br><i>#40;Does Group G exist?#41;</i>"}
        CheckSC{"<b>7c. AXIOM SC #40;Param#41;</b><br>Are Constants Stable?<br>"}
        ActionSSB["<b>ACTION: SYM. BREAKING</b><br>Generates Mass Gap"]
        ModeSC_Rest["<b>Mode S.C</b>: Parameter Instability<br><i>#40;Vacuum Decay#41;</i>"]
        SurgSC_Rest["<b>SURGERY:</b><br>Auxiliary Extension"]
        CheckTB{"<b>7d. AXIOM TB #40;Action#41;</b><br>Is Cost Finite?<br>"}
        ActionTunnel["<b>ACTION: TUNNELING</b><br>Instanton Decay"]
        ModeTE_Rest["<b>Mode T.E</b>: Topological Twist<br><i>#40;Metastasis#41;</i>"]
        SurgTE_Rest["<b>SURGERY:</b><br>Structural"]

        StiffnessCheck -- "No #40;Flat#41;" --> BarrierGap
        BarrierGap -- "No #40;Stagnation#41;" --> BifurcateCheck
        BifurcateCheck -- "No #40;Stable#41;" --> ModeSD
        ModeSD --> SurgSD
        BifurcateCheck -- "Yes #40;Unstable#41;" --> SymCheck
        SymCheck -- "Yes #40;Symmetric#41;" --> CheckSC
        CheckSC -- "Yes" --> ActionSSB
        CheckSC -- "No" --> ModeSC_Rest
        ModeSC_Rest --> SurgSC_Rest
        SymCheck -- "No #40;Asymmetric#41;" --> CheckTB
        CheckTB -- "Yes" --> ActionTunnel
        CheckTB -- "No" --> ModeTE_Rest
        ModeTE_Rest --> SurgTE_Rest
    end

    BarrierCap -- "Yes #40;Blocked#41;" --> StiffnessCheck
    SurgCD -.-> StiffnessCheck
    GeomCheck -- "Yes #40;Safe#41;" --> StiffnessCheck

    %% ==================== AXIOM TB: TOPOLOGY ====================
    subgraph AXIOM_TB["<b>TB: TOPOLOGY</b>"]
        TopoCheck{"<b>8. AXIOM TB #40;Topology#41;</b><br>Is Sector Accessible?"}
        BarrierAction{"<b>ACTION BARRIER</b><br>Is Energy < Gap?<br>"}
        ModeTE["<b>Mode T.E</b>: Topological Twist"]
        SurgTE["<b>SURGERY:</b><br>Tunnel"]
        TameCheck{"<b>9. AXIOM TB #40;Tameness#41;</b><br>Is Topology Simple?"}
        BarrierOmin{"<b>O-MINIMAL BARRIER</b><br>Is it Definable?<br>"}
        ModeTC["<b>Mode T.C</b>: Labyrinthine"]
        SurgTC["<b>SURGERY:</b><br>O-minimal Regularization"]
        ErgoCheck{"<b>10. AXIOM TB #40;Mixing#41;</b><br>Does it Mix?"}
        BarrierMix{"<b>MIXING BARRIER</b><br>Is Trap Escapable?<br>"}
        ModeTD["<b>Mode T.D</b>: Glassy Freeze"]
        SurgTD["<b>SURGERY:</b><br>Mixing Enhancement"]

        TopoCheck -- "No #40;Protected#41;" --> BarrierAction
        BarrierAction -- "No #40;Breached#41;" --> ModeTE
        ModeTE --> SurgTE
        TopoCheck -- "Yes #40;Safe#41;" --> TameCheck
        TameCheck -- "No" --> BarrierOmin
        BarrierOmin -- "No #40;Breached#41;" --> ModeTC
        ModeTC --> SurgTC
        TameCheck -- "Yes" --> ErgoCheck
        ErgoCheck -- "No" --> BarrierMix
        BarrierMix -- "No #40;Breached#41;" --> ModeTD
        ModeTD --> SurgTD
    end

    BarrierGap -- "Yes #40;Blocked#41;" --> TopoCheck
    SurgSD -.-> TopoCheck
    StiffnessCheck -- "Yes #40;Safe#41;" --> TopoCheck
    ActionSSB -- "Mass Gap Guarantees Stiffness" --> TopoCheck
    SurgSC_Rest -.-> TopoCheck
    ActionTunnel -- "New Sector Reached" --> TameCheck
    SurgTE_Rest -.-> TameCheck
    BarrierAction -- "Yes #40;Blocked#41;" --> TameCheck
    SurgTE -.-> TameCheck
    BarrierOmin -- "Yes #40;Blocked#41;" --> ErgoCheck
    SurgTC -.-> ErgoCheck

    %% ==================== AXIOM REP: DICTIONARY ====================
    subgraph AXIOM_REP["<b>REP: DICTIONARY</b>"]
        ComplexCheck{"<b>11. AXIOM REP #40;Dictionary#41;</b><br>Is it Computable?"}
        BarrierEpi{"<b>EPISTEMIC BARRIER</b><br>Is Description Finite?<br>"}
        ModeDC["<b>Mode D.C</b>: Semantic Horizon"]
        SurgDC["<b>SURGERY:</b><br>Viscosity Solution"]

        ComplexCheck -- "No" --> BarrierEpi
        BarrierEpi -- "No #40;Breached#41;" --> ModeDC
        ModeDC --> SurgDC
    end

    BarrierMix -- "Yes #40;Blocked#41;" --> ComplexCheck
    SurgTD -.-> ComplexCheck
    ErgoCheck -- "Yes" --> ComplexCheck

    %% ==================== AXIOM GC: GRADIENT ====================
    subgraph AXIOM_GC["<b>GC: GRADIENT</b>"]
        OscillateCheck{"<b>12. AXIOM GC #40;Gradient#41;</b><br>Does it Oscillate?"}
        BarrierFreq{"<b>FREQUENCY BARRIER</b><br>Is Integral Finite?<br>"}
        ModeDE["<b>Mode D.E</b>: Oscillatory"]
        SurgDE["<b>SURGERY:</b><br>De Giorgi-Nash-Moser"]

        OscillateCheck -- "Yes" --> BarrierFreq
        BarrierFreq -- "No #40;Breached#41;" --> ModeDE
        ModeDE --> SurgDE
    end

    BarrierEpi -- "Yes #40;Blocked#41;" --> OscillateCheck
    SurgDC -.-> OscillateCheck
    ComplexCheck -- "Yes" --> OscillateCheck

    %% ==================== AXIOM BOUND: BOUNDARY ====================
    subgraph AXIOM_BOUND["<b>BOUND: BOUNDARY</b>"]
        BoundaryCheck{"<b>13. BOUNDARY CHECK</b><br>Is System Open?"}
        OverloadCheck{"<b>14. BOUNDARY: CONTROL</b><br>Is Input Bounded?"}
        BarrierBode{"<b>BODE BARRIER</b><br>Is Sensitivity Bounded?<br>"}
        ModeBE["<b>Mode B.E</b>: Injection"]
        SurgBE["<b>SURGERY:</b><br>Saturation"]
        StarveCheck{"<b>15. BOUNDARY: SUPPLY</b><br>Is Input Sufficient?"}
        BarrierInput{"<b>INPUT BARRIER</b><br>Is Reserve Sufficient?<br>"}
        ModeBD["<b>Mode B.D</b>: Starvation"]
        SurgBD["<b>SURGERY:</b><br>Reservoir"]
        AlignCheck{"<b>16. BOUNDARY: GAUGE</b><br>Is it Aligned?"}
        BarrierVariety{"<b>VARIETY BARRIER</b><br>Does Control Match Disturbance?<br>"}
        ModeBC["<b>Mode B.C</b>: Misalignment"]
        SurgBC["<b>SURGERY:</b><br>Adjoint"]

        BoundaryCheck -- "Yes" --> OverloadCheck
        OverloadCheck -- "No" --> BarrierBode
        BarrierBode -- "No #40;Breached#41;" --> ModeBE
        ModeBE --> SurgBE
        OverloadCheck -- "Yes" --> StarveCheck
        StarveCheck -- "No" --> BarrierInput
        BarrierInput -- "No #40;Breached#41;" --> ModeBD
        ModeBD --> SurgBD
        StarveCheck -- "Yes" --> AlignCheck
        AlignCheck -- "No" --> BarrierVariety
        BarrierVariety -- "No #40;Breached#41;" --> ModeBC
        ModeBC --> SurgBC
    end

    BarrierFreq -- "Yes #40;Blocked#41;" --> BoundaryCheck
    SurgDE -.-> BoundaryCheck
    OscillateCheck -- "No" --> BoundaryCheck
    BarrierBode -- "Yes #40;Blocked#41;" --> StarveCheck
    SurgBE -.-> StarveCheck
    BarrierInput -- "Yes #40;Blocked#41;" --> AlignCheck
    SurgBD -.-> AlignCheck

    %% ==================== THE CATEGORICAL LOCK ====================
    subgraph LOCK_BOX["<b>LOCK</b>"]
        BarrierExclusion{"<b>17. THE CATEGORICAL LOCK</b><br>Is Hom#40;Bad, S#41; Empty?<br>"}
        VICTORY(["<b>GLOBAL REGULARITY</b><br><i>#40;Structural Exclusion Confirmed#41;</i>"])
        ModeCat["<b>FATAL ERROR</b><br>Structural Inconsistency"]

        BarrierExclusion -- "Yes #40;Blocked#41;" --> VICTORY
        BarrierExclusion -- "No #40;Morphism Exists#41;" --> ModeCat
    end

    BoundaryCheck -- "No" --> BarrierExclusion
    BarrierVariety -- "Yes #40;Blocked#41;" --> BarrierExclusion
    AlignCheck -- "Yes" --> BarrierExclusion
    SurgBC -.-> BarrierExclusion

    %% ==================== STYLES ====================
    %% Axiom Boxes - Dark blue with blue border
    style AXIOM_D fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_REC fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_C fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_SC fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_CAP fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_LS fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_TB fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_REP fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_GC fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px
    style AXIOM_BOUND fill:#1e3a5f,stroke:#3b82f6,color:#ffffff,stroke-width:2px

    %% The Lock - Purple with thick border
    style LOCK_BOX fill:#4c1d95,stroke:#8b5cf6,color:#ffffff,stroke-width:3px

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
    style ModeSC_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff
    style ModeTE_Rest fill:#ef4444,stroke:#dc2626,color:#ffffff

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

    %% Intermediate nodes - Purple
    style Start fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style Profile fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Restoration checks - Blue
    style BifurcateCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style SymCheck fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckSC fill:#3b82f6,stroke:#2563eb,color:#ffffff
    style CheckTB fill:#3b82f6,stroke:#2563eb,color:#ffffff

    %% Restoration mechanisms - Purple
    style ActionSSB fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style ActionTunnel fill:#8b5cf6,stroke:#7c3aed,color:#ffffff

    %% Surgery recovery nodes - Purple
    style SurgCE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD_Alt fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgCD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgSC_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE_Rest fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgTD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgDE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBE fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBD fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
    style SurgBC fill:#8b5cf6,stroke:#7c3aed,color:#ffffff
```
