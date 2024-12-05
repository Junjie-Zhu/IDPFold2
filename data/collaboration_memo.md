# A brief memo on the collaboration

<hr>

## Discussion 12.04

- Use trajectories described in CALVADOS3 as test set

<hr>

- Collect entries from AFDB for running more simulations as training set
  - Filter entries of low sequence similarity
  - Better folded-disordered-folded states (at least balence in secondary structure to avoid helix hallucination)

- Expected size of fineturning dataset: < 150,000
  - Expected number of entries: 1,000 ~ 5,000 
  - Expected number of conformations per entry: 20 ~ 100

<hr>

- Problems in trajectory process
  - CALVADOS3 simulations result in COM representation for folded regions while CA for disordered regions
  - Folded regions tend to remain stable, which may provide limited information for training

- Expected solution
  - Backmapping can be done with cg2all (or our pretrained model)
    - We take reference coordinates from CCD for each residue
    - CG representations are used to move the reference coordinates to a target position
      - We first recenter the reference coordinates to the origin
      - For COM representations, we add each COM in CG structures to the recentered reference coordinates
      - For CA representations, we add $coord_{CA}^{CG} - coord_{CA}^{ref}$ to the recentered reference coordinates
      - The translated reference coordinates are then randomly rotated to improve the robustness of the model
    - We also directly give model the pairwise distance of CG representations to model

<hr>
