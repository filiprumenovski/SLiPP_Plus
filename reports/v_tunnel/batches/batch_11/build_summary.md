# v_tunnel build summary

- rows: 668
- structures: 100
- warnings: 29
- elapsed_s: 1378.5
- output: processed/v_tunnel/batches/batch_11.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 11
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [1100, 1200)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.957
- has_tunnel_frac: 0.918
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge  | tunnel_primary_aromatic_fraction | tunnel_max_length | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | ---------------------- | -------------------------------- | ----------------- | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| MYR      | 8.168539325842696 | 8.777711614332215     | 1.4456068764520993               | 0.0                       | 1.2233928517354022       | 0.6417745606975382        | 0.17525585775063135           | 0.3595505617977528     | 0.13713785289082053              | 39.94359686832171 | 270.35605483943317  | 0.882071228332303     | 0.0                     | 0.5795240192625735       | 0.09363295880149813          | 1.0                           | 0.9887640449438202           | 0.9063670411985019 |
| PP       | 9.209476309226932 | 4.777178746401737     | 2.495559051710091                | 0.0                       | 1.2082699183350643       | 0.8668552219123058        | -0.8115570947806732           | -0.0024937655860349127 | 0.1165984315260044               | 37.20572584150641 | 376.6364976575893   | 1.1470713303539957    | 0.0                     | 0.7155762715659858       | 0.07231920199501247          | 1.0                           | 0.9351620947630923           | 0.9251870324189526 |
