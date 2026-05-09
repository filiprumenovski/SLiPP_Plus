# v_tunnel build summary

- rows: 493
- structures: 100
- warnings: 40
- elapsed_s: 1094.1
- output: processed/v_tunnel/batches/batch_14.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 14
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [1400, 1500)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.919
- has_tunnel_frac: 0.878
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| PLM      | 4.833333333333333 | 9.053627175281287     | 1.4400806974922746               | 0.0                       | 1.2595780231099165       | 0.6039429416083936        | 0.5581273448143885            | 0.2849462365591398    | 0.11442748653165427              | 29.38877330005646  | 137.90241438078547  | 0.8926090180413492    | 0.0                     | 0.5753029070227669       | 0.08602150537634409          | 1.0                           | 0.967741935483871            | 0.8763440860215054 |
| PP       | 6.543973941368078 | 4.973389253950055     | 2.3647776391766744               | 0.0                       | 1.1921638349990173       | 0.8101904458239461        | -0.7267615517789332           | -0.019543973941368076 | 0.11044776668011119              | 30.289442809606054 | 226.07847839091377  | 1.1261309147431258    | 0.0                     | 0.8953070361665271       | 0.09446254071661238          | 1.0                           | 0.8892508143322475           | 0.8794788273615635 |
