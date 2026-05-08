# v_tunnel build summary

- rows: 608
- structures: 100
- warnings: 59
- elapsed_s: 1.1
- output: processed/v_tunnel/batches/batch_0.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 0
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [0, 100)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.903
- has_tunnel_frac: 0.822
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| ADN      | 4.411214953271028 | 4.604671943561966     | 1.623292833424353                | 0.0                       | 1.1284698027084128       | 0.6663282542272034        | -0.5822214184885124           | -0.7242990654205608   | 0.08709110656808429              | 23.185664100001322 | 113.53706457778122  | 0.8593636984241083    | 0.0                     | 0.34714841741805524      | 0.04205607476635514          | 1.0                           | 0.9766355140186916           | 0.7663551401869159 |
| PP       | 6.644670050761421 | 4.622244359180733     | 2.4500013545113637               | 0.0                       | 1.1956656937036716       | 0.8011742857540971        | -0.7999690280581613           | 0.11421319796954314   | 0.10239494855907028              | 29.386524016080575 | 228.76193572010106  | 1.1425414133265788    | 0.0                     | 0.7269300519346235       | 0.08629441624365482          | 1.0                           | 0.8629441624365483           | 0.8527918781725888 |
