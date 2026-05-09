# v_tunnel build summary

- rows: 965
- structures: 100
- warnings: 176
- elapsed_s: 6153.0
- output: processed/v_tunnel/batches/batch_7.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 7
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [700, 800)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.818
- has_tunnel_frac: 0.735
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count       | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ------------------ | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| COA      | 3.3966745843230406 | 4.606885109616237     | 1.3336311224065818               | 0.0                       | 1.1463045423836207       | 0.6064590026515061        | -0.3341441444851037           | 0.44418052256532065   | 0.09677948493888765              | 19.958489521655913 | 91.50727697233752   | 0.779769971497971     | 0.0                     | 0.3739024860478128       | 0.04513064133016627          | 1.0                           | 0.8978622327790974           | 0.7220902612826603 |
| PP       | 4.466911764705882  | 3.71406395688199      | 1.973161579718704                | 0.0                       | 1.144798027756403        | 0.6946241829001291        | -0.5326818360080298           | -0.04963235294117647  | 0.0841710999686056               | 19.750281206802057 | 124.18238003504024  | 0.9642373620106031    | 0.0                     | 0.5886806388281524       | 0.05514705882352941          | 1.0                           | 0.7555147058823529           | 0.7444852941176471 |
