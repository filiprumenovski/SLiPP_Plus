# v_tunnel build summary

- rows: 816
- structures: 100
- warnings: 136
- elapsed_s: 3151.3
- output: processed/v_tunnel/batches/batch_6.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 6
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [600, 700)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.833
- has_tunnel_frac: 0.777
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count       | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ------------------ | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| COA      | 5.0394736842105265 | 5.560045801670851     | 1.512860314519498                | 0.0                       | 1.1560441166835267       | 0.6463259373535865        | -0.34990456557551874          | 0.2710526315789474    | 0.09664508627819994              | 29.124823184103327 | 160.3335748292822   | 0.8570849463090779    | 0.0                     | 0.598004536250673        | 0.09210526315789473          | 1.0                           | 0.8868421052631579           | 0.7789473684210526 |
| PP       | 4.690366972477064  | 4.343004068800945     | 2.1261673106363204               | 0.0                       | 1.1886472068453986       | 0.7174736974092699        | -0.6071529891327948           | -0.07110091743119266  | 0.0877710233107856               | 21.240416509638603 | 116.59571417708648  | 1.0034154243807458    | 0.0                     | 0.7181574519437774       | 0.0779816513761468           | 1.0                           | 0.786697247706422            | 0.7752293577981652 |
