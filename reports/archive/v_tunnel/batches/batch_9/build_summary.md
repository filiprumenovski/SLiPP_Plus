# v_tunnel build summary

- rows: 964
- structures: 100
- warnings: 192
- elapsed_s: 5162.8
- output: processed/v_tunnel/batches/batch_9.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 9
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [900, 1000)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.801
- has_tunnel_frac: 0.731
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| COA      | 3.485639686684073 | 5.508950333538906     | 1.3098840205447182               | 0.0                       | 1.149286847074563        | 0.5897335767382085        | -0.38711225967639545          | 0.6005221932114883    | 0.09061091451515246              | 20.367751396557466 | 88.23886606101738   | 0.8075237549885876    | 0.0                     | 0.5311620978715723       | 0.09399477806788512          | 1.0                           | 0.8825065274151436           | 0.7284595300261096 |
| PP       | 4.110154905335628 | 3.9675907928264627    | 1.9351327404488767               | 0.0                       | 1.1488723283621762       | 0.6728575843088733        | -0.6408109221947929           | -0.08433734939759036  | 0.08797745007908915              | 21.332958363879595 | 107.48932567271451  | 0.9738130896696467    | 0.0                     | 0.6444806703206706       | 0.06368330464716007          | 1.0                           | 0.7469879518072289           | 0.7332185886402753 |
