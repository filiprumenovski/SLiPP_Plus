# v_tunnel build summary

- rows: 769
- structures: 100
- warnings: 111
- elapsed_s: 2636.0
- output: processed/v_tunnel/batches/batch_3.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 3
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [300, 400)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.856
- has_tunnel_frac: 0.813
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| BGC      | 4.337962962962963 | 4.848060664692566     | 1.6685280374514766               | 0.0                       | 1.1469189373575241       | 0.7323824780030902        | -0.8094312465936677           | -0.7361111111111112   | 0.17248521749224577              | 22.682933268368227 | 106.45311820127456  | 0.9865715141514918    | 0.0                     | 0.4819964270365439       | 0.05555555555555555          | 1.0                           | 0.9444444444444444           | 0.8425925925925926 |
| PP       | 4.23508137432188  | 4.26206206038465      | 2.118524750167266                | 0.0                       | 1.1688223525094246       | 0.7372749793033478        | -0.7591488538743308           | -0.16998191681735986  | 0.11172108827796727              | 21.155278638248497 | 102.04500187849129  | 1.0371822527938186    | 0.0                     | 0.7302810351159154       | 0.081374321880651            | 1.0                           | 0.8209764918625678           | 0.8010849909584087 |
