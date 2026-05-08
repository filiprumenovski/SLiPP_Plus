# v_tunnel build summary

- rows: 602
- structures: 100
- warnings: 36
- elapsed_s: 1329.6
- output: processed/v_tunnel/batches/batch_15.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 15
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [1500, 1600)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.940
- has_tunnel_frac: 0.884
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| PLM      | 4.102127659574468 | 10.009320728927316    | 1.2945385769419127               | 0.0                       | 1.2388024639109065       | 0.5612911745920404        | 0.2526886322832396            | 0.2851063829787234    | 0.11166680628558964              | 25.636885749523888 | 103.80854328602112  | 0.8633028789706187    | 0.0                     | 0.7438382948219824       | 0.11063829787234042          | 1.0                           | 0.9829787234042553           | 0.8468085106382979 |
| PP       | 7.002724795640327 | 4.377462139053358     | 2.544976095235343                | 0.0                       | 1.1905541237062924       | 0.8566992912168008        | -0.9197728260140009           | 0.14168937329700274   | 0.12984351309975117              | 30.240334781610805 | 236.28910463712216  | 1.1856351071061006    | 0.0                     | 0.7048569083952496       | 0.04632152588555858          | 1.0                           | 0.9128065395095368           | 0.9073569482288828 |
