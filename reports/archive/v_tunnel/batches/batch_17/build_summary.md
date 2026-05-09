# v_tunnel build summary

- rows: 547
- structures: 80
- warnings: 122
- elapsed_s: 1673.6
- output: processed/v_tunnel/batches/batch_17.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 17
- batch_size: 100
- total_structures: 1780
- selected_structures: 80
- structure_range: [1700, 1780)

## Preflight

- structures_total: 80
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.777
- has_tunnel_frac: 0.720
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| PP       | 6.35              | 3.583912287695983     | 1.9722517895728107               | 0.0                       | 1.1488072092456414       | 0.6816342213505194        | -0.3966873297438472           | 0.0775                | 0.08956927365055915              | 28.902007719969532 | 229.9024291237705   | 0.8939726816594602    | 0.0                     | 0.5753252667853592       | 0.04                         | 1.0                           | 0.7425                       | 0.73               |
| STE      | 2.687074829931973 | 8.326261127336846     | 1.1034276625354227               | 0.0                       | 1.201530550371691        | 0.47190961421131983       | 0.19420205877108876           | 0.17687074829931973   | 0.07424340256270962              | 19.173567370037023 | 63.54700317846856   | 0.7284700923137336    | 0.0                     | 0.9270649072620312       | 0.17687074829931973          | 1.0                           | 0.8707482993197279           | 0.6938775510204082 |
