# v_tunnel build summary

- rows: 1589
- structures: 100
- warnings: 1118
- elapsed_s: 12017.4
- output: processed/v_tunnel/batches/batch_5.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 5
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [500, 600)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.296
- has_tunnel_frac: 0.277
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count       | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ------------------ | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ----------------- | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| CLR      | 1.5060975609756098 | 2.0993770052064162    | 0.4379935325662173               | 0.0                       | 1.0544792523614814       | 0.20317038920514063       | 0.390157203750273             | 0.036585365853658534  | 0.05247736969579003              | 7.683791395371434 | 42.49842078172776   | 0.2940271867642936    | 0.0                     | 0.23288981179693827      | 0.039634146341463415         | 1.0                           | 0.3475609756097561           | 0.2804878048780488 |
| COA      | 3.5789473684210527 | 4.702137345716618     | 1.1642413557973206               | 0.0                       | 1.101325254654233        | 0.4865178020043076        | -0.26462583327324257          | 0.5789473684210527    | 0.05571989673342125              | 18.40099828368364 | 88.47421515205997   | 0.5637800661339684    | 0.0                     | 0.4087838015214695       | 0.05263157894736842          | 1.0                           | 0.7894736842105263           | 0.5789473684210527 |
| PP       | 1.5096618357487923 | 1.3837374669474787    | 0.6780982014330189               | 0.0                       | 1.0548833206452164       | 0.2472289062862299        | -0.07866102888914434          | 0.05233494363929147   | 0.036616620588783316             | 6.568619490700966 | 36.45210245503576   | 0.3735683387185523    | 0.0                     | 0.23990643716167742      | 0.02818035426731079          | 1.0                           | 0.2753623188405797           | 0.2713365539452496 |
