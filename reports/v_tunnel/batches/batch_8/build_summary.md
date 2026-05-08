# v_tunnel build summary

- rows: 972
- structures: 100
- warnings: 178
- elapsed_s: 4412.1
- output: processed/v_tunnel/batches/batch_8.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 8
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [800, 900)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.817
- has_tunnel_frac: 0.753
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| COA      | 3.502314814814815 | 5.091007489484518     | 1.2989237278230237               | 0.0                       | 1.1500360474681162       | 0.6067316589348102        | -0.19873512625273118          | 0.32175925925925924   | 0.0977780499764106               | 20.44276549788706  | 89.44002671756182   | 0.8187034484454655    | 0.0                     | 0.4498868745081634       | 0.05787037037037037          | 1.0                           | 0.8842592592592593           | 0.7523148148148148 |
| PP       | 4.177777777777778 | 3.950171919043405     | 1.9434837532888445               | 0.0                       | 1.1597872476819093       | 0.6946751687590986        | -0.6481249633047329           | 0.016666666666666666  | 0.08345175207482695              | 20.713250619441027 | 105.31312894068425  | 0.9964217740021523    | 0.0                     | 0.6452233653993359       | 0.07777777777777778          | 1.0                           | 0.762962962962963            | 0.7537037037037037 |
