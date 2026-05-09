# v_tunnel build summary

- rows: 773
- structures: 100
- warnings: 186
- elapsed_s: 4461.2
- output: processed/v_tunnel/batches/batch_12.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 12
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [1200, 1300)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.759
- has_tunnel_frac: 0.708
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| MYR      | 6.097014925373134 | 7.499240380254747     | 1.036698325175172                | 0.0                       | 1.1707550892236804       | 0.49816074259550713       | 0.03613383669418098           | 0.5298507462686567    | 0.12163426385818256              | 31.829367920579177 | 185.19948627928974  | 0.6929111508021064    | 0.0                     | 0.4717768074910346       | 0.06716417910447761          | 1.0                           | 0.7835820895522388           | 0.7238805970149254 |
| OLA      | 3.802469135802469 | 6.90590424226439      | 1.4298129829675266               | 0.0                       | 1.21739031101405         | 0.6399380973423243        | 0.6619517295757441            | 0.42592592592592593   | 0.1230826442918908               | 21.556185006064386 | 85.61339807723189   | 0.9061190755121011    | 0.0                     | 0.7218248748682615       | 0.12962962962962962          | 1.0                           | 0.9876543209876543           | 0.845679012345679  |
| PP       | 5.39412997903564  | 3.3256291193639904    | 1.653418396874176                | 0.0                       | 1.1283333093298296       | 0.6037371731094529        | -0.2758580739841578           | -0.039832285115303984 | 0.08457327861270605              | 23.624809556045868 | 201.03258635645923  | 0.8028734983948326    | 0.0                     | 0.4949474509047757       | 0.041928721174004195         | 1.0                           | 0.6750524109014675           | 0.6561844863731656 |
