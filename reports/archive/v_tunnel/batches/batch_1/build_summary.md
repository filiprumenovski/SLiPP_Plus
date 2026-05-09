# v_tunnel build summary

- rows: 1163
- structures: 100
- warnings: 196
- elapsed_s: 2241.2
- output: processed/v_tunnel/batches/batch_1.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 1
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [100, 200)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.831
- has_tunnel_frac: 0.718
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count      | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ----------------- | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| ADN      | 3.2               | 4.095267820880455     | 0.990326653311341                | 0.0                       | 1.1100987956896136       | 0.4359145969563447        | -0.3144245801897351           | -0.535                | 0.07657175422110014              | 16.803701385890474 | 102.97549476457576  | 0.6029808702085748    | 0.0                     | 0.3570556491221995       | 0.06                         | 1.0                           | 0.935                        | 0.535              |
| B12      | 5.076923076923077 | 5.346007782038177     | 1.0553268438655403               | 0.0                       | 1.126190731362392        | 0.48077275072412173       | -0.4607959481220307           | 0.16783216783216784   | 0.12452132971552149              | 23.553035008500903 | 171.2053808150566   | 0.6076395483872621    | 0.0                     | 0.5407633776326093       | 0.08391608391608392          | 1.0                           | 0.916083916083916            | 0.6013986013986014 |
| PP       | 4.748780487804878 | 4.296369795811716     | 2.0631576352622134               | 0.0                       | 1.1505430562937966       | 0.7211016860069329        | -0.6272283474773646           | 0.0524390243902439    | 0.09403894608753187              | 23.190582543623908 | 136.99571042808893  | 1.0117796507968395    | 0.0                     | 0.759611309585655        | 0.08902439024390243          | 1.0                           | 0.7914634146341464           | 0.7829268292682927 |
