# v_tunnel build summary

- rows: 743
- structures: 100
- warnings: 189
- elapsed_s: 1547.0
- output: processed/v_tunnel/batches/batch_13.parquet
- output_size_mb: 0.2
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 13
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [1300, 1400)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.746
- has_tunnel_frac: 0.717
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count       | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ------------------ | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| OLA      | 3.7365269461077846 | 6.5695488375545485    | 1.3076955437879205               | 0.0                       | 1.1923155638812162       | 0.5944276309081578        | 0.8909095529136154            | 0.12574850299401197   | 0.1455645746707198               | 23.43191325783913  | 89.48499781094375   | 0.8197859387397909    | 0.0                     | 0.6591387468742769       | 0.0718562874251497           | 1.0                           | 0.8922155688622755           | 0.8023952095808383 |
| PLM      | 6.454545454545454  | 8.43263022665566      | 1.4873952370147248               | 0.0                       | 1.2380154603796651       | 0.6784395943569793        | 0.7633433499625039            | 0.35064935064935066   | 0.14636809868481232              | 32.317961037244665 | 216.2243324053867   | 0.9572076741154476    | 0.0                     | 0.6654621393220821       | 0.1038961038961039           | 1.0                           | 0.948051948051948            | 0.9090909090909091 |
| PP       | 5.274549098196393  | 3.8070171068392553    | 1.765216330926711                | 0.0                       | 1.1514084987384312       | 0.6021987746952253        | -0.2354308603218749           | -0.006012024048096192 | 0.09398442090678197              | 23.77080524904689  | 188.7197425863788   | 0.8107727562426204    | 0.0                     | 0.6076689576230482       | 0.0781563126252505           | 1.0                           | 0.6653306613226453           | 0.6593186372745491 |
