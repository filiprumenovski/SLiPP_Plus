# v_tunnel build summary

- rows: 1023
- structures: 100
- warnings: 149
- elapsed_s: 3738.5
- output: processed/v_tunnel/batches/batch_2.parquet
- output_size_mb: 0.3
- cache_dir: processed/v_tunnel/structure_json
- analysis_output_root: None
- analysis_manifest_path: None

## Batch

- batch_index: 2
- batch_size: 100
- total_structures: 1780
- selected_structures: 100
- structure_range: [200, 300)

## Preflight

- structures_total: 100
- structures_with_missing_inputs: 0
- missing_input_frac: 0.000
- missing_protein_pdb: 0
- missing_structure_dir: 0
- missing_pockets_dir: 0

## Quality

- context_present_frac: 1.000
- profile_present_frac: 0.854
- has_tunnel_frac: 0.747
- min_context_present_frac: 0.000
- min_profile_present_frac: 0.000
- max_missing_structure_frac: 0.020

## Class means

| class_10 | tunnel_count       | tunnel_primary_length | tunnel_primary_bottleneck_radius | tunnel_primary_avg_radius | tunnel_primary_curvature | tunnel_primary_throughput | tunnel_primary_hydrophobicity | tunnel_primary_charge | tunnel_primary_aromatic_fraction | tunnel_max_length  | tunnel_total_length | tunnel_min_bottleneck | tunnel_branching_factor | tunnel_length_over_axial | tunnel_extends_beyond_pocket | tunnel_pocket_context_present | tunnel_caver_profile_present | tunnel_has_tunnel  |
| -------- | ------------------ | --------------------- | -------------------------------- | ------------------------- | ------------------------ | ------------------------- | ----------------------------- | --------------------- | -------------------------------- | ------------------ | ------------------- | --------------------- | ----------------------- | ------------------------ | ---------------------------- | ----------------------------- | ---------------------------- | ------------------ |
| B12      | 2.7043478260869565 | 7.249011454306708     | 0.8593798322665872               | 0.0                       | 1.1655332404519685       | 0.381560486971177         | -0.3792171724798064           | -0.2608695652173913   | 0.09924455823253103              | 20.554429462690248 | 94.09731942491673   | 0.6459419887326044    | 0.0                     | 0.7527418699257614       | 0.18695652173913044          | 1.0                           | 0.9260869565217391           | 0.5782608695652174 |
| BGC      | 4.587786259541985  | 6.259641704095167     | 1.420243507048309                | 0.0                       | 1.1358468097766112       | 0.6209801808770867        | -0.7213331710837195           | -0.7022900763358778   | 0.14839003734462466              | 32.014724409314375 | 134.41131145404276  | 0.8130950832450312    | 0.0                     | 0.5850524332471932       | 0.11450381679389313          | 1.0                           | 0.9236641221374046           | 0.7709923664122137 |
| PP       | 4.81570996978852   | 4.413083485420209     | 2.042588843104803                | 0.0                       | 1.1657472101942246       | 0.7325579942274323        | -0.753639158692845            | -0.08157099697885196  | 0.09476114088869932              | 24.97166973126145  | 137.6747263122922   | 1.0223745711067094    | 0.0                     | 0.7629620232931549       | 0.08761329305135952          | 1.0                           | 0.8157099697885196           | 0.8006042296072508 |
