# Tunnel Features

`v_tunnel` adds 15 CAVER-derived columns to `v_sterol`. The goal is to separate pockets that merely look elongated in their alpha-sphere cloud from pockets that have a real exit tunnel through the protein. That distinction is biologically relevant for the remaining fatty-acid and steryl-ester errors: steryl esters should need longer, more buried access paths than free fatty acids, while drug-like non-lipids often occupy compact sites with little tunnel structure.

The primary tunnel is the tunnel with the highest CAVER throughput for a pocket starting point. Its length, bottleneck radius, average radius, curvature, and throughput describe the main route out of the pocket. `tunnel_length_over_axial` compares that route with the alpha-sphere axial length, and `tunnel_extends_beyond_pocket` marks routes that continue at least 3 A past the pocket extent.

The aggregate geometry columns count and summarize all tunnels assigned to the same starting point: `tunnel_count`, `tunnel_max_length`, `tunnel_total_length`, `tunnel_min_bottleneck`, and `tunnel_branching_factor`. These are intended to capture whether a pocket is a single buried channel, a branched access network, or a surface-exposed site with no meaningful tunnel.

The lining chemistry columns use CAVER's residue table for the primary tunnel. `tunnel_primary_hydrophobicity` is the mean Kyte-Doolittle score, `tunnel_primary_charge` is `LYS + ARG - ASP - GLU`, and `tunnel_primary_aromatic_fraction` is the aromatic residue fraction across PHE, TYR, TRP, and HIS. All failure modes emit finite safe defaults so missing CAVER output is visible to the model as zero-tunnel signal rather than as dropped rows.
