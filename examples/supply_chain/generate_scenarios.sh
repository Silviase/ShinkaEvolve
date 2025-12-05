#!/usr/bin/env bash
# Generate curated supply chain scenarios into configs/generated/<name>.
# Adjust seeds/num_cases if needed. Requires `uv` in PATH.

set -euo pipefail

# basic-weekend-bump (Lv1,スター型基準)
uv run python examples/supply_chain/case_generator.py \
  --level 1 --num_cases 30 --seed 101 \
  --out_dir examples/supply_chain/configs/generated/lv1_basic_weekend_bump \
  --preview_dir examples/supply_chain/configs/generated/lv1_basic_weekend_bump \
  --preview \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --stem_base case

# normal-some-bottleneck (Lv2, 細めツリーで単一ボトルネック想定)
uv run python examples/supply_chain/case_generator.py \
  --level 2 --num_cases 30 --seed 201 \
  --out_dir examples/supply_chain/configs/generated/lv2_normal_some_bottleneck \
  --preview_dir examples/supply_chain/configs/generated/lv2_normal_some_bottleneck \
  --preview \
  --stem_base case \
  --depth_range 2,3 --width_range 2,2 \
  --no-allow_cross_edges --shortcut_fraction 0.0 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --cost_shortcut_mode premium

# normal-promo-spike (Lv2, 並列経路あり)
uv run python examples/supply_chain/case_generator.py \
  --level 2 --num_cases 30 --seed 202 \
  --out_dir examples/supply_chain/configs/generated/lv2_normal_promo_spike \
  --preview_dir examples/supply_chain/configs/generated/lv2_normal_promo_spike \
  --preview \
  --stem_base case \
  --depth_range 2,4 --width_range 2,3 \
  --allow_cross_edges \
  --shortcut_fraction 0.3 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode mixed

# hard-multi-echelon (Lv4, 多段＋クロスエッジ)
uv run python examples/supply_chain/case_generator.py \
  --level 4 --num_cases 30 --seed 301 \
  --out_dir examples/supply_chain/configs/generated/lv4_hard_multi_echelon \
  --preview_dir examples/supply_chain/configs/generated/lv4_hard_multi_echelon \
  --preview \
  --stem_base case \
  --depth_range 4,5 --width_range 2,3 \
  --allow_cross_edges \
  --shortcut_fraction 0.2 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode premium

# hard-black-friday (Lv3, 長めホライズン想定)
uv run python examples/supply_chain/case_generator.py \
  --level 3 --num_cases 30 --seed 302 \
  --out_dir examples/supply_chain/configs/generated/lv3_hard_black_friday \
  --preview_dir examples/supply_chain/configs/generated/lv3_hard_black_friday \
  --preview \
  --stem_base case \
  --depth_range 3,4 --width_range 2,3 \
  --allow_cross_edges \
  --shortcut_fraction 0.3 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode mixed \
  --size_factor 1.2

# hard-multi-shortage (Lv4, リスク高め)
uv run python examples/supply_chain/case_generator.py \
  --level 4 --num_cases 30 --seed 303 \
  --out_dir examples/supply_chain/configs/generated/lv4_hard_multi_shortage \
  --preview_dir examples/supply_chain/configs/generated/lv4_hard_multi_shortage \
  --preview \
  --stem_base case \
  --depth_range 3,5 --width_range 2,4 \
  --allow_cross_edges \
  --shortcut_fraction 0.25 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode premium

# hard-demand-whiplash (Lv5, レジーム変化あり)
uv run python examples/supply_chain/case_generator.py \
  --level 5 --num_cases 30 --seed 304 \
  --out_dir examples/supply_chain/configs/generated/lv5_hard_demand_whiplash \
  --preview_dir examples/supply_chain/configs/generated/lv5_hard_demand_whiplash \
  --preview \
  --stem_base case \
  --depth_range 4,5 --width_range 2,4 \
  --allow_cross_edges \
  --shortcut_fraction 0.2 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode mixed

# hard-out-of-phase (Lv3, 逆位相を想定)
uv run python examples/supply_chain/case_generator.py \
  --level 3 --num_cases 30 --seed 305 \
  --out_dir examples/supply_chain/configs/generated/lv3_hard_out_of_phase \
  --preview_dir examples/supply_chain/configs/generated/lv3_hard_out_of_phase \
  --preview \
  --stem_base case \
  --depth_range 3,4 --width_range 2,3 \
  --allow_cross_edges \
  --shortcut_fraction 0.2 \
  --allow_demand_edges \
  --demand_edge_fraction 0.2 \
  --allow_multi_edges \
  --multi_edge_prob 0.2 \
  --cost_shortcut_mode premium
