# Resumen de perfiles (ncu y nsys)

| Aplicación | Size | Bloque × grid | Kernel (ms, ncu) | Ocupación lograda / teórica | Limitante de ocupación | SM | DRAM | L2 | L1/TEX | Eficiencia de warp | Eficiencia de saltos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| apsp_floyd | 3,741 | (32, 32, 1) × (117, 117, 1) | 1.20 | 65 % / 67 % | Shared Mem, Warps | 83 % | 48 % | 10 % | 89 % | 100 % | — (0 saltos divergentes) |
| eta_batch_soa | 5,000 | (256, 1, 1) × (20, 600, 1) | 0.757 | 85 % / 100 % | Warps | 5 % | 37 % | 25 % | 21 % | 100 % | 100 % |
| eta_batch_aos | 5,000 | (256, 1, 1) × (20, 600, 1) | 0.760 | 84 % / 100 % | Warps | 5 % | 37 % | 30 % | 25 % | 100 % | — (0 saltos divergentes) |
| demand_heatmap | 2,048 | (16, 16, 1) × (128, 128, 1) | 0.743 | 82 % / 100 % | Warps | 82 % | 23 % | 8 % | 47 % | 100 % | 100 % |
| demand_heatmap_tiles | 2,048 | (16, 16, 1) × (128, 128, 1) | 0.477 | 95 % / 100 % | Registers, Warps | 91 % | 36 % | 11 % | 75 % | 99 % | 100 % |
| coverage_field | 1,024 | (16, 16, 1) × (64, 64, 1) | 0.0693 | 84 % / 100 % | Warps | 20 % | 93 % | 30 % | 33 % | 94 % | 55 % |
| delivery_histogram | 10,000,000 | (256, 1, 1) × (120, 1, 1) | 0.219 | 99 % / 100 % | Warps | 16 % | 96 % | 25 % | 14 % | 100 % | 100 % |
| histogram_naive | 10,000,000 | (256, 1, 1) × (39063, 1, 1) | 2.87 | 81 % / 100 % | Warps | 2 % | 9 % | 9 % | 10 % | 100 % | — (0 saltos divergentes) |
| best_courier | 5,000 | (256, 1, 1) × (600, 1, 1) | 0.0720 | 97 % / 100 % | Warps | 21 % | 89 % | 23 % | 22 % | 98 % | 97 % |
| best_courier_divergent | 5,000 | (256, 1, 1) × (600, 1, 1) | 0.0717 | 97 % / 100 % | Warps | 21 % | 88 % | 23 % | 22 % | 98 % | 97 % |
| orders_in_progress | 10,000,000 | (512, 1, 1) × (19532, 1, 1) | 0.565 | 92 % / 100 % | Warps | 86 % | 73 % | 12 % | 86 % | 100 % | 100 % |
| candidate_filter | 200,000 | (256, 1, 1) × (782, 1, 1) | 0.0185 | 92 % / 100 % | Warps | 19 % | 36 % | 31 % | 34 % | 69 % | 47 % |
| candidate_filter_naive | 200,000 | (256, 1, 1) × (782, 1, 1) | 0.0179 | 89 % / 100 % | Warps | 15 % | 39 % | 34 % | 38 % | 59 % | — (0 saltos divergentes) |
| delivery_timeline | 10,000,000 | (256, 1, 1) × (9766, 1, 1) | 1.24 | 90 % / 100 % | Registers, Warps | 78 % | 67 % | 72 % | 78 % | 70 % | 67 % |
| delivery_percentiles | 10,000,000 | (256, 1, 1) × (39063, 1, 1) | 0.865 | 93 % / 100 % | Warps | 10 % | 96 % | 19 % | 15 % | 100 % | — (0 saltos divergentes) |

## apsp_floyd

- **ncu** (`floydWarshallPhase3`, size 3,741, bloque (32, 32, 1), grid (117, 117, 1), 30 registros por hilo): ocupación lograda **65 %** de 67 % teórica, limitada por **Shared Mem, Warps** (bloques por SM: registros 2, shared 1, warps 1). Throughput: SM 83 %, memoria 83 % (DRAM 48 %, L2 10 %, L1/TEX 89 %; 91.8 GB/s; acierto en L2 68 %). Eficiencia de warp 100 % de hilos activos (99 % sin predicar), eficiencia de saltos — (0 saltos divergentes), 24.0 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 2 `cudaMemcpy` = 80.3 ms; 351 lanzamientos = 72.6 ms; 4 esperas (`*Synchronize`) = 260 ms; reservas y liberaciones = 1.52 ms.
- Archivos versionados: `analysis/data/final/perfiles/apsp_floyd_ncu_details.csv`, `apsp_floyd_ncu.out`, `apsp_floyd_nsys.out` y `apsp_floyd_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## eta_batch_soa

- **ncu** (`computeEtaSoA`, size 5,000, bloque (256, 1, 1), grid (20, 600, 1), 20 registros por hilo): ocupación lograda **85 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 10, shared 16, warps 6). Throughput: SM 5 %, memoria 37 % (DRAM 37 %, L2 25 %, L1/TEX 21 %; 70.5 GB/s; acierto en L2 67 %). Eficiencia de warp 100 % de hilos activos (97 % sin predicar), eficiencia de saltos 100 %, 192 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 13 `cudaMemcpy` = 23.1 ms; 1 lanzamientos = 82.6 ms; 6 esperas (`*Synchronize`) = 1.49 ms; reservas y liberaciones = 3.38 ms.
- Archivos versionados: `analysis/data/final/perfiles/eta_batch_soa_ncu_details.csv`, `eta_batch_soa_ncu.out`, `eta_batch_soa_nsys.out` y `eta_batch_soa_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## eta_batch_aos

- **ncu** (`computeEtaAoS`, size 5,000, bloque (256, 1, 1), grid (20, 600, 1), 20 registros por hilo): ocupación lograda **84 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 10, shared 16, warps 6). Throughput: SM 5 %, memoria 37 % (DRAM 37 %, L2 30 %, L1/TEX 25 %; 70.5 GB/s; acierto en L2 84 %). Eficiencia de warp 100 % de hilos activos (98 % sin predicar), eficiencia de saltos — (0 saltos divergentes), 201 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 11 `cudaMemcpy` = 23.1 ms; 1 lanzamientos = 3.98 ms; 6 esperas (`*Synchronize`) = 1.55 ms; reservas y liberaciones = 4.04 ms.
- Archivos versionados: `analysis/data/final/perfiles/eta_batch_aos_ncu_details.csv`, `eta_batch_aos_ncu.out`, `eta_batch_aos_nsys.out` y `eta_batch_aos_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## demand_heatmap

- **ncu** (`smoothDemandKernel`, size 2,048, bloque (16, 16, 1), grid (128, 128, 1), 32 registros por hilo): ocupación lograda **82 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 8, shared 16, warps 6). Throughput: SM 82 %, memoria 46 % (DRAM 23 %, L2 8 %, L1/TEX 47 %; 44.8 GB/s; acierto en L2 72 %). Eficiencia de warp 100 % de hilos activos (92 % sin predicar), eficiencia de saltos 100 %, 12.0 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 4 `cudaMemcpy` = 9.30 ms; 1 lanzamientos = 1.19 ms; 4 esperas (`*Synchronize`) = 1.51 ms; reservas y liberaciones = 124 ms.
- Archivos versionados: `analysis/data/final/perfiles/demand_heatmap_ncu_details.csv`, `demand_heatmap_ncu.out`, `demand_heatmap_nsys.out` y `demand_heatmap_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## demand_heatmap_tiles

- **ncu** (`smoothDemandTiledKernel`, size 2,048, bloque (16, 16, 1), grid (128, 128, 1), 40 registros por hilo): ocupación lograda **95 %** de 100 % teórica, limitada por **Registers, Warps** (bloques por SM: registros 6, shared 21, warps 6). Throughput: SM 91 %, memoria 74 % (DRAM 36 %, L2 11 %, L1/TEX 75 %; 69.7 GB/s; acierto en L2 73 %). Eficiencia de warp 99 % de hilos activos (91 % sin predicar), eficiencia de saltos 100 %, 12.5 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 4 `cudaMemcpy` = 7.16 ms; 1 lanzamientos = 0.626 ms; 4 esperas (`*Synchronize`) = 0.917 ms; reservas y liberaciones = 150 ms.
- Archivos versionados: `analysis/data/final/perfiles/demand_heatmap_tiles_ncu_details.csv`, `demand_heatmap_tiles_ncu.out`, `demand_heatmap_tiles_nsys.out` y `demand_heatmap_tiles_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## coverage_field

- **ncu** (`coverageJacobiStep`, size 1,024, bloque (16, 16, 1), grid (64, 64, 1), 16 registros por hilo): ocupación lograda **84 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 16, warps 6). Throughput: SM 20 %, memoria 93 % (DRAM 93 %, L2 30 %, L1/TEX 33 %; 178 GB/s; acierto en L2 54 %). Eficiencia de warp 94 % de hilos activos (89 % sin predicar), eficiencia de saltos 55 %, 63.0 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 4 `cudaMemcpy` = 2.84 ms; 200 lanzamientos = 2.68 ms; 4 esperas (`*Synchronize`) = 4.63 ms; reservas y liberaciones = 130 ms.
- Archivos versionados: `analysis/data/final/perfiles/coverage_field_ncu_details.csv`, `coverage_field_ncu.out`, `coverage_field_nsys.out` y `coverage_field_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## delivery_histogram

- **ncu** (`histogramPrivatizedKernel`, size 10,000,000, bloque (256, 1, 1), grid (120, 1, 1), 16 registros por hilo): ocupación lograda **99 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 12, warps 6). Throughput: SM 16 %, memoria 96 % (DRAM 96 %, L2 25 %, L1/TEX 14 %; 184 GB/s; acierto en L2 0 %). Eficiencia de warp 100 % de hilos activos (100 % sin predicar), eficiencia de saltos 100 %, 79.5 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 3 `cudaMemcpy` = 9.10 ms; 1 lanzamientos = 4.91 ms; 4 esperas (`*Synchronize`) = 2.76 ms; reservas y liberaciones = 1.45 ms.
- Archivos versionados: `analysis/data/final/perfiles/delivery_histogram_ncu_details.csv`, `delivery_histogram_ncu.out`, `delivery_histogram_nsys.out` y `delivery_histogram_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## histogram_naive

- **ncu** (`histogramGlobalAtomicsKernel`, size 10,000,000, bloque (256, 1, 1), grid (39063, 1, 1), 16 registros por hilo): ocupación lograda **81 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 16, warps 6). Throughput: SM 2 %, memoria 9 % (DRAM 9 %, L2 9 %, L1/TEX 10 %; 18.0 GB/s; acierto en L2 76 %). Eficiencia de warp 100 % de hilos activos (95 % sin predicar), eficiencia de saltos — (0 saltos divergentes), 671 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 3 `cudaMemcpy` = 7.68 ms; 1 lanzamientos = 0.987 ms; 4 esperas (`*Synchronize`) = 3.32 ms; reservas y liberaciones = 1.26 ms.
- Archivos versionados: `analysis/data/final/perfiles/histogram_naive_ncu_details.csv`, `histogram_naive_ncu.out`, `histogram_naive_nsys.out` y `histogram_naive_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## best_courier

- **ncu** (`bestCourierConvergent`, size 5,000, bloque (256, 1, 1), grid (600, 1, 1), 24 registros por hilo): ocupación lograda **97 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 10, shared 21, warps 6). Throughput: SM 21 %, memoria 89 % (DRAM 89 %, L2 23 %, L1/TEX 22 %; 170 GB/s; acierto en L2 17 %). Eficiencia de warp 98 % de hilos activos (82 % sin predicar), eficiencia de saltos 97 %, 86.9 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 5 `cudaMemcpy` = 3.61 ms; 1 lanzamientos = 3.17 ms; 5 esperas (`*Synchronize`) = 0.996 ms; reservas y liberaciones = 5.59 ms.
- Archivos versionados: `analysis/data/final/perfiles/best_courier_ncu_details.csv`, `best_courier_ncu.out`, `best_courier_nsys.out` y `best_courier_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## best_courier_divergent

- **ncu** (`bestCourierDivergent`, size 5,000, bloque (256, 1, 1), grid (600, 1, 1), 24 registros por hilo): ocupación lograda **97 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 10, shared 21, warps 6). Throughput: SM 21 %, memoria 88 % (DRAM 88 %, L2 23 %, L1/TEX 22 %; 169 GB/s; acierto en L2 17 %). Eficiencia de warp 98 % de hilos activos (82 % sin predicar), eficiencia de saltos 97 %, 86.5 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 5 `cudaMemcpy` = 3.53 ms; 1 lanzamientos = 0.788 ms; 5 esperas (`*Synchronize`) = 0.530 ms; reservas y liberaciones = 1.31 ms.
- Archivos versionados: `analysis/data/final/perfiles/best_courier_divergent_ncu_details.csv`, `best_courier_divergent_ncu.out`, `best_courier_divergent_nsys.out` y `best_courier_divergent_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## orders_in_progress

- **ncu** (`scanBlockKoggeStone`, size 10,000,000, bloque (512, 1, 1), grid (19532, 1, 1), 16 registros por hilo): ocupación lograda **92 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 8, shared 6, warps 3). Throughput: SM 86 %, memoria 86 % (DRAM 73 %, L2 12 %, L1/TEX 86 %; 140 GB/s; acierto en L2 50 %). Eficiencia de warp 100 % de hilos activos (96 % sin predicar), eficiencia de saltos 100 %, 21.8 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 4 `cudaMemcpy` = 14.1 ms; 6 lanzamientos = 1.32 ms; 4 esperas (`*Synchronize`) = 1.84 ms; reservas y liberaciones = 117 ms.
- Archivos versionados: `analysis/data/final/perfiles/orders_in_progress_ncu_details.csv`, `orders_in_progress_ncu.out`, `orders_in_progress_nsys.out` y `orders_in_progress_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## candidate_filter

- **ncu** (`filterCandidatesBlockPrivate`, size 200,000, bloque (256, 1, 1), grid (782, 1, 1), 16 registros por hilo): ocupación lograda **92 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 15, warps 6). Throughput: SM 19 %, memoria 36 % (DRAM 36 %, L2 31 %, L1/TEX 34 %; 69.8 GB/s; acierto en L2 77 %). Eficiencia de warp 69 % de hilos activos (64 % sin predicar), eficiencia de saltos 47 %, 81.5 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 8 `cudaMemcpy` = 15.0 ms; 1 lanzamientos = 3.87 ms; 5 esperas (`*Synchronize`) = 0.621 ms; reservas y liberaciones = 2.42 ms.
- Archivos versionados: `analysis/data/final/perfiles/candidate_filter_ncu_details.csv`, `candidate_filter_ncu.out`, `candidate_filter_nsys.out` y `candidate_filter_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## candidate_filter_naive

- **ncu** (`filterCandidatesNaive`, size 200,000, bloque (256, 1, 1), grid (782, 1, 1), 16 registros por hilo): ocupación lograda **89 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 16, warps 6). Throughput: SM 15 %, memoria 39 % (DRAM 39 %, L2 34 %, L1/TEX 38 %; 73.7 GB/s; acierto en L2 78 %). Eficiencia de warp 59 % de hilos activos (55 % sin predicar), eficiencia de saltos — (0 saltos divergentes), 96.3 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 8 `cudaMemcpy` = 16.5 ms; 1 lanzamientos = 1.01 ms; 5 esperas (`*Synchronize`) = 0.619 ms; reservas y liberaciones = 2.45 ms.
- Archivos versionados: `analysis/data/final/perfiles/candidate_filter_naive_ncu_details.csv`, `candidate_filter_naive_ncu.out`, `candidate_filter_naive_nsys.out` y `candidate_filter_naive_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## delivery_timeline

- **ncu** (`mergeRound`, size 10,000,000, bloque (256, 1, 1), grid (9766, 1, 1), 40 registros por hilo): ocupación lograda **90 %** de 100 % teórica, limitada por **Registers, Warps** (bloques por SM: registros 6, shared 16, warps 6). Throughput: SM 78 %, memoria 72 % (DRAM 67 %, L2 72 %, L1/TEX 78 %; 129 GB/s; acierto en L2 89 %). Eficiencia de warp 70 % de hilos activos (63 % sin predicar), eficiencia de saltos 67 %, 17.5 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 6 `cudaMemcpy` = 32.3 ms; 12 lanzamientos = 4.35 ms; 4 esperas (`*Synchronize`) = 18.3 ms; reservas y liberaciones = 126 ms.
- Archivos versionados: `analysis/data/final/perfiles/delivery_timeline_ncu_details.csv`, `delivery_timeline_ncu.out`, `delivery_timeline_nsys.out` y `delivery_timeline_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.

## delivery_percentiles

- **ncu** (`radixScatter`, size 10,000,000, bloque (256, 1, 1), grid (39063, 1, 1), 16 registros por hilo): ocupación lograda **93 %** de 100 % teórica, limitada por **Warps** (bloques por SM: registros 16, shared 16, warps 6). Throughput: SM 10 %, memoria 96 % (DRAM 96 %, L2 19 %, L1/TEX 15 %; 185 GB/s; acierto en L2 30 %). Eficiencia de warp 100 % de hilos activos (97 % sin predicar), eficiencia de saltos — (0 saltos divergentes), 129 ciclos por instrucción emitida.
- **nsys** (API de CUDA del hilo de CPU, toda la corrida de `--bench-once --version gpu_e2e`): 3 `cudaMemcpy` = 12.2 ms; 175 lanzamientos = 3.94 ms; 4 esperas (`*Synchronize`) = 56.4 ms; reservas y liberaciones = 107 ms.
- Archivos versionados: `analysis/data/final/perfiles/delivery_percentiles_ncu_details.csv`, `delivery_percentiles_ncu.out`, `delivery_percentiles_nsys.out` y `delivery_percentiles_nsys_cuda_*_sum.csv`. Los binarios (`.ncu-rep`, `.nsys-rep`, `.sqlite`) no se suben: guardan el nombre y el entorno de la máquina; se regeneran con los comandos de arriba.
