# Final-output symmetry validation for all 32 point groups

Each example is a reproducible synthetic Si/Ge crystal built from two general-position 
orbits of the listed representative space group. These are numerical fixtures, not claims 
of thermodynamic stability. `zeros` lists conventional Cartesian/Voigt components forced 
to zero by the representative point-group embedding. `none` means symmetry forces no 
individual component to zero (it may still impose equalities between components).
Elastic Voigt indices use 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.

| # | PG | fixture crystal | atoms | spglib | dielectric zeros | dielectric | elastic zeros | elastic |
|---:|:---:|:---:|---:|:---:|:---|:---:|:---|:---:|
| 1 | 1 | synthetic Si1Ge1 (P1) | 2 | 1 | none | PASS (0.00e+00) | none | PASS (0.00e+00) |
| 2 | -1 | synthetic Si2Ge2 (P-1) | 4 | -1 | none | PASS (0.00e+00) | none | PASS (0.00e+00) |
| 3 | 2 | synthetic Si2Ge2 (P2) | 4 | 2 | yz, xy | PASS (3.70e-10) | C14, C16, C24, C26, C34, C36, C45, C56 | PASS (7.80e-10) |
| 4 | m | synthetic Si2Ge2 (Pm) | 4 | m | yz, xy | PASS (8.23e-10) | C14, C16, C24, C26, C34, C36, C45, C56 | PASS (5.57e-10) |
| 5 | 2/m | synthetic Si4Ge4 (P2/m) | 8 | 2/m | yz, xy | PASS (3.29e-10) | C14, C16, C24, C26, C34, C36, C45, C56 | PASS (1.21e-09) |
| 6 | 222 | synthetic Si4Ge4 (P222) | 8 | 222 | yz, xz, xy | PASS (2.06e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (6.07e-10) |
| 7 | mm2 | synthetic Si4Ge4 (Pmm2) | 8 | mm2 | yz, xz, xy | PASS (1.65e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (1.16e-09) |
| 8 | mmm | synthetic Si8Ge8 (Pmmm) | 16 | mmm | yz, xz, xy | PASS (2.78e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (5.13e-10) |
| 9 | 4 | synthetic Si4Ge4 (P4) | 8 | 4 | yz, xz, xy | PASS (1.85e-10) | C14, C15, C24, C25, C34, C35, C36, C45, C46, C56 | PASS (4.05e-09) |
| 10 | -4 | synthetic Si4Ge4 (P-4) | 8 | -4 | yz, xz, xy | PASS (2.06e-10) | C14, C15, C24, C25, C34, C35, C36, C45, C46, C56 | PASS (3.51e-09) |
| 11 | 4/m | synthetic Si8Ge8 (P4/m) | 16 | 4/m | yz, xz, xy | PASS (1.23e-10) | C14, C15, C24, C25, C34, C35, C36, C45, C46, C56 | PASS (4.98e-09) |
| 12 | 422 | synthetic Si8Ge8 (P422) | 16 | 422 | yz, xz, xy | PASS (8.23e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (2.35e-10) |
| 13 | 4mm | synthetic Si8Ge8 (P4mm) | 16 | 4mm | yz, xz, xy | PASS (2.10e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (3.29e-10) |
| 14 | -42m | synthetic Si8Ge8 (P-42m) | 16 | -42m | yz, xz, xy | PASS (1.54e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (2.01e-10) |
| 15 | 4/mmm | synthetic Si16Ge16 (P4/mmm) | 32 | 4/mmm | yz, xz, xy | PASS (2.11e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (8.39e-11) |
| 16 | 3 | synthetic Si3Ge3 (P3) | 6 | 3 | yz, xz, xy | PASS (6.86e-11) | C16, C26, C34, C35, C36, C45 | PASS (1.83e-09) |
| 17 | -3 | synthetic Si6Ge6 (P-3) | 12 | -3 | yz, xz, xy | PASS (1.10e-10) | C16, C26, C34, C35, C36, C45 | PASS (7.45e-09) |
| 18 | 32 | synthetic Si6Ge6 (P312) | 12 | 32 | yz, xz, xy | PASS (2.47e-10) | C14, C16, C24, C26, C34, C35, C36, C45, C56 | PASS (1.77e-09) |
| 19 | 3m | synthetic Si6Ge6 (P3m1) | 12 | 3m | yz, xz, xy | PASS (5.49e-11) | C15, C16, C25, C26, C34, C35, C36, C45, C46 | PASS (3.73e-09) |
| 20 | -3m | synthetic Si12Ge12 (P-31m) | 24 | -3m | yz, xz, xy | PASS (2.16e-10) | C14, C16, C24, C26, C34, C35, C36, C45, C56 | PASS (2.90e-09) |
| 21 | 6 | synthetic Si6Ge6 (P6) | 12 | 6 | yz, xz, xy | PASS (6.45e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (7.62e-10) |
| 22 | -6 | synthetic Si6Ge6 (P-6) | 12 | -6 | yz, xz, xy | PASS (2.37e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (1.01e-10) |
| 23 | 6/m | synthetic Si12Ge12 (P6/m) | 24 | 6/m | yz, xz, xy | PASS (5.42e-10) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (2.10e-10) |
| 24 | 622 | synthetic Si12Ge12 (P622) | 24 | 622 | yz, xz, xy | PASS (8.23e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (8.38e-11) |
| 25 | 6mm | synthetic Si12Ge12 (P6mm) | 24 | 6mm | yz, xz, xy | PASS (8.75e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (4.75e-10) |
| 26 | -6m2 | synthetic Si12Ge12 (P-6m2) | 24 | -6m2 | yz, xz, xy | PASS (8.23e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (1.51e-10) |
| 27 | 6/mmm | synthetic Si24Ge24 (P6/mmm) | 48 | 6/mmm | yz, xz, xy | PASS (9.48e-11) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (2.88e-10) |
| 28 | 23 | synthetic Si12Ge12 (P23) | 24 | 23 | yz, xz, xy | PASS (8.04e-13) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (5.23e-11) |
| 29 | m-3 | synthetic Si24Ge24 (Pm-3) | 48 | m-3 | yz, xz, xy | PASS (3.43e-12) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (4.25e-11) |
| 30 | 432 | synthetic Si24Ge24 (P432) | 48 | 432 | yz, xz, xy | PASS (9.43e-12) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (5.83e-12) |
| 31 | -43m | synthetic Si24Ge24 (P-43m) | 48 | -43m | yz, xz, xy | PASS (2.57e-12) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (1.51e-10) |
| 32 | m-3m | synthetic Si48Ge48 (Pm-3m) | 96 | m-3m | yz, xz, xy | PASS (1.71e-12) | C14, C15, C16, C24, C25, C26, C34, C35, C36, C45, C46, C56 | PASS (1.47e-10) |

The number in parentheses is maximum forced-zero leakage divided by `max(1, max_abs_output)`. Full group-invariance residuals are retained in the JSON report.
