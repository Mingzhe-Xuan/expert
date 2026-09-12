# Source manifest

获取/审计日期：2026-09-12。

| Local file | Primary source URL | SHA-256 | Status / usage |
|---|---|---|---|
| [GMTNet.pdf](../ref/GMTNet.pdf) | https://proceedings.mlr.press/v235/yan24a.html | `C691E2538CD24E1857D4AFD41D63DBA50FD3337E157BF3B5612D9BE06F161363` | 已有；JARVIS tensor canonical result tables |
| [ETGNN.pdf](../ref/ETGNN.pdf) | https://doi.org/10.1021/acs.jpclett.3c01200 | `E8D68B95DB2789C443A53B000CD42A6AFB553BA379A3DB12B6C26260EC087638` | 已有；JARVIS-DFPT full BEC result |
| [GoeCTP_into_equivariant.pdf](../ref/GoeCTP_into_equivariant.pdf) | https://arxiv.org/abs/2410.02372 | `12EB667346D6D654829D231B0E8500D86518B4BB84849AC45B1D2C59480A3ABC` | 已有；统一重跑 tables |
| [PGEqNN.pdf](../ref/PGEqNN.pdf) | https://arxiv.org/abs/2607.16871 | `62F9C2D07F86326B7639E45F08430C7E448AB1383B20FCFB890447D2E19F406D` | 已有；不同 MP/PG benchmark，仅作背景 |
| [MatTen_main.pdf](sources/MatTen_main.pdf) | https://arxiv.org/pdf/2307.15242 | `8D4EB5A6E6CC3869BC1371FCFE8330C3AB36A0E3AA82F2DBF0A2A56C4B98C128` | 新下载；PDF 可由 Poppler 读取 |
| [CEITNet.pdf](sources/CEITNet.pdf) | https://arxiv.org/pdf/2602.04323 | `817B9AEE1261CDD23FE23B3D65D3A9023377DF11E32A12904B3752A2AF72623A` | 新下载；14 pages，PDF 可由 Poppler 读取 |
| [JARVIS_DFPT_2020.pdf](sources/JARVIS_DFPT_2020.pdf) | https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=928845 | `F3B8BF3887A78519AE08330F424E4831DB0B1B464D21F5BB7B69BB6B5469191B` | 新下载；PDF 可由 Poppler 读取 |
| [ALIGNN_README_snapshot.md](sources/ALIGNN_README_snapshot.md) | https://github.com/atomgptlab/alignn/blob/main/README.md | `A1270D26BC086EC91508B94E98152C0C7360F7098C053DB2BA951D48AFCECC77` | 新下载；会随官方仓库更新，故保存本地快照 |

## 未能归档的公开来源

| Source | URL | Reason |
|---|---|---|
| IrredNet OpenReview submission | https://openreview.net/forum?id=65s5CctH7I | 公开 PDF endpoint 返回 HTTP 403 anti-bot；结果已从 OpenReview 索引的 Table 3 交叉核对并在主表标 B |
| MatTen ESI | https://www.rsc.org/suppdata/d3/dd/d3dd00233k/d3dd00233k1.pdf | 原公开 supplement URL 当前返回 HTTP 404；主文和 RSC 索引仍可访问 |

未归档文件不以空文件或 HTML challenge page 代替。若后续端点恢复，应下载后补充
checksum，并重新检查其中的数据 split 与完整 tensor metric。
