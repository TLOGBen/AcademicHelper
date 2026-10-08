# 搜尋設計與平台查核

查核基準：2026-10-06。網址是官方來源入口；遇新語法、限制或 API 變動仍須查當次文件。

## PubMed／NCBI

[PubMed User Guide](https://pubmed.ncbi.nlm.nih.gov/help/)：MeSH + title/abstract 自由詞捕捉已索引和新文章。同義詞 OR、概念 AND，以括號固定群組。`[tiab]`、`[Mesh]`、`[aid]`、`[uid]` 各有用途。引號或欄位標籤改變 automatic term mapping；phrase 不在索引可能拆詞，保留實際 query translation 及警告。Wildcard 前至少四個字元；現已支援 proximity，不沿用舊文獻「無 proximity」的描述。

名稱搜尋不附 population、translation、community、Chinese、free full text、RCT 或近五年限制。某些使用研究只在正文提工具名，因此須引文及相鄰構念補查。年代與語言限制依納入條件；是否免費只影響取得。

[E-utilities 官方說明](https://www.nlm.nih.gov/dataguide/eutilities/how_eutilities_works.html)、[用量及 key](https://eutilities.github.io/site/API_Key/usageandkey/)：`db=pubmed` 是 PubMed；ESearch 得 IDs，EFetch 得書目／摘要 XML。無 key 每 IP 3 requests/s，有 key 預設 10/s，仍共享 IP 流量。key 用環境變數，不放紀錄；提供 email／tool 參數，不代使用者寄信登記。

Helper 一輪上限 9999 IDs。更多命中需拆可追溯日期區間等再跑去重，或用官方 EDirect；保留每個分段查詢。試跑可低上限，不稱截取清單為完整檢索。

## Cochrane Library

[官方技術補充文件](https://training.cochrane.org/chapter04-tech-supplonlinepdfv65270924)，第 84 頁：Wiley Cochrane 有 `:ti,ab,kw`、`NEAR`、`*` 和 MeSH。到 [Search Manager](https://www.cochranelibrary.com/advanced-search/search-manager) 分行輸入，以 `#1` 引用集合；使用 `NEAR/n` 先查當次說明，不把 PubMed `[tiab]` 當 Cochrane 欄位。

翻譯驗證常是觀察性研究，Cochrane 補找回顧／介入背景。CDSR、CENTRAL 分開紀錄；未找到量表不證明量表不存在。不可假定有公開 REST API 可下載全文或把 CENTRAL 列當 PDF。用官方網頁／授權介面匯出 RIS。

## Google Scholar

[官方說明](https://scholar.google.com/intl/en/scholar/help.html)：原題名引號、多次短查詢、引用追蹤、Related articles、All versions。Cite 支援 RefMan 等；官方不提供 bulk access，單查詢最多查看 1000 結果。總數不等於可取得文獻數，不繞過身份或驗證碼。

版本分跑縮寫、全名、translation、validation、Chinese、Taiwan 及中日文名稱。若只做 web search，標為網頁線索；重要文獻回原期刊核對題名、年、DOI。

## 測量研究／COSMIN

[COSMIN 手冊 V2](https://www.cosmin.nl/wp-content/uploads/COSMIN-manual-V2_final.pdf)、[PubMed filters](https://www.cosmin.nl/tools/pubmed-search-filters/)：構念、族群、工具、測量特性構成設計，已指定量表保留名稱搜尋。簡短的 `valid* OR reliab* OR translat* ...` 是自編詞，不命名為 validated COSMIN filter；採官方完整 filter 須記版。敏感 filter 大量不相關，不以結果少判更好；轉別平台後不繼承已驗證效能。

維持 broad 名稱層，再分開做 methods、population、context、language／country。語言版本與出版語言分開，臺灣繁體版、簡體漢化、英文發表文章、日文施測母版不是同一屬性。
