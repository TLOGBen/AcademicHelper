# 助手用：研究流程與工具選擇

每次沿用已知題目、證據、筆記與決策，先讀對應 skill。技能文件提供工作流程，MCP 提供可呼叫工具，Python helpers 處理特定檔案與 API；三者不能混為同一個自動完成研究的服務。

## 依研究需要選流程

| 研究者需要 | 正式 skill | 主要交付 |
| --- | --- | --- |
| 多篇文獻比較與統整 Excel | [lit-comparison](../skills/lit-comparison/SKILL.md) | 五面向比較、交叉缺口、研究啟示與來源重點 |
| 理解題目與文獻 | [research](../skills/research/SKILL.md) | 研究重點、先讀文獻、來源與證據表 |
| 選研究方向 | [suggest-direction](../skills/suggest-direction/SKILL.md) | 優先方向、資源取捨、可開始的草案 |
| 判斷文章用途 | [relevance](../skills/relevance/SKILL.md) | 支持什麼、可放哪裡、適用與不確定性 |
| 評估能不能做 | [can-this-work](../skills/can-this-work/SKILL.md) | 最低可行方案、招募／方法／授權條件 |
| 修論文與準備口試 | [committee-review](../skills/committee-review/SKILL.md) | 優先修改、有來源的評議與答辯練習 |
| 找文獻與閱讀清單 | [pico-literature-search](../.agents/skills/pico-literature-search/SKILL.md) | 開始閱讀入口、實際可用 PDF、Excel／CSV |
| 整理 Zotero 文庫與閱讀標記 | [zotero-library](../skills/zotero-library/SKILL.md) | 分類／標籤預覽、有來源的筆記與引用註釋 |
| 連接 Zotero | [setup-zetero](../.agents/skills/setup-zetero/SKILL.md) | 安全授權、文庫選擇、唯讀連線診斷 |

## MCP：用 client 的實際 schema 呼叫

| 工具 | 用法與界線 |
| --- | --- |
| `search_papers` | 跨來源找候選並依 DOI 去重；保留各來源失敗及查詢上限 |
| `deep_search` | 由 DOI 找相關文章；目前不逐層遍歷引用圖，不稱為完整引文追蹤 |
| `expand_topics` | 離線產生相關查詢字串；候選用語還需核對資料庫策略 |
| `find_gaps` | 形成研究缺口候選；需追加搜尋與領域證據才能確認 |
| `evaluate_paper_tool` | 提供角色 prompt、評估維度與 rubric；不會自動完成 LLM 評分 |
| `prepare_review_context` | 預取文獻並建立評審上下文；助手完成有證據的分析 |

工具 unavailable 時先確認安裝／client discovery；可用正式 skill 和其他已授權搜尋工具繼續，但說明使用了什麼與未完成什麼。不得杜撰工具回傳。委員模擬僅是修稿建議；有授權且 host 支援分工時才使用子代理，否則循序分析並說明，不把多角色寫成真人認可。

## PICO helper：由助手操作

路徑為 `.agents/skills/pico-literature-search/scripts/literature_tool.py`。以真實 Python 路徑執行，先讀 skill 引用文件，`--config` 指向當次研究 profile。明確設定工作區／輸出位置；程式預設工作區來自 `LITERATURE_WORKSPACE` 或目前工作目錄，不是自動判定的研究資料夾。

| Action | 參數及用途 |
| --- | --- |
| `plan` | `--config <profile> --out <new output>`；輸出策略，無真實搜尋 |
| `run` | 加 `--queries <profile query IDs> --max-per-query <limit> --pdf-limit <limit>`；用新輸出目錄，執行有界搜尋與可用全文取得 |
| `import` | `--out <existing output> --ris <file> --source <database> --query-id <id>`；合併 RIS，保留原資料庫搜尋式與來源紀錄 |
| `rank` | `--out <existing output> --top <count>`；更新引用與待取得全文排序，必要時用 `--quality-property` 指定可比較測量特性 |

`import`／`rank` 會重輸出成果；先保留人工 Excel 筆記，這些筆記不會自動回寫 manifest。對同次研究沿用相同 profile。Exit 2 可能表示部分查詢失敗或 Excel 未完成；讀 `summary.json`／當次輸出判斷，保留有效成果，不能刪掉結果改稱沒有文獻。

以 `開始閱讀.html` 作為交付入口。核對 manifest、CSV／XLSX、PDF 實際存在且標題相符；只把當次成功的 Excel 當目前成果。取得不到全文時保留書目與待取得狀態，不承諾所有 PDF。Cochrane／Google Scholar 使用實際可用、授權的瀏覽或匯出流程，不批次擷取網站；沒有執行的來源明確標示。

COSMIN 按測量特性與可查核證據評讀，未有證據保留待評。引用數只協助取得排序，不是品質分數；搜尋有上限或未完成來源時不能稱完整回顧。

## Zotero 與入庫

setup-zetero 只診斷環境與唯讀連線。按 [助手執行指南](../.agents/skills/setup-zetero/references/execution.md) 使用 `--check-env`、必要時 `--discover-library`、再執行預設 doctor；只有實際 library GET 成功才稱已連接。

文庫搜尋、分類／標籤整理與 AI 筆記可使用本版九個 Zotero MCP 工具，按 [Zotero MCP 指南](zotero-mcp.md) 執行；PDF 註解僅讀取，不提供寫入。DOI journalArticle 書目可預覽／匯入。

需要完整入庫／PDF 上傳時，先確認另行安裝的 `zotero-literature-import`、讀其正式指引並尊重使用者的寫入範圍。先準備查重／預覽，保留人工筆記與原始檔，依該流程處理部分失敗續跑。此 repository 不包含入庫 skill，GET 成功、書目成功與 PDF 上傳成功需分別回報。

## 成果驗收

簡單問題直接回答；長任務先給結論、成果入口與重要限制。文獻需區分題名／摘要／全文閱讀；結論附可查來源位置。保留方法、排除理由與未完成項目，更新研究入口，讓研究者可以繼續閱讀或決策，不需理解工具的內部結構。
