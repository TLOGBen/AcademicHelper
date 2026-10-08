# AcademicHelper

AcademicHelper 是學術研究輔助工具，提供文獻搜尋、研究缺口候選與論文評審上下文。Python 服務使用 MCP stdio；`agents/` 與 `skills/` 另提供評審角色與操作指引。MCP 工具本身不呼叫 LLM。

## 開發環境

建議使用 Python 3.12 與 [uv](https://docs.astral.sh/uv/)。從 repository 根目錄執行；Cloud task 已隔離，直接使用現有 checkout，除非明確需要，不另建 Git worktree。

```sh
uv sync --python 3.12 --group dev
uv run python -m pytest -q
```

程式使用 MCP 1.x 的 FastMCP 介面，依賴已限制為 `mcp>=1.30.0,<2`，避免解析至不相容的 MCP 2.x。開發依賴由 `dev` group 安裝；測試使用 mock，不需要外部 API key。Cloud 的已準備環境另保留經雜湊驗證的依賴清單，可使用 `--no-sync` 避免重新解析。

## 啟動 MCP

```sh
uv run academic-helper
```

這是 stdio 服務，需由 MCP client 保持 stdin/stdout 連線，沒有網頁、資料庫、監聽 port 或 localhost preview。Claude plugin 的 `.mcp.json` 使用 `uv run --directory ${CLAUDE_PLUGIN_ROOT} academic-helper`，可從不同工作目錄啟動；其他 MCP client 請將變數換成實際 repository 絕對路徑，或直接執行已安裝的 `academic-helper`。

| MCP 工具 | 用途 |
| --- | --- |
| `search_papers` | 跨來源搜尋並依 DOI 去重 |
| `deep_search` | 以 DOI 搜尋相關論文；目前實作尚未逐層遍歷引用圖 |
| `expand_topics` | 產生相關查詢字串，不使用網路 |
| `find_gaps` | 依分析模式建立研究缺口候選 |
| `evaluate_paper_tool` | 建立委員、完整角色 prompt、評估維度與 rubric，不自動評分 |
| `prepare_review_context` | 預取文獻並建立含待評文章的評審上下文 |

啟動驗證應完成 MCP initialize、列出上述工具，並成功呼叫 `expand_topics`。Cloud 設定時已驗證這個流程。

八份 `agents/*.md` 是角色設定的正式來源：YAML 提供 focus／評估維度，Markdown 本文作為 prompt。checkout 載入與工作目錄無關，wheel 也包含相同角色。`committee-review` 使用工具回傳的 prompt；有子代理時獨立分工，無子代理時循序分析並標明流程。評審結果是模擬建議，不代表真實委員認可或標準化品質評級。

## Skills 與安裝

Claude Code 可透過本 repository 的 marketplace 安裝 `academic-helper` plugin，或在本機 checkout 使用 `claude --plugin-dir /absolute/path/to/AcademicHelper`。需先安裝 uv；MCP 會使用 plugin 目錄解析 Python 依賴。manifest 保留預設 `skills/`，另載入 `.agents/skills/` 中的兩個 skills，不複製另一套來源。

| Skill | 用途 |
| --- | --- |
| `research` | 可追溯的文獻搜尋、評讀與綜合 |
| `suggest-direction` | 依證據與資源形成研究方向候選 |
| `relevance` | 判斷構念、族群、版本與用途的適配 |
| `can-this-work` | 研究方法、招募、權限、資源與時程的可行性 |
| `committee-review` | 有證據與限制的多角色口試模擬 |
| `pico-literature-search` | 資料庫策略、PMC 全文與閱讀清單 |
| `setup-zetero` | Zotero 環境變數與唯讀連線診斷 |

Codex Cloud 自動使用 `.agents/skills/` 中的 repository skills；五個 `skills/` 中的 Claude skills 需由相應 host 載入或依同一 commit 安裝至 Codex skill 目錄。名稱相同的本機版本先比對，保留未提交修改。

既有研究 skills 已移除固定年代與強制數字排名，按當下日期設定搜尋範圍，區分題名／摘要／全文、有限搜尋與完整回顧、缺口候選與驗證。COSMIN 按適用的測量特性評讀；引用數與期刊聲望不能替代品質。工具版本及最新規範須從實際來源確認。

## 外部文獻來源

即時搜尋需要 HTTPS 存取 `api.openalex.org`、`api.semanticscholar.org`、`api.crossref.org`。`SEMANTIC_SCHOLAR_API_KEY` 在目前程式中為選用；只透過環境變數提供。需要自訂來源時，可使用 `OPENALEX_BASE_URL`、`SEMANTIC_SCHOLAR_BASE_URL`、`CROSSREF_BASE_URL`。

本次 Cloud 的三個來源請求皆收到 proxy CONNECT 403，即時搜尋尚未驗證成功。請在 Cloud 環境設定增補所需網域，保留既有白名單；設定實際生效後再測試。儲存待審設定、目前機器的執行狀態及發布環境是不同步驟，儲存 draft 不會立即變更網路或完成發布。

## PICO 文獻搜尋 skill

[.agents/skills/pico-literature-search/SKILL.md](.agents/skills/pico-literature-search/SKILL.md) 提供 PICO／量表研究的搜尋策略、PubMed 書目取回、PMC 開放全文 PDF、RIS 匯入、閱讀清單與待取得全文排序。Cochrane／Google Scholar 使用各平台搜尋與匯出，不執行批次網頁擷取。引用次數不代表研究品質；COSMIN 評級需要對應測量特性的可靠證據。

Python helper 只需 standard library。Excel 匯出另需 Node 與 `@oai/artifact-tool`；在 Cloud 使用實際 runtime 路徑設定 `CODEX_NODE`／`CODEX_NODE_MODULES`。即時搜尋與 PDF 需要 `eutils.ncbi.nlm.nih.gov`、`pmc-oa-opendata.s3.amazonaws.com`；引用查核另用 `api.openalex.org`。

```sh
uv run --no-sync python -m unittest discover -s .agents/skills/pico-literature-search/scripts -p test_literature_tool.py -v
uv run --no-sync python .agents/skills/pico-literature-search/scripts/literature_tool.py plan --out outputs/plan_check
```

詳見 [Cloud 安裝與驗證](docs/codex-cloud-literature.md)。離線 plan／模擬匯出不等於已執行資料庫搜尋。rank 會核對 PDF 是否實際可讀且檔頭有效；搬移或刪除後失效的路徑會保留為 `pdf_missing_path`，文章回到待取得清單。manifest 的相對 PDF 路徑依輸出目錄解析。

## Zotero 環境設定 skill

本 repository 的正式來源是 [.agents/skills/setup-zetero/SKILL.md](.agents/skills/setup-zetero/SKILL.md)。skill 名稱保留為 `setup-zetero`，服務名稱為 Zotero。helper 僅使用 Python standard library，檢查以下變數：

| 變數 | 設定方式 |
| --- | --- |
| `ZOTERO_API_KEY` | Secret；只從環境讀取，不存入 repository、腳本、skill 或聊天 |
| `ZOTERO_LIBRARY_ID` | 非敏感設定；填入實際 library 的正整數 ID |
| `ZOTERO_LIBRARY_TYPE` | `user` 或 `group`，未設定時預設 `user` |

Cloud 請使用環境設定的安全輸入介面提供 key，並設定 ID/type；本機可由可信的 secret 管理工具注入程序環境。Windows 也可透過「使用者環境變數」介面設定：helper 優先採用目前程序值，僅在變數不存在時讀取 HKCU 的 User 環境值。Linux／Cloud 只讀程序環境；Windows User 設定不會自動同步至 Cloud。不要輸出 key 或把個人設定檔一併複製。

```sh
# 僅檢查變數存在與格式，不發送請求
uv run --no-sync python .agents/skills/setup-zetero/scripts/zotero_doctor.py --check-env
# 預設：對官方 api.zotero.org 執行唯讀 GET
uv run --no-sync python .agents/skills/setup-zetero/scripts/zotero_doctor.py
```

預設檢查會驗證 key metadata 與所選 library 的讀取存取，保留 HTTPS 驗證，不建立或修改任何書目、PDF 或其他資料。Cloud 需允許 `api.zotero.org`；proxy CONNECT 403 表示網路規則阻擋，不能據此判定 key 失效。變數存在、模擬測試通過或 GET 成功都不代表已完成文獻匯入。

helper 的獨立測試可跨平台執行，不需憑證或網路：

```sh
uv run --no-sync python -m unittest discover -s .agents/skills/setup-zetero/scripts -p test_zotero_doctor.py -v
```

若需要安裝至本機 User scope，從同一個已確認的 branch／commit 複製完整 `setup-zetero` 資料夾至 `~/.codex/skills/setup-zetero`。先比對已存在的版本並保留本機修改與其他 skills；不要附帶憑證、個人 library 值或設定檔。

此 skill 負責環境設定與診斷。文獻匯入、PDF 入庫、評讀及 Excel 匯出由另一個聊天維護的 `zotero-literature-import` 工作流程處理；該 skill 尚未安裝於此 checkout，也尚未在這個 Cloud 驗證 Zotero 認證或匯入。

## 0.2.0 驗證範圍

2026-10-08 的更新通過 400 項主程式測試、20 項 Zotero 與 15 項 PICO 離線測試。實際 MCP initialize／列出工具／主題擴展／評審上下文呼叫成功；獨立安裝 wheel 後，也能從專案外載入八個角色並執行評審工具。Excel 曾以明確標示的模擬資料驗證三個工作表與預覽。這些驗證不包含真實資料庫搜尋、Zotero 認證或 PDF 入庫。
