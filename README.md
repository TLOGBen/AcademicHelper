# AcademicHelper

AcademicHelper 是學術研究輔助工具，提供文獻搜尋、研究缺口候選與論文評審上下文。Python 服務使用 MCP stdio；`agents/` 與 `skills/` 另提供評審角色與操作指引。MCP 工具本身不呼叫 LLM。

## 開發環境

建議使用 Python 3.12 與 [uv](https://docs.astral.sh/uv/)。從 repository 根目錄執行；Cloud task 已隔離，直接使用現有 checkout，除非明確需要，不另建 Git worktree。

```sh
uv venv --python 3.12
uv pip install -e . "mcp==1.30.0" "pytest==8.4.2" "pytest-asyncio==0.23.8"
uv run --no-sync python -m pytest -q
```

這組版本已在 Cloud 驗證。程式使用 MCP 1.x 的 FastMCP 介面；目前 `pyproject.toml` 的版本下限會讓一般 `uv sync`／`uv run` 解析至不相容的 MCP 2.x。開發與啟動請使用上述安裝方式及 `--no-sync`，保留已驗證的環境。測試使用 mock，不需要外部 API key。

## 啟動 MCP

```sh
uv run --no-sync academic-helper
```

這是 stdio 服務，需由 MCP client 保持 stdin/stdout 連線，沒有網頁、資料庫、監聽 port 或 localhost preview。client 的工作目錄需指向此 repository。

現有 `.mcp.json` 使用 `uv run academic-helper`。啟動 client 前，將 `UV_NO_SYNC` 傳入其程序環境：

```sh
# Bash
export UV_NO_SYNC=true
```

```powershell
# PowerShell
$env:UV_NO_SYNC = "true"
```

| MCP 工具 | 用途 |
| --- | --- |
| `search_papers` | 跨來源搜尋並依 DOI 去重 |
| `deep_search` | 以 DOI 搜尋相關論文；目前實作尚未逐層遍歷引用圖 |
| `expand_topics` | 產生相關查詢字串，不使用網路 |
| `find_gaps` | 依分析模式建立研究缺口候選 |
| `evaluate_paper_tool` | 建立委員、評分維度與 rubric |
| `prepare_review_context` | 預取文獻並建立評審上下文 |

啟動驗證應完成 MCP initialize、列出上述工具，並成功呼叫 `expand_topics`。Cloud 設定時已驗證這個流程。

`evaluate_paper_tool` 與 `prepare_review_context` 目前仍有 repository 問題：程式預期的 `src/academic_helper/agents/` 不存在，根目錄 `agents/*.md` 也缺少 loader 要求的 `focus`、`scoring_dimensions`、`prompt_template`。需要另外修正路徑與資料格式；測試中的 mock 不代表這兩個工具可使用現有檔案運作。

## 外部文獻來源

即時搜尋需要 HTTPS 存取 `api.openalex.org`、`api.semanticscholar.org`、`api.crossref.org`。`SEMANTIC_SCHOLAR_API_KEY` 在目前程式中為選用；只透過環境變數提供。需要自訂來源時，可使用 `OPENALEX_BASE_URL`、`SEMANTIC_SCHOLAR_BASE_URL`、`CROSSREF_BASE_URL`。

本次 Cloud 的三個來源請求皆收到 proxy CONNECT 403，即時搜尋尚未驗證成功。請在 Cloud 環境設定增補所需網域，保留既有白名單；設定實際生效後再測試。儲存待審設定、目前機器的執行狀態及發布環境是不同步驟，儲存 draft 不會立即變更網路或完成發布。

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
