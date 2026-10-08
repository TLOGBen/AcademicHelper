---
name: setup-zetero
description: 設定與診斷 Zotero 文獻工作流程的環境變數、Cloud 網路與唯讀 API 存取；適用於首次設定、換機、Cloud 啟動或 Zotero 認證失敗。
---

# Zotero 環境設定

沿用 `setup-zetero` 這個 skill 名稱；服務名稱是 Zotero。設定完成後交接給既有的 `zotero-literature-import`，不另建入庫實作。預設以繁體中文回報。

## 確認環境與憑證來源

- 使用現有 checkout；Cloud task 已隔離，除非使用者要求，不建立 Git worktree。
- 沿用使用者指定的 library 與既有認證。先查可用環境設定中的綁定名稱，再以 helper 檢查本機變數是否存在；不要列出整個環境、輸出 key 或讀出個人設定檔內容。
- API key 只讀 `ZOTERO_API_KEY`。程序環境值優先；Windows 可在程序變數不存在時讀取持久化的個人 User 環境值。Linux／Cloud 只讀目前程序環境。
- Library 使用 `ZOTERO_LIBRARY_ID`（正整數）與 `ZOTERO_LIBRARY_TYPE`（`user` 或 `group`，預設 `user`）。ID 與 type 可回報；key 只回報是否設定。
- Windows User 環境變數不會自動同步至 Cloud。Cloud 缺 key 時，請使用者在環境／個人 vault 的安全輸入介面提供 `ZOTERO_API_KEY`，不得要求貼在聊天、從其他聊天複製 key，或存入 repository、腳本、skill、安裝紀錄。
- 不因 token 變數未設定就重複新增認證。若平台提供代理綁定，使用其支援的 HTTPS 路由並實測；placeholder 不等同失效 key。

## 執行設定與檢查

Helper 位於此 skill 的 `scripts/zotero_doctor.py`，只使用 Python standard library。以可用的 Python 3.11+ 執行，將下列 `<skill_dir>` 換成這份 `SKILL.md` 所在目錄：

```text
python <skill_dir>/scripts/zotero_doctor.py --check-env
python <skill_dir>/scripts/zotero_doctor.py
```

第一個命令不發送網路請求，只確認必要變數與格式。第二個命令以 verified HTTPS 對官方 `api.zotero.org` 執行唯讀 GET，確認 key metadata 與所選 library 的讀取存取。實際 library 回應成功才宣稱連線有效；存在變數本身不足以證明有效認證。不要使用示範 key 作真實請求。

缺少設定時繼續可獨立完成的安裝與驗證，再提供精確缺項。需要保存 Cloud 設定時，使用可用的設定工具增補變數需求與 `api.zotero.org` 網路需求，不寫入 key 值、不覆蓋未知白名單。保存 draft 不會套用 runtime 或發布環境；等設定實際生效後重測。

將 library ID/type 以非敏感變數設定；不要把這個 skill 的 library ID 固定為某位使用者。已有來源可推知 target 時直接沿用，只有無法推知且阻止進度時才詢問。

## 解讀結果並排除失敗

- 缺 key、缺 ID 或格式錯誤：補齊對應變數，不發網路請求。不要請使用者把 key 貼在聊天。
- Proxy CONNECT 403：檢查環境的 `api.zotero.org` 網路規則與實際生效狀態；不要判定為失效 key。
- Zotero HTTP 401／403：先分別確認認證與目標 library 權限；不要因單次失敗要求重建同一個 secret。
- HTTP 429：保留 rate limit 結果，遵循 Retry-After；不要原樣連續重試。
- TLS／網路錯誤：檢查平台代理與受信任憑證設定，不關閉 TLS 驗證。
- API 格式／服務錯誤：保留狀態與已驗證範圍；不輸出 server response body，以免內容反射 key。

Helper 不建立、修改、刪除書目、筆記、分類或 PDF；也不以 GET 驗證宣稱實際寫入成功。權限 metadata 可以幫助診斷，但寫入／附件上傳須由入庫工作流程依使用者授權另行驗證。

## 交接與重用

確認 `zotero-literature-import` 是否實際安裝，再交接相同環境變數介面。僅有這個 setup skill 不代表文獻搜尋、COSMIN 評讀、Excel 匯出或 PDF 入庫已就緒。

在 AcademicHelper Cloud 中可使用：

```text
/workspace/AcademicHelper/.venv/bin/python /workspace/AcademicHelper/.agents/skills/setup-zetero/scripts/zotero_doctor.py --check-env
```

維護者可執行唯讀模擬測試：

```text
python -m unittest discover -s <skill_dir>/scripts -p test_zotero_doctor.py -v
```

Windows 安裝同一版本時，從已核實的 branch／commit 複製整個 `setup-zetero` 目錄至使用者的 skills 目錄，保留既有修改與其他 skills；不要把 API key 或個人設定一併打包。專案與 User scope 安裝需來自同一份 repository 來源。

完成時回報變數名稱、所選 library、實際 GET／模擬測試結果、保存的設定與缺項。清楚區分本機驗證、Cloud 驗證、設定保存、發布，以及尚未執行的匯入。
