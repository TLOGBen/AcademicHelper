# 助手用設定與診斷

本文件由助手執行，不作為一般使用者的操作清單。沿用當次工具和環境；所有缺項先由助手判斷與處理，使用者只完成必要的帳號授權或文庫選擇。

## 工具與環境

使用現有 checkout 或已安裝的 skill，不為設定另建 worktree。以可用的 Python 3.11+ 執行 skill 相對目錄 `scripts/zotero_doctor.py`；沒有 runtime 探索工具時用 shell 查實際執行檔，不猜固定個人路徑。

| 變數 | 助手處理 |
| --- | --- |
| `ZOTERO_API_KEY` | 只讀環境；由實際安全值輸入介面注入，不接受聊天或檔案中的 key |
| `ZOTERO_LIBRARY_ID` | 沿用目標，或查找後設定非敏感的正整數 ID |
| `ZOTERO_LIBRARY_TYPE` | 個人 `user`、群組 `group`，未設定預設個人 |

程序值優先，即使為空也不以其他來源覆蓋。Windows 在變數不存在時可讀 HKCU User 環境值；Linux／Cloud 只讀程序環境。不要列印 key、整個環境或個人設定檔。需要設定 ID/type 時，用 host 的設定工具；僅有 shell 時可在 doctor 子程序的環境傳入這兩個非敏感值，並分別確認持久化與當次驗證，不能混為一談。

## 命令

`<python>`、`<skill_dir>` 由助手換成實際位置：

```text
<python> <skill_dir>/scripts/zotero_doctor.py --check-env
<python> <skill_dir>/scripts/zotero_doctor.py --discover-library
<python> <skill_dir>/scripts/zotero_doctor.py
```

- `--check-env`：離線檢查必要變數與格式，無網路。
- `--discover-library`：只需要 key；對 `/keys/current` 執行一次官方 HTTPS GET，僅在 metadata 明確允許個人文庫讀取且 userID 合法時，回傳候選 `library.id/type`。不保存設定、不讀取書目、不驗證文庫本身；只在個人目標缺 ID 時使用，不取代使用者明確選定的文庫。群組目標或群組專用權限回傳 `library_selection_required`，不猜文庫。
- 預設：對 `/keys/current` 與所選文庫 `/items/top?limit=1` 執行唯讀 GET，後者成功才算連接就緒。

三種模式共用固定官方 host、已驗證的 TLS、代理處理、禁止重新導向及有限 JSON 讀取。輸出只有安全的狀態與必要的非敏感文庫識別資料，不含 key、使用者名稱、私有書目或原始錯誤回應。識別時 `config_valid` 保持 false，實際文庫檢查仍未執行；不要將成功 exit code 一律翻譯為已連接。

## 狀態與下一步

| 狀態 | 助手動作與使用者說明 |
| --- | --- |
| `missing_config` | 自行補 ID/type；只有 key 缺少／格式錯誤才引導安全授權。Exit 2，不發網路請求 |
| `config_ready` | 格式已備妥，繼續實際讀取驗證 |
| `library_discovered` | 設定回傳的個人目標，繼續預設 doctor；不能直接宣布就緒 |
| `library_selection_required` | 尊重群組選擇／現有權限，使用名稱或頁面連結選定目標；不默認改個人文庫 |
| `ready` | 所選文庫讀取成功，向使用者回報已連接 |
| `network_policy` | 處理 `api.zotero.org` 的實際網路規則；CONNECT 403 不是 key 失效 |
| `unauthorized`／`forbidden` | 分別查授權與文庫權限；僅讓使用者補必要授權，不反覆建立 secret |
| `rate_limited` | 遵循服務／host 提供的等待資訊，無 Retry-After 時採保守退避；不立即連續重試，不假裝已排程 |
| `network_or_tls_error` | 核對代理與受信任憑證；保留 TLS 驗證 |
| `unexpected_response`／`http_error` | 保留實際已驗證範圍，檢查服務；不顯示 raw body、headers 或 exception |

GET 權限 metadata 不等於實際寫入驗證。寫入或 PDF 上傳交由已安裝的入庫流程依使用者要求處理。

## Cloud 與重用

可用時以 Cloud 設定工具增補變數 metadata 和 `api.zotero.org`，保留既有網路規則／設定，絕不傳入 key 值。安全值由使用者在平台提供的安全介面填入。保存 draft 不會套用網路、注入變數或發布；回報已準備內容與最少剩餘操作，生效後再做真實檢查。若工具不能提供安全值輸入，就明說此步仍需使用者完成，不能改存明文檔案。

安裝完整 skill 目錄時沿用同一 repository branch／commit，先比對既有安裝並保留本機修改與其他 skills。Windows 個人憑證不隨 skill 或聊天同步到 Cloud。確認 `zotero-literature-import` 實際存在且使用相同環境介面後再交接；不把 setup 測試當入庫驗證。

維護者離線測試：

```text
<python> -m unittest discover -s <skill_dir>/scripts -p test_zotero_doctor.py -v
```
