import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const resolve = createRequire(import.meta.url);
const runtime = process.env.CODEX_NODE_MODULES || path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies/node');
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(resolve.resolve('@oai/artifact-tool',{paths:[runtime]})).href);

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('Usage: node export_excel.mjs manifest.json output.xlsx');
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const wb = Workbook.create();
const safe = v => typeof v === 'string' && /^[=+@-]/.test(v) ? "'" + v : v ?? '';
// Excel dates have no timezone; store UTC+8 wall time and label the column.
const localStamp = v => /^\d{4}-\d{2}-\d{2}T/.test(v || '') ? new Date(new Date(v).getTime() + 8 * 3600000) : v || '';
const brief = s => (s ?? '').replace(/\s+/g, ' ').slice(0, 240);
const headers = ['序號','文章標題','年份','作者','期刊','文獻類別','PMID','DOI','PMCID','來源','命中搜尋式','書目或官方網址','全文入口','PDF 狀態','PDF 本地路徑','PDF 來源網址','PDF 授權','摘要節錄','篩選決定','排除理由','閱讀筆記'];
function classify(r) {
  return r.kind || (r.pmid === '27531046' || /gseoh/i.test((r.title ?? '') + ' ' + (r.abstract ?? '')) ? 'GSEOH 名稱或原文命中（仍需閱讀）' : '相關量表／方法候選（需人工篩選）');
}
const rows = data.records.map((r,i) => [i+1,r.title,r.year,r.authors,r.journal,classify(r),r.pmid,r.doi,r.pmcid,r.source,(r.query_ids??[]).join('; '),r.url,r.fulltext_url,r.pdf_status,r.pdf_path,r.pdf_url,r.license,brief(r.abstract),r.screening_decision||'未篩選',r.exclusion_reason,r.reading_notes].map(safe));
function createSheet(name, headers, rows, widths, heights) {
  const sh = wb.worksheets.add(name);
  const matrix = [headers, ...rows];
  const range = sh.getRangeByIndexes(0,0,matrix.length,headers.length);
  range.values = matrix;
  range.format.font = { name:'Arial',size:11,color:'#203238' };
  range.format.verticalAlignment = 'top';
  range.format.wrapText = true;
  range.format.rowHeight = heights;
  sh.getRangeByIndexes(0,0,1,headers.length).format = { fill:'#294d58',font:{name:'Arial',size:11,bold:true,color:'#ffffff'},rowHeight:32,verticalAlignment:'center',horizontalAlignment:'center',wrapText:true };
  widths.forEach((width,i)=>sh.getRangeByIndexes(0,i,matrix.length,1).format.columnWidth=width);
  sh.freezePanes.freezeRows(1);
  sh.freezePanes.freezeColumns(name==='文獻清單'?2:1);
  sh.showGridLines=false;
  const end = String.fromCharCode(64+headers.length);
  const tableName = name==='文獻清單'?'LiteratureRecords':name==='待取得全文'?'PendingFulltexts':'SearchRecords';
  const table = sh.tables.add(`A1:${end}${matrix.length}`,true,tableName);
  table.name = tableName;
  return sh;
}
const literature = createSheet('文獻清單',headers,rows,[8,86,9,42,40,40,15,48,18,38,30,75,75,47,100,100,22,105,17,45,80],78);
literature.getRangeByIndexes(1,0,Math.max(rows.length,1),1).setNumberFormat('0');
literature.getRangeByIndexes(1,2,Math.max(rows.length,1),1).setNumberFormat('0');
literature.getRangeByIndexes(1,18,Math.max(rows.length,1),3).format.fill='#fff4cf';
const executed = new Map(data.searches.map(x=>[x.id,x]));
const all = data.strategy.map(x=>({...x,...(executed.get(x.id)||{})}));
all.push(...data.searches.filter(x=>!data.strategy.some(p=>p.id===x.id)));
const searchRows = all.map(x=>[x.database||'',x.id,x.label||'',x.query||'',x.status,localStamp(x.searched_at),x.total_hits??null,x.retrieved??null,x.truncated===undefined?'':x.truncated?'是（候選清單不完整）':'否',x.url||'',JSON.stringify(x.warnings||{}),x.error||''].map(safe));
const searchSheet = createSheet('搜尋紀錄',['資料庫／介面','搜尋 ID','目的','實際搜尋式／待執行策略','執行狀態','搜尋時間（UTC+8）','命中總筆數','取得筆數','是否截取','搜尋網址','API 警告','失敗原因'],searchRows,[32,26,65,135,44,32,16,16,35,100,60,80],145);
if (searchRows.length) searchSheet.getRangeByIndexes(1,5,searchRows.length,1).setNumberFormat('yyyy-mm-dd hh:mm:ss');
if (data.missing_pdf_priority?.length) {
  const pending = data.missing_pdf_priority.map(r=>[r.priority,r.title,r.year,r.scope,r.citation_count??null,r.citation_source,localStamp(r.citation_checked_at),r.citation_status,r.quality_property||'',r.quality_rating||'待評',r.quality_status,r.quality_assessments?.length?JSON.stringify(r.quality_assessments):'',r.doi,r.pmid,r.fulltext_url,r.pdf_status].map(safe));
  const pendingSheet = createSheet('待取得全文',['順位','文章標題','年份','文獻範圍','引用次數','引用來源','查詢時間（UTC+8）','引用數狀態','比較的測量特性','COSMIN 該特性評級','品質評估狀態','各測量特性評級／證據','DOI','PMID','全文入口','PDF 取得狀態'],pending,[8,100,9,45,14,18,32,48,35,35,75,100,50,16,100,50],90);
  pendingSheet.getRangeByIndexes(1,6,pending.length,1).setNumberFormat('yyyy-mm-dd hh:mm:ss');
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'文獻清單!A1:C5',include:'values',tableMaxRows:5,tableMaxCols:3,maxChars:1200})).ndjson);
const errors = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1200});
console.log(errors.ndjson);
const previews = [['文獻清單','A1:E5','literature_preview.png'],['搜尋紀錄','A1:C5','search_preview.png']];
if (data.missing_pdf_priority?.length) previews.push(['待取得全文','A1:H5','pending_preview.png'],['待取得全文','I1:L5','quality_preview.png']);
for (const [name,range,file] of previews) {
  const preview = await wb.render({sheetName:name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(path.dirname(output),file),new Uint8Array(await preview.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(output);
console.log(`Exported ${rows.length} records: ${output}`);
