import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from literature_tool import make_plan, write_plan, merge_records, parse_pubmed, parse_ris, s3_https, pdf_bytes_ok, missing_pdf_priority, enrich_citations, refresh_pdf_paths, prepare_pdfs, record_row


class ToolTests(unittest.TestCase):
    def test_pubmed_structured_date_identifiers_and_abstract(self):
        xml = b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>A <i>scale</i> study</ArticleTitle><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2017</Year><Month>Oct</Month></PubDate></JournalIssue></Journal><Abstract><AbstractText Label="METHODS">Measure</AbstractText></Abstract><AuthorList><Author><LastName>Ohara</LastName><Initials>Y</Initials></Author></AuthorList></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1111/ggi.12873</ArticleId><ArticleId IdType="pmc">PMC100</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'''
        record = parse_pubmed(xml)[0]
        self.assertEqual((record['year'], record['pmid'], record['pmcid']), (2017, '123', 'PMC100'))
        self.assertEqual(record['title'], 'A scale study')
        self.assertEqual(record['abstract'], 'METHODS: Measure')

    def test_merge_preserves_query_and_source_not_conflicting_ids(self):
        a = dict(title='A study',year=2021,doi='https://doi.org/10.123/A',source='PubMed',query_ids=['names'])
        b = dict(title='A study',year=2021,doi='10.123/a',source='Scholar',query_ids=['s1'])
        c = dict(title='A study',year=2021,doi='10.123/different',source='Cochrane',query_ids=['c1'])
        rows = merge_records([a,b,c])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['query_ids'], ['names','s1'])
        self.assertIn('Scholar', rows[0]['source'])

    def test_ris_multiline_and_incomplete_record(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            p = Path(d)/'source.ris'
            p.write_text('TY  - JOUR\nTI  - Title\nPY  - 2021/10/01\nAB  - First\n continuation\nDO  - 10.123/a\nER  -\n',encoding='utf-8')
            row = parse_ris(p,'Scholar','s1')[0]
            self.assertEqual(row['abstract'],'First continuation')
            self.assertEqual(row['year'],2021)
            p.write_text('TY  - JOUR\nTI  - Lost record',encoding='utf-8')
            with self.assertRaises(ValueError):
                parse_ris(p,'Scholar','s1')

    def test_pdf_and_official_bucket_boundary(self):
        self.assertFalse(pdf_bytes_ok(b'<html>blocked</html>'))
        self.assertTrue(pdf_bytes_ok(b'%PDF-1.7\n'+b'0'*200))
        self.assertEqual(s3_https('s3://pmc-oa-opendata/PMC1.1/a.pdf?md5=abc'),'https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/a.pdf?md5=abc')
        with self.assertRaises(ValueError):
            s3_https('https://attacker.example/a.pdf')

    def test_custom_pico_profile_does_not_fall_back_to_gseoh(self):
        cfg = {'topic':'成人氣喘患者吸入技巧','pico':{'P':'成人氣喘','I':'衛教','C':'常規衛教','O':'吸入技巧'},'queries':[{'id':'custom','database':'PubMed','label':'A different topic','query':'asthma[tiab]'}]}
        queries = make_plan(cfg)
        self.assertEqual(queries[0]['query'],'asthma[tiab]')
        self.assertNotIn('GSEOH',json.dumps(queries))
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            write_plan(Path(folder),cfg,queries)
            page = (Path(folder)/'搜尋式.html').read_text(encoding='utf-8')
            self.assertIn(cfg['topic'],page)
            self.assertNotIn('GSEOH',page)
            self.assertNotIn('OHSES',page)

    def test_missing_citation_not_zero_and_quality_not_invented(self):
        records = [dict(title='GSEOH zero',citation_count=0),dict(title='GSEOH unknown'),dict(title='GSEOH cited',citation_count=50)]
        ranked = missing_pdf_priority(records)
        self.assertEqual([r['title'] for r in ranked],['GSEOH cited','GSEOH zero','GSEOH unknown'])
        self.assertIsNone(ranked[-1]['citation_count'])
        self.assertEqual(ranked[0]['quality_assessments'],[])

    def test_quality_comparison_requires_same_property_and_evidence(self):
        def assessment(prop,rating):
            return {'property':prop,'rating':rating,'source':'test-fixture.pdf','location':'p.4','tool':'COSMIN'}
        records = [dict(title='low cite sound',citation_count=2,quality_assessments=[assessment('content validity','very good')]),
                   dict(title='high cite flawed',citation_count=500,quality_assessments=[assessment('content validity','inadequate')]),
                   dict(title='other property',citation_count=999,quality_assessments=[assessment('reliability','very good')])]
        ranked = missing_pdf_priority(records,quality_property='content validity')
        self.assertEqual([r['title'] for r in ranked],['low cite sound','high cite flawed','other property'])
        self.assertEqual(ranked[-1]['quality_rating'],'待評／尚無可比較評級')

    def test_priority_for_new_topic_uses_its_own_core_terms(self):
        records = [dict(title='GSEOH unrelated',citation_count=100),dict(title='Asthma relevant',citation_count=2)]
        ranked = missing_pdf_priority(records,core_terms=['asthma'])
        self.assertEqual(ranked[0]['title'],'Asthma relevant')
        self.assertEqual(ranked[1]['scope'],'研究問題候選（需篩選）')
        generic = missing_pdf_priority(records,core_terms=[])
        self.assertTrue(all(r['scope']=='研究問題候選（需篩選）' for r in generic))

    def test_deleted_pdf_returns_to_pending_and_preserves_previous_location(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            pdf = Path(folder) / 'deleted.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            record = dict(title='Retained paper', pdf_path=str(pdf), pdf_status='已下載（PMC 官方雲端）')
            self.assertEqual(missing_pdf_priority([record], out=folder), [])
            pdf.unlink()
            refresh_pdf_paths([record], Path(folder))
            self.assertNotIn('pdf_path', record)
            self.assertEqual(record['pdf_missing_path'], str(pdf))
            self.assertIn('待重新取得', record['pdf_status'])
            self.assertEqual(missing_pdf_priority([record], out=folder)[0]['title'], 'Retained paper')

    def test_restored_pdf_updates_exported_status_and_preserves_missing_location(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            out = Path(folder)
            pdf = out / 'restored.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            for workflow in ('rank', 'import'):
                with self.subTest(workflow=workflow):
                    record = dict(title='Restored paper', pdf_path='deleted.pdf')
                    refresh_pdf_paths([record], out)
                    self.assertEqual(len(missing_pdf_priority([record], out=out)), 1)
                    record['pdf_path'] = 'restored.pdf'
                    if workflow == 'rank':
                        refresh_pdf_paths([record], out)
                    else:
                        prepare_pdfs([record], None, out, 0)
                    self.assertEqual(record['pdf_status'], '已有本地 PDF（已恢復）')
                    self.assertEqual(record_row(1, record)[13], record['pdf_status'])
                    self.assertEqual(record['pdf_path'], str(pdf.resolve()))
                    self.assertEqual(record['pdf_missing_path'], 'deleted.pdf')
                    self.assertEqual(missing_pdf_priority([record], out=out), [])
                    refresh_pdf_paths([record], out)
                    self.assertEqual(record['pdf_status'], '已有本地 PDF（已恢復）')

    def test_refresh_retains_valid_pdf_download_provenance(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            out = Path(folder)
            pdf = out / 'retained.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            record = dict(title='Downloaded paper', pdf_path='retained.pdf',
                          pdf_status='已下載（PMC 官方雲端）')
            refresh_pdf_paths([record], out)
            self.assertEqual(record['pdf_status'], '已下載（PMC 官方雲端）')

    def test_relative_manifest_pdf_is_resolved_from_output_not_workspace(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, tempfile.TemporaryDirectory(dir=Path.cwd()) as unrelated:
            out = Path(folder)
            (out / 'pdfs').mkdir()
            pdf = out / 'pdfs' / 'retained.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            record = dict(title='Moved output', pdf_path='pdfs/retained.pdf')
            with patch('literature_tool.ROOT', Path(unrelated)):
                self.assertEqual(missing_pdf_priority([record], out=out), [])
                prepare_pdfs([record], None, out, 0)
            self.assertEqual(record['pdf_path'], str(pdf.resolve()))

    def test_invalid_pdf_path_is_included_in_pending(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            fake = Path(folder) / 'blocked.pdf'
            fake.write_bytes(b'<html>not a PDF</html>')
            record = dict(title='Blocked download', pdf_path=str(fake))
            self.assertEqual(len(missing_pdf_priority([record], out=folder)), 1)

    def test_citation_enrichment_checks_missing_file_not_only_path_string(self):
        class Client:
            def get(self, url, params):
                self.params = params
                return json.dumps({'results': [{'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/10.123/missing', 'cited_by_count': 7}]})
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            out = Path(folder)
            (out / 'raw').mkdir()
            retained = out / 'retained.pdf'
            retained.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            records = [dict(title='Missing', doi='10.123/missing', pdf_path='deleted.pdf'),
                       dict(title='Retained', doi='10.123/retained', pdf_path='retained.pdf')]
            client = Client()
            enrich_citations(records, client, out)
            self.assertEqual(client.params['filter'], 'doi:10.123/missing')
            self.assertEqual(records[0]['citation_count'], 7)
            self.assertNotIn('citation_status', records[1])

    def test_local_pdf_still_resolves_from_research_workspace(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as workspace, tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
            pdf = Path(workspace) / 'source.pdf'
            pdf.write_bytes(b'%PDF-1.7\n' + b'0' * 200)
            record = dict(title='User file', local_pdf='source.pdf')
            with patch('literature_tool.ROOT', Path(workspace)):
                self.assertEqual(missing_pdf_priority([record], out=folder), [])
                prepare_pdfs([record], None, Path(folder), 0)
            self.assertEqual(record['pdf_path'], str(pdf.resolve()))
            self.assertEqual(record['pdf_status'], '已有本地 PDF（先前取得）')


if __name__ == '__main__':
    unittest.main()
