"""Offline publication-helper tests. These do not exercise any model or GitHub service."""
from __future__ import annotations
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('publisher',ROOT/'scripts/omni_jev_publish_issues.py')
pub=importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(pub)

class FakeGitHub:
    def __init__(self):
        self.issues=[]
        self.posts=0
        self.readbacks=0
        self.enabled=True
        self.name='bacoco/Omni-JEV'
        self.corrupt=False
    def all_issues(self,repo):
        return copy.deepcopy(self.issues)
    def request(self,method,endpoint,payload=None):
        if method=='GET' and endpoint=='repos/bacoco/Omni-JEV':
            return {'full_name':self.name,'has_issues':self.enabled}
        if method=='POST':
            self.posts+=1
            item={'number':len(self.issues)+1,'title':payload['title'],'body':payload['body']}
            item['html_url']=f"https://github.com/bacoco/Omni-JEV/issues/{item['number']}"
            self.issues.append(item)
            return copy.deepcopy(item)
        if method=='GET' and '/issues/' in endpoint:
            self.readbacks+=1
            item=copy.deepcopy(self.issues[int(endpoint.rsplit('/',1)[1])-1])
            if self.corrupt:item['body']='unexpected content'
            return item
        raise AssertionError((method,endpoint))

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.repo,self.items=pub.load_manifest()
        self.client=FakeGitHub()
        self.report={'completed':False,'items':[]}
    def run_publish(self,selected=None,report=None):
        with contextlib.redirect_stdout(io.StringIO()):
            pub.publish(self.client,self.repo,self.items,
                        selected or {i['id'] for i in self.items},report or self.report)
    def test_manifest_has_thirteen_valid_drafts(self):
        self.assertEqual(len(self.items),13)
        self.assertEqual(self.items[-1]['id'],'OJ-000')
    def test_dry_run_does_not_construct_network_client(self):
        with patch.object(pub,'GitHubClient',side_effect=AssertionError('no network')),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pub.main([]),0)
    def test_serial_rerun_skips_all_existing(self):
        self.run_publish()
        self.assertEqual(self.client.posts,13)
        self.assertEqual(self.client.readbacks,13)
        self.assertTrue(self.report['completed'])
        before=copy.deepcopy(self.client.issues)
        later={'completed':False,'items':[]}
        self.run_publish(report=later)
        self.assertEqual(self.client.posts,13)
        self.assertEqual(before,self.client.issues)
        self.assertTrue(all(i['action']=='skipped_existing' for i in later['items']))
    def test_summary_only_is_self_contained(self):
        self.run_publish({'OJ-000'})
        self.assertEqual(self.client.posts,1)
        body=self.client.issues[0]['body']
        self.assertIn('Noul',body)
        self.assertIn('local draft; not linked',body)
        self.assertNotIn('{{',body)
    def test_dependencies_and_summary_get_real_links(self):
        self.run_publish()
        four=next(i for i in self.client.issues if pub.marker('OJ-004') in i['body'])
        self.assertIn('[OJ-001](https://github.com/bacoco/Omni-JEV/issues/1)',four['body'])
        summary=self.client.issues[-1]['body']
        self.assertNotIn('local draft; not linked',summary)
    def test_unmarked_title_conflict_prevents_posts(self):
        self.client.issues=[{'number':1,'title':self.items[0]['title'],'body':'manual content','html_url':'existing'}]
        with self.assertRaises(pub.PublishError):self.run_publish()
        self.assertEqual(self.client.posts,0)
    def test_disabled_issues_do_not_change_settings(self):
        self.client.enabled=False
        with self.assertRaises(pub.PublishError):self.run_publish()
        self.assertEqual(self.client.posts,0)
    def test_repository_mismatch_stops_before_writes(self):
        self.client.name='another/repository'
        with self.assertRaises(pub.PublishError):self.run_publish()
        self.assertEqual(self.client.posts,0)
    def test_failed_readback_is_recorded_not_retried(self):
        self.client.corrupt=True
        with self.assertRaises(pub.PublishError):self.run_publish({'OJ-001'})
        self.assertEqual(self.client.posts,1)
        self.assertFalse(self.report['completed'])
        self.assertEqual(self.report['items'][0]['action'],'created_unverified')
    def test_unknown_selection_is_rejected_offline(self):
        with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pub.main(['--only','OJ-999']),1)
    def test_bodies_have_no_unrendered_template_placeholders(self):
        for item in self.items:
            body=(ROOT/item['body_file']).read_text()
            self.assertNotIn('{{',body)
            self.assertIn(pub.marker(item['id']),body)

if __name__=='__main__':unittest.main()
