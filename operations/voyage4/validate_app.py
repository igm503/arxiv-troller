"""Read-only application integration checks, with V4 enabled only inside this process."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
os.environ.setdefault('DJANGO_SETTINGS_MODULE','arxiv_troller.settings_django_deploy')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'django'))
import django
django.setup()
from django.db import connection
from django.test import override_settings,RequestFactory
from django.contrib.auth.models import AnonymousUser
from papers import views,voyage4_search
from papers.models import Paper,Tag
import archive as a

with connection.cursor() as q:
    q.execute("SELECT value FROM voyage4.control WHERE key='rolling'");rolling=q.fetchone()[0]
    if isinstance(rolling,str):rolling=json.loads(rolling)
source=Paper.objects.get(pk=3008578);results=[]
with override_settings(VOYAGE4_ENABLED=True,ALLOWED_HOSTS=['testserver']),patch.object(voyage4_search,'controls',return_value={'ready':True,'rolling':rolling}),patch.object(views.EMBEDDING_MODEL.objects,'filter',side_effect=AssertionError('Unexpected legacy fallback')):
    for category in ['', 'cs.LG']:
        context=dict(single_paper_id=str(source.id),date_filter='1month',category_filter=category,exclude_ids=set(),current_tag=None)
        start=time.perf_counter();papers,_=views.paper_search(context);lat=(time.perf_counter()-start)*1000
        assert len(papers)==20 and source.id not in [p.id for p in papers]
        assert all(p.created>=views.get_date_cutoff('1month') and (not category or category in p.categories) for p in papers)
        context['exclude_ids'].update(p.id for p in papers)
        next_page,_=views.paper_search(context);assert not(set(p.id for p in papers)&set(p.id for p in next_page))
        results.append(dict(case='monthly'+('_field' if category else ''),papers=len(papers),next_page=len(next_page),first_call_ms=lat))
    request=RequestFactory().get('/',{'single_paper':source.id,'date_filter':'1month','category':'cs.LG'});request.user=AnonymousUser()
    response=views.search(request);assert response.status_code==200
    results.append(dict(case='rendered_anonymous_search',status=response.status_code,html_bytes=len(response.content)))
    tag=Tag.objects.filter(tagged_papers__isnull=False).first()
    if tag:
        context=dict(parsed_tag_for_search=tag,date_filter='1month',category_filter='',exclude_ids=set(),current_tag=None)
        papers,_=views.tag_search(context);assert papers
        results.append(dict(case='existing_tag_similarity',papers=len(papers)))
a.atomic_json(a.ROOT/'application-validation.json',dict(passed=True,checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),legacy_fallback_forbidden_in_test=True,results=results))
print(json.dumps(results,indent=2))
