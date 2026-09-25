"""Read-only application integration checks, with V4 enabled only inside this process."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
os.environ.setdefault('DJANGO_SETTINGS_MODULE','arxiv_troller.settings')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'django'))
import django
django.setup()
from django.db import connection
from django.test import override_settings,RequestFactory
from django.contrib.auth.models import AnonymousUser
from papers import views,voyage4_search
from papers.models import Paper,Tag
import archive as a

def main():
    with connection.cursor() as q:
        q.execute("SELECT value FROM voyage4.control WHERE key='rolling'");rolling=q.fetchone()[0]
        q.execute('SELECT paper_id FROM voyage4.rolling30 ORDER BY created DESC LIMIT 1');source_ids=[r[0] for r in q.fetchall()]
        if isinstance(rolling,str):rolling=json.loads(rolling)
    source=Paper.objects.filter(id__in=source_ids).latest('created');results=[]
    with override_settings(VOYAGE4_ENABLED=True,ALLOWED_HOSTS=['testserver']),patch.object(voyage4_search,'controls',return_value={'ready':True,'rolling':rolling}),patch.object(views.EMBEDDING_MODEL.objects,'filter',side_effect=AssertionError('Unexpected legacy fallback')):
        for category in ['', 'cs.LG']:
            context=dict(single_paper_id=str(source.id),date_filter='1month',category_filter=category,exclude_ids=set(),current_tag=None)
            start=time.perf_counter();papers,_=views.paper_search(context);lat=(time.perf_counter()-start)*1000
            a.require(len(papers)==20 and source.id not in [p.id for p in papers], 'validate_app.py: len(papers)==20 and source.id not in [p.id for p in papers]')
            a.require(all(p.created>=views.get_date_cutoff('1month') and (not category or category in p.categories) for p in papers), "validate_app.py: all(p.created>=views.get_date_cutoff('1month') and (not category or category in p.categories) for p in papers)")
            context['exclude_ids'].update(p.id for p in papers)
            next_page,_=views.paper_search(context);a.require(not(set(p.id for p in papers)&set(p.id for p in next_page)), 'validate_app.py: not(set(p.id for p in papers)&set(p.id for p in next_page))')
            results.append(dict(case='monthly'+('_field' if category else ''),papers=len(papers),next_page=len(next_page),first_call_ms=lat))
        request=RequestFactory().get('/',{'single_paper':source.id,'date_filter':'1month','category':'cs.LG'});request.user=AnonymousUser()
        response=views.search(request);a.require(response.status_code==200, 'validate_app.py: response.status_code==200')
        results.append(dict(case='rendered_anonymous_search',status=response.status_code,html_bytes=len(response.content)))
        detail = views.paper_detail(request, source.id)
        a.require(b'Find Similar Papers' in detail.content, 'Voyage-only detail link missing')
        results.append(dict(case='voyage_only_detail', status=detail.status_code))
        context = dict(date_filter='1month', category_filter='', exclude_ids=set(), current_tag=None,
                       title_query=source.title, query='learning')
        a.require(list(views.title_search(context)[0]), 'Title search failed')
        list(views.keyword_search(context)[0])

        tag=Tag.objects.filter(tagged_papers__isnull=False).first()
        if tag:
            context=dict(parsed_tag_for_search=tag,date_filter='1month',category_filter='',exclude_ids=set(),current_tag=None)
            papers,_=views.tag_search(context);a.require(papers, 'validate_app.py: papers')
            results.append(dict(case='existing_tag_similarity',papers=len(papers)))
    a.atomic_json(a.ROOT/'application-validation.json',dict(passed=True,checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),legacy_fallback_forbidden_in_test=True,results=results))
    print(json.dumps(results,indent=2))

if __name__ == "__main__":
    main()
