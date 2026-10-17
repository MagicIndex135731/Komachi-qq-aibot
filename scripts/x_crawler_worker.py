"""One bounded container crawler request; stdin/stdout contain no credentials."""
import json
import logging
import os
from pathlib import Path
import signal
import sys

if __package__ in {None,''}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from app.providers.x_crawler import XCrawler,CrawlerError,load_session
from app.providers.x_posts import MAX_X_CRAWLER_REQUEST_BYTES


def main():
    logging.disable(logging.CRITICAL)
    os.environ['TWS_TELEMETRY']='0'
    os.umask(0o077)
    signal.alarm(42)
    cookies={}
    try:
        body=sys.stdin.buffer.read(MAX_X_CRAWLER_REQUEST_BYTES+1)
        if len(body)>MAX_X_CRAWLER_REQUEST_BYTES:raise ValueError()
        request=json.loads(body)
        cookies=load_session(Path(os.environ['X_CRAWLER_SESSION_FILE']))
        result=XCrawler(cookies).acquire(request['action'],request['target'],
            limit=request.get('limit',6),lookback_days=request.get('lookback_days'),
            **{key:request[key] for key in ('media_types','sort','min_likes','min_replies','min_reposts',
                'subject_terms','topic_terms','topic_match') if key in request})
        print(json.dumps({'ok':True,'data':result},ensure_ascii=False,separators=(',',':')))
    except CrawlerError as exc:
        print(json.dumps({'ok':False,'error':exc.code}))
    except Exception:
        print(json.dumps({'ok':False,'error':'parse_failed'}))
    finally:cookies.clear();signal.alarm(0)


if __name__=='__main__':main()
