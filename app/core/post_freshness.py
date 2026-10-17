"""Pure publication-date checks shared by discovery and request admission."""
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
import re

_RECENT = re.compile(r'最新|最近|近期|刚(?:发|上传)|\b(?:latest|recent|newest)\b',re.I)
_WINDOW = re.compile(r'(?:最近|近|过去)\s*(\d{1,3}|[一二两三四五六七八九十百]{1,4})\s*(天|周|日|星期)')


def publication_time(value) -> datetime | None:
    if not isinstance(value,str) or len(value)>100:return None
    try:
        try:stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
        except ValueError:stamp=parsedate_to_datetime(value)
        return stamp.astimezone(UTC) if stamp.tzinfo is not None else None
    except (ValueError,TypeError,OverflowError):return None


def constraint_text(request: str) -> str:
    """Quoted examples and negated clauses cannot create a hard date condition."""
    text=re.sub(r'「[^」]*」|“[^”]*”|‘[^’]*’|"[^"\n]*"', ' ', request)
    text=re.sub(r'(?:不要|不必|不用|不需要|不限|不要求|不是|别只|无需|并非|not |without |no need to )[^，。；;\n]*', ' ', text, flags=re.I)
    return text


def explicit_recent_days(request: str) -> int | None:
    text=constraint_text(request)
    match=_WINDOW.search(text)
    if match:
        value=match[1]
        if value.isdigit():number=int(value)
        else:
            number=0;digit=0
            for char in value:
                if char in '十百':number+=(digit or 1)*{'十':10,'百':100}[char];digit=0
                else:digit={'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}[char]
            number+=digit
        return min(365,max(1,number)*(7 if match[2] in {'周','星期'} else 1))
    match=re.search(r'\b(?:last|past)\s+(\d{1,3})\s+(days?|weeks?)\b',text,re.I)
    return min(365,max(1,int(match[1]))*(7 if match[2].lower().startswith('week') else 1)) if match else None


def publication_filter(request: str, now: datetime) -> dict:
    text=constraint_text(request)
    days=explicit_recent_days(request)
    after=now.astimezone(UTC)-timedelta(days=days) if days else None
    before=None;proof=''
    if days:
        match=_WINDOW.search(text) or re.search(r'\b(?:last|past)\s+\d{1,3}\s+(?:days?|weeks?)\b',text,re.I)
        proof=match[0]
    else:
        # A year is a hard range only when explicitly tied to the requested time.
        match=re.search(r'(?<!\d)((?:19|20)\d{2})年(?:的|里|期间)',text)
        relative=re.search(r'去年|前年',text)
        if relative:
            year=now.year-(2 if relative[0]=='前年' else 1)
            after=datetime(year,1,1,tzinfo=UTC);before=datetime(year+1,1,1,tzinfo=UTC);proof=relative[0]
        if match:
            year=int(match[1]);after=datetime(year,1,1,tzinfo=UTC);before=datetime(year+1,1,1,tzinfo=UTC);proof=match[0]
        match=re.search(r'(?:从|自|since\s+|after\s+)(20\d{2}-\d{2}-\d{2})',text,re.I)
        if match:
            try:after=datetime.fromisoformat(match[1]).replace(tzinfo=UTC);proof=match[0]
            except ValueError:pass
    return {'published_after':after.isoformat() if after else None,
        'published_before':before.isoformat() if before else None,
        'lookback_days':days,'provenance':'explicit_user' if proof else 'unbounded', 'proof':proof}


def recent_cutoff(request: str, now: datetime) -> datetime | None:
    """Only an explicit user range is an admission constraint."""
    value=publication_filter(request,now)['published_after']
    return datetime.fromisoformat(value) if value else None
