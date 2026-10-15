import httpx
import pytest

from app.agent.artifacts import AutonomousArtifactStore
from app.agent.web_images import build_web_image_tools
from app.providers.web_images import WebImageClient
from test_autonomous_social import BOT, GROUP, OTHER_GROUP, add_source, image_bytes


class Runner:
    async def run(self, function, *args, **kwargs):
        return function(*args, **kwargs)


class Guard:
    valid = True
    def require_valid(self):
        if not self.valid:
            raise RuntimeError('source revoked')


def tools_for(monkeypatch, engine, tmp_path, respond, *, group=GROUP, sources=()):
    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = WebImageClient(http_client=client)
    monkeypatch.setattr('app.agent.web_images.WebImageClient', lambda _: provider)
    store = AutonomousArtifactStore(tmp_path / 'artifacts', account_id=BOT, scope=f'group:{group}',
        engine=engine, source_message_ids=lambda: sources)
    guard = Guard()
    tools = {tool.name: tool for tool in build_web_image_tools(None, runner=Runner(), guard=guard, store=store)}
    return tools, store, guard, client


async def test_page_candidates_actual_pixels_and_provenance_do_not_send(sqlite_engine, tmp_path, monkeypatch):
    source = add_source(sqlite_engine, platform_id='web-image-request')
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, content=image_bytes()) if request.url.path == '/image.png' else httpx.Response(200, headers={'Content-Type': 'text/html'},
            text='<h2>鹿目圆人设图</h2><img alt="小圆" src="/image.png">')
    tools, store, guard, client = tools_for(monkeypatch, sqlite_engine, tmp_path, respond, sources=(source,))
    try:
        listed = await tools['list_web_images'].handler({'ref_id': 'https://example.com/page'})
        image_id = listed.data['images'][0]['image_id']
        downloaded = await tools['download_web_image'].handler({'image_id': image_id})
        assert downloaded.status == 'ok' and downloaded.data['actual_pixels']
        artifact_id = downloaded.data['artifact_id']
        assert store.get(artifact_id).read_bytes() == image_bytes()
        assert store.provenance(artifact_id) == (source,)
        assert downloaded.images[0].url == '' and downloaded.images[0].local_path
        assert (await tools['download_web_image'].handler({'image_id': image_id})).data['artifact_id'] == artifact_id
        await tools['list_web_images'].handler({'ref_id': 'https://example.com/page'})
        assert len(calls) == 2
        other = AutonomousArtifactStore(store.root, account_id=BOT, scope=f'group:{OTHER_GROUP}',
            engine=sqlite_engine, source_message_ids=lambda: ())
        assert other.get(artifact_id) is None
        fresh = {tool.name: tool for tool in build_web_image_tools(None, runner=Runner(), guard=guard, store=store)}
        assert (await fresh['download_web_image'].handler({'image_id': image_id})).status == 'blocked'
        guard.valid = False
        with pytest.raises(RuntimeError, match='revoked'):
            await tools['download_web_image'].handler({'image_id': image_id})
    finally:
        client.close()


@pytest.mark.parametrize('content', [b'not pixels', b'<html>Login</html>'])
async def test_download_invalid_pixels_never_produces_artifact(sqlite_engine, tmp_path, monkeypatch, content):
    def respond(request):
        return httpx.Response(200, content=content) if request.url.path == '/a.png' else httpx.Response(200, headers={'Content-Type': 'text/html'}, text='<img src="/a.png">')
    tools, store, _, client = tools_for(monkeypatch, sqlite_engine, tmp_path, respond)
    try:
        page = await tools['list_web_images'].handler({'ref_id': 'https://example.com/'})
        result = await tools['download_web_image'].handler({'image_id': page.data['images'][0]['image_id']})
        assert result.status == 'failed' and result.images == ()
        assert not store.list() and not list(store.root.glob('tmp*'))
    finally:
        client.close()


async def test_paging_and_page_download_budgets(sqlite_engine, tmp_path, monkeypatch):
    def respond(request):
        return httpx.Response(200, content=image_bytes()) if request.url.path.endswith('.png') else httpx.Response(200, headers={'Content-Type': 'text/html'},
            text=''.join(f'<img src="/{n}.png">' for n in range(25)))
    tools, _, _, client = tools_for(monkeypatch, sqlite_engine, tmp_path, respond)
    try:
        page = await tools['list_web_images'].handler({'ref_id': 'https://example.com/page'})
        assert len(page.data['images']) == 20 and page.data['next_offset'] == 20
        second = await tools['list_web_images'].handler({'ref_id': 'https://example.com/page', 'offset': 20})
        assert len(second.data['images']) == 5 and second.data['next_offset'] is None
        batch = await tools['download_web_images'].handler({'image_ids': [item['image_id'] for item in page.data['images']]})
        assert batch.status == 'ok' and len(batch.data['artifact_ids']) == 20 and batch.data['all_requested_downloaded']
        assert len(batch.images) == 2
        assert (await tools['download_web_image'].handler({'image_id': second.data['images'][0]['image_id']})).status == 'blocked'
        for n in range(3):
            assert (await tools['list_web_images'].handler({'ref_id': f'https://example.com/page{n}'})).status == 'ok'
        assert (await tools['list_web_images'].handler({'ref_id': 'https://example.com/fifth'})).status == 'blocked'
    finally:
        client.close()
