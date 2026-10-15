"""Images edits keep subject labels aligned with surviving uploaded bytes."""
import base64

import httpx
import pytest

from app.core.message_content import ImageAttachment
from app.providers.llm_client import LlmClient


@pytest.mark.parametrize('with_labels', [True, False])
def test_multipart_roles_follow_loaded_references_and_user_instruction_is_last(tmp_path, with_labels):
    first = tmp_path / 'first.png'; first.write_bytes(b'synthetic-first-image')
    second = tmp_path / 'second.png'; second.write_bytes(b'synthetic-second-image')
    bodies = []
    def handle(request):
        bodies.append(request.content)
        return httpx.Response(200, json={'data': [{'b64_json': base64.b64encode(b'synthetic-result').decode()}]})
    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = LlmClient(base_url='https://synthetic.invalid/v1', api_key='synthetic',
                       model='gemini-3.1-flash-image', http_client=http)
    prompt = 'Keep both characters and change the background only.'
    try:
        client.edit_image(prompt=prompt, model='gemini-3.1-flash-image', max_attempts=1, images=[
            ImageAttachment(url='', local_path=str(tmp_path / 'missing.png'), reference_subject='missing character'),
            ImageAttachment(url='', local_path=str(first), reference_subject='first character' if with_labels else None),
            ImageAttachment(url='', local_path=str(second), reference_subject='second character' if with_labels else None),
        ])
    finally:
        http.close()
    assert len(bodies) == 1
    body = bodies[0].decode()
    assert body.index('synthetic-first-image') < body.index('synthetic-second-image')
    assert 'missing character' not in body
    if with_labels:
        assert body.index('Reference image 1: "first character"') < body.index('Reference image 2: "second character"') < body.index(prompt)
    else:
        assert 'Reference image' not in body
        assert 'Final image instruction:' not in body


def test_no_loaded_reference_never_sends_text_only_image_generation(tmp_path):
    def handle(request):
        pytest.fail('missing references must not issue an image request')
    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = LlmClient(base_url='https://synthetic.invalid/v1', api_key='synthetic', model='gemini-3.1-flash-image', http_client=http)
    try:
        with pytest.raises(ValueError, match='usable input image'):
            client.edit_image(prompt='Use the character.', model='gemini-3.1-flash-image', max_attempts=1,
                              images=[ImageAttachment(url='', local_path=str(tmp_path / 'missing.png'), reference_subject='character')])
    finally:
        http.close()
