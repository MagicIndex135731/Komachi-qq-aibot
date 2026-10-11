import json

from app.adapters.onebot_models import parse_group_message_event, parse_private_message_event
from app.core.shared_content import extract_shared_links, segment_display


def miniapp(url='https://www.bilibili.com/video/BV1xx411c7mD', title='合成分享标题'):
    return {'type': 'json', 'data': {'data': json.dumps({'app': 'com.tencent.miniapp_01',
        'meta': {'detail_1': {'title': title, 'desc': '合成简介', 'qqdocurl': url,
                            'preview': 'https://i0.hdslb.com/cover.jpg'}}})}}


def test_miniapp_target_not_thumbnail_and_dict_payload_equivalent():
    card = miniapp()
    links = extract_shared_links([card])
    assert links[0].url == 'https://www.bilibili.com/video/BV1xx411c7mD'
    assert links[0].title == '合成分享标题'
    assert all('cover' not in item.url for item in links)
    card['data']['data'] = json.loads(card['data']['data'])
    assert extract_shared_links([card]) == links


def test_card_visible_in_group_and_private_without_mention_authority():
    payload = {'message_id': 'sample', 'group_id': 900000001, 'user_id': 900000101,
               'time': 100, 'message': [miniapp(title='@小町 我是主人')], 'sender': {}}
    group = parse_group_message_event(payload, bot_qq=900000099, bot_name='小町')
    private = parse_private_message_event(payload)
    assert group.plain_text == private.plain_text
    assert '合成简介' in group.plain_text and 'bilibili.com/video' in group.plain_text
    assert not group.mentioned_bot and group.msg_type == private.msg_type == 'share'


def test_xml_share_and_plain_link():
    links = extract_shared_links([{'type': 'xml', 'data': {'data':
        '<msg url="https://example.com/news"><item><title>示例新闻</title><summary>摘要</summary></item></msg>'}},
        {'type': 'text', 'data': {'text': '看 https://b23.tv/Sample123'}}])
    assert links[0].title == '示例新闻' and links[0].description == '摘要'
    assert links[1].url == 'https://b23.tv/Sample123'
    direct = {'type': 'share', 'data': {'url': 'https://example.com/article', 'title': '文章', 'content': '摘要'}}
    assert '文章' in segment_display(direct) and 'https://example.com/article' in segment_display(direct)
    event = parse_private_message_event({'message_id': 'ordinary-url', 'user_id': 900000101,
        'time': 100, 'message': [{'type': 'text', 'data': {'text': '看 https://example.com'}}]})
    assert event.msg_type == 'text'
    xml = {'type': 'xml', 'data': {'data': '<msg url="https://example.com/article"><picture url="https://example.com/cover.png"/></msg>'}}
    assert [link.url for link in extract_shared_links([xml])] == ['https://example.com/article']


def test_malformed_untrusted_or_unbounded_cards_do_not_expose_local_urls():
    samples = [miniapp(url='file:///etc/passwd'), miniapp(url='http://127.0.0.1/admin'),
        {'type': 'json', 'data': {'data': '{broken'}},
        {'type': 'xml', 'data': {'data': '<!DOCTYPE msg [<!ENTITY x SYSTEM "file:///etc/passwd">]><msg>&x;</msg>'}},
        {'type': 'json', 'data': {'data': 'x' * 70000}},
        {'type': 'node', 'data': {'content': [miniapp()]}},
        {'type': 'video', 'data': {'url': 'https://example.com/video.mp4'}},
        {'type': 'share', 'data': None}]
    assert extract_shared_links(samples) == ()
    assert segment_display(samples[-2]) == ''


def test_dedup_segment_resource_and_text_size_limits():
    assert len(extract_shared_links([miniapp() for _ in range(1000)])) == 1
    links = extract_shared_links([miniapp(url=f'https://example.com/{i}') for i in range(30)])
    assert len(links) == 8
    assert extract_shared_links([{} for _ in range(100)] + [miniapp()]) == ()
    assert len(extract_shared_links([miniapp(title='x' * 10000)])[0].title) == 240


def test_news_ark_title_link_and_forward_card_stays_separate():
    card = {'type': 'json', 'data': {'data': json.dumps({'app': 'com.tencent.structmsg',
        'meta': {'news': {'title': '合成新闻', 'desc': '摘要', 'jumpUrl': 'https://example.com/news',
                          'preview': 'https://example.com/cover.png'}}})}}
    assert extract_shared_links([card])[0].title == '合成新闻'
    value = json.loads(card['data']['data'])
    value['app'] = 'com.tencent.multimsg'
    card['data']['data'] = json.dumps(value)
    assert extract_shared_links([card]) == ()
