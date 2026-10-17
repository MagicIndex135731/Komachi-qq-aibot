import io
import base64

from PIL import Image,ImageDraw,ImageOps

from app.agent.media_identity import observed_thumbnail,same_observed_photo


def thumb(pixels):
    buffer=io.BytesIO();pixels.save(buffer,format='PNG')
    return observed_thumbnail({'data':base64.b64encode(buffer.getvalue()).decode()})


def photo():
    image=Image.new('RGB',(100,150),'navy');draw=ImageDraw.Draw(image)
    draw.rectangle((5,5,35,130),fill='yellow')
    draw.ellipse((20,30,80,100),fill='red')
    draw.polygon([(50,45),(92,135),(20,120)],fill='green')
    return image


def test_full_frame_resize_and_bordered_cover_crop_match_without_merging_different_pose():
    image=photo()
    assert same_observed_photo(thumb(image),thumb(image.resize((200,300))))
    cover=ImageOps.expand(image.crop((10,40,90,90)).resize((320,200)),border=(12,0,12,0),fill='white')
    assert same_observed_photo(thumb(image),thumb(cover))
    changed=photo();ImageDraw.Draw(changed).rectangle((40,45,90,115),fill='orange')
    assert not same_observed_photo(thumb(image),thumb(changed))


def test_bad_or_unmaterialized_pixel_input_produces_no_identity():
    assert observed_thumbnail({'data':'not base64'}) is None
    assert observed_thumbnail({'data':'data:image/png;base64,eA=='}) is None
    assert observed_thumbnail({'url':'https://example.com/image.jpg'}) is None
