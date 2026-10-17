"""Conservative pixel evidence for resized photos and common page cover crops.

This is a bounded duplicate check, not person recognition or semantic proof.
Uncertain or unsupported shapes retain separate identities for model review.
"""
import base64
import io

from PIL import Image, ImageChops, ImageOps, ImageStat


def observed_thumbnail(attachment):
    try:
        if isinstance(attachment,dict):
            raw=attachment.get('data','')
            if not isinstance(raw,str) or len(raw)>12*1024*1024:return None
            if raw.startswith('data:'):raw=raw.split(',',1)[1]
            source=io.BytesIO(base64.b64decode(raw,validate=True))
        else:
            source=getattr(attachment,'local_path',None)
            if not source:return None
        with Image.open(source) as image:
            if image.width*image.height>16_000_000 or getattr(image,'is_animated',False):return None
            image=ImageOps.exif_transpose(image).convert('RGB')
            # Uniform borders occur on page covers, without changing the photo.
            corners=[image.getpixel(p) for p in ((0,0),(image.width-1,0),(0,image.height-1),(image.width-1,image.height-1))]
            if all(max(abs(a-b) for a,b in zip(corners[0],c))<=4 for c in corners):
                box=ImageChops.difference(image,Image.new('RGB',image.size,corners[0])).convert('L').point(
                    lambda p:255 if p>4 else 0).getbbox()
                if box:image=image.crop(box)
            image.thumbnail((96,96),Image.Resampling.LANCZOS)
            return image.copy()
    except (OSError,ValueError,TypeError,IndexError,Image.DecompressionBombError):return None


def same_observed_photo(left,right):
    def matches(full,other):
        ratio=other.width/other.height
        width=min(full.width,round(full.height*ratio))
        height=min(full.height,round(full.width/ratio))
        if width*height<full.width*full.height*.25:return False
        target=other.resize((24,24),Image.Resampling.BILINEAR)
        similar_ratio=abs(full.width/full.height-ratio)<.05
        for scale in ((1,) if similar_ratio else (1,.9,.8,.75,.7,.6,.5)):
            w,h=max(1,round(width*scale)),max(1,round(height*scale))
            dx,dy=full.width-w,full.height-h
            # Portrait-to-landscape page covers commonly trim both axes.
            xs=sorted({0,dx//4,dx//2,3*dx//4,dx})
            ys=range(0,dy+1,2) if dy else (0,)
            for x in xs:
                for y in ys:
                    sample=full.crop((x,y,x+w,y+h)).resize((24,24),Image.Resampling.BILINEAR)
                    diff=ImageChops.difference(sample,target)
                    means=ImageStat.Stat(diff).mean
                    if max(means)<=7.5 and sum(means)/3<=5:
                        histogram=diff.convert('L').histogram()
                        if sum(histogram[25:])<=23:return True
        return False
    return matches(left,right) or matches(right,left)
