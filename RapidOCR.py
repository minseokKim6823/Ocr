from rapidocr import RapidOCR

engine = RapidOCR()

img_url = "https://mblogthumb-phinf.pstatic.net/MjAxODAzMTNfMTE2/MDAxNTIwODk3MjIwMDI3.6mc6cx702UIJHC5yLDPD3j9akaoFTI5VYlmmLZ0lurEg.SKshFmJL2TyR9-ewlH3eZFzgpw-9TBaH-yNAfdK4ingg.JPEG.wd0506/%EC%9E%90%EA%B8%B0%EC%95%9E%EC%88%98%ED%91%9C.jpg?type=w800"
result = engine(img_url, return_word_box=True, return_single_char_box=True)
print(result)

result.vis("vis_result.jpg")



