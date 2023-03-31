import dl_translate as dlt

# pip3 install dl_translate
mt = dlt.TranslationModel("./weights/m2m100_418M", model_family="m2m100") # , device="cpu"

def en_to_zh(text):
    return mt.translate(text, source="en", target="zh")

def zh_to_en(text):
    return mt.translate(text, source="zh", target="en")
