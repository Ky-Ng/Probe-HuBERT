class TenseLax:
    _pairs = (
        ("iy", "ih"),
        ("eh", "ey"),
        ("eh", "ae"),
        ("ow", "ao"),
        ("uw", "uh"),
    )

    _TIMIT_to_IPA_map = {
        "iy": "i",
        "ih": "ɪ",
        "eh": "ɛ",
        "ey":  "eɪ",
        "ae": "æ",
        "ow": "oʊ",
        "ao": "ɔ",
        "uw": "u",
        "uh": "ʊ"
    }

    def getIPA(TIMIT_SEG: str) -> str:
        return TenseLax._TIMIT_to_IPA_map[TIMIT_SEG]

    def getPairs() -> tuple[tuple]:
        return TenseLax._pairs

# print("Testing Tense Lax")
# print(TenseLax.getIPA("ey"))
# print(type(TenseLax.getPairs()))