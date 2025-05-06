from lib_ml import TextPreprocessor


def test_roundtrip():
    texts = ["Wow, loved this place!", "Crust is not good."]
    tp = TextPreprocessor(max_features=50).fit(texts)

    X = tp.transform(["Wow, loved this place!"])
    assert X.shape[0] == 1
    # 'wow love place' is 3 tokens which are 3 non-zero cells
    assert X.nnz == 3
