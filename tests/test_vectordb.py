
from openrag.components.indexer.vectordb import MilvusDB

class TestMilvusDB:
    def test_complex_templating():
        partition = ["bob.localhost"]
        filter = dict(file_id="a3b2c1", custom_param=314)
        expr, params = MilvusDB._build_expr_template_and_params(partition, filter)
        assert expr == "partition in {partition} and file_id == {file_id} and custom_param == {custom_param}"

        # Note how parameter values are not converted to str
        assert params == {"partition": ["bob.localhost"], "file_id": "a3b2c1", "custom_param": 314}

    def test_templating_no_filter():
        # If there is no filter, the search is run on every document
        partition = ["all"]
        filter = dict()
        expr, params = MilvusDB._build_expr_template_and_params(partition, filter)
        assert expr == ""
        assert params == {}