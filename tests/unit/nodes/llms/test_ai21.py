import pytest

from dynamiq.connections import AI21 as AI21Connection
from dynamiq.nodes.llms import AI21
from dynamiq.utils.litellm import get_litellm


@pytest.mark.parametrize("model", ["jamba-mini", "ai21/jamba-mini", "ai21_chat/jamba-mini"])
def test_ai21_models_go_to_ai21(model):
    llm = AI21(connection=AI21Connection(api_key="test"), model=model)

    _, provider, _, api_base = get_litellm().get_llm_provider(llm.model)

    assert llm.model == "ai21_chat/jamba-mini"
    assert provider == "ai21_chat"
    assert api_base.startswith("https://api.ai21.com/")
