import tinker
from oai_utils.tinker import LogprobLitellmModel
from oai_utils.tinker.litellm_model import TinkerLLM
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor


def setup_tinkermodel(
    service_client: tinker.ServiceClient, model_name: str, path: str | None = None
) -> tuple[LogprobLitellmModel, Tokenizer, Renderer]:
    sampling_client = service_client.create_sampling_client(
        base_model=model_name, model_path=path
    )
    tokenizer = sampling_client.get_tokenizer()
    image_processor = get_image_processor(model_name)

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer, image_processor)
    tinker_llm = TinkerLLM(
        model_name=model_name, renderer=renderer, tokenizer=tokenizer
    )
    tinker_llm.rewrite_litellm_custom_providers()
    litellm_model_name = f"tinker/{model_name}"
    model = LogprobLitellmModel(
        model=litellm_model_name, sampling_client=sampling_client
    )

    return model, tokenizer, renderer
