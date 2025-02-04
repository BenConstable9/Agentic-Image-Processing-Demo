from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# Setup Models
GPT_4O_MODEL = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ["OpenAI__CompletionDeployment"],
    model=os.environ["OpenAI__CompletionDeployment"],
    api_version=os.environ["OpenAI__ApiVersion"],
    azure_endpoint=os.environ["OpenAI__Endpoint"],
    azure_ad_token_provider=None,
    model_capabilities={
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
    temperature=0,
)

GPT_4O_MINI_MODEL = AzureOpenAIChatCompletionClient(
    azure_deployment=os.environ["OpenAI__CompletionDeployment"],
    model=os.environ["OpenAI__CompletionDeployment"],
    api_version=os.environ["OpenAI__ApiVersion"],
    azure_endpoint=os.environ["OpenAI__Endpoint"],
    azure_ad_token_provider=None,
    model_capabilities={
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
    temperature=0,
)
