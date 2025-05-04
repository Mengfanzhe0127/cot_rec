import asyncio
from typing import List

import yaml
from litellm import Router


class Model:
    def __init__(self, model_list_file_path, model_name, disable_cooldowns=True, *args, **kwargs):
        with open(model_list_file_path) as f:
            model_list = yaml.safe_load(f)

        self.router = Router(model_list=model_list, disable_cooldowns=disable_cooldowns, *args, **kwargs)

        self.model_name = model_name

    async def _chat_completion(self, prompt, *args, **kwargs):
        messages = [{"role": "user", "content": prompt}]
        response_dict = await self.router.acompletion(
            model=self.model_name,
            messages=messages,
            *args, **kwargs
        )

        if 'logprobs' in kwargs and kwargs['logprobs'] is True:
            response_list = [dict(
                content=choice.message.content,
                logprobs=choice.logprobs
            ) for choice in response_dict.choices]
        else:
            response_list = [choice.message.content for choice in response_dict.choices]

        return response_list

    async def generate(self, prompt, *args, **kwargs) -> List[str]:
        return await self._chat_completion(prompt, *args, **kwargs)

    async def batch_generate(self, prompt_list, *args, **kwargs) -> List[List[str]]:
        return await asyncio.gather(*(self._chat_completion(prompt, *args, **kwargs) for prompt in prompt_list))
