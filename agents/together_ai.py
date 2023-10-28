import os
import time
import requests
from types import SimpleNamespace

URL = "https://api.together.xyz/inference"

class TogetherAIAgent():
    def __init__(self, kwargs: dict):
        self.api_key = os.getenv('TOGETHERAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.args.model = self.args.model.removesuffix("-tg")
        self.set_instance_state(action='start')

    def set_instance_state(self, action='start'):
        """
        action: 'start' or 'stop'
        """
        url = "https://api.together.xyz/instances/{}?model=togethercomputer%2F{}".format(action, self.args.model)

        while True:
            response = requests.post(url, headers=self.instance_headers)
            if response.status_code == 200:
                break
            print(">>> Error: {}\nRetrying...".format(response.text))
            time.sleep(1)

        if action == 'start':
            print("\n>>> Instance {} started!".format(self.args.model))
        elif action == 'stop':
            print("\n>>> Instance {} stopped!".format(self.args.model))

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "toghercomputer/llama-2-70b-chat"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 1.0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

        self.instance_headers = {
            "accept": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        self.inference_headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
    
    def generate(self, prompt):
        payload = {
            "model": "togethercomputer/" + self.args.model,
            "max_tokens": self.args.max_tokens,
            # "stop": "",
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": 50,
            "repetition_penalty": 1,
            "prompt": prompt,
        }

        response = requests.post(URL, json=payload, headers=self.inference_headers)

        return response

    def parse_basic_text(self, response):
        output = response.json()['output']['choices'][0]['text'].strip()
        return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output
