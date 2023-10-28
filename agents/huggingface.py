import torch
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration


class HuggingFaceAgent():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=self.do_sample)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=self.do_sample)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class FlanT5Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
        self.model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

    def interact(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class FlanUL2Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
        self.do_sample = args.do_sample_for_local_models

class Llama2Agent(HuggingFaceAgent):
    def __init__(self, args):
        if type(args) is dict:
            args = SimpleNamespace(**args)
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/{}".format(args.model))
        if "70b" in args.model or "13b" in args.model:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/{}".format(args.model), device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/{}".format(args.model), device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        # attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_new_tokens=128)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses