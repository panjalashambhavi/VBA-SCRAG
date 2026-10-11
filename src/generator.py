from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:

    def __init__(self):

        print("Loading language model...")

        model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        self.model.eval()

    def generate(self, question, evidence):

        context = "\n".join(
            [item["text"] for item in evidence]
        )

        prompt = f"""
Answer the question using only the provided evidence.

Question:
{question}

Evidence:
{context}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return answer