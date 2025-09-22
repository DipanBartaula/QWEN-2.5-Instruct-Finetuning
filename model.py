from transformers import AutoModelForCausalLM, AutoTokenizer

def get_qwen_model(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """
    Returns QWEN 2.5 Instruct model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    return model, tokenizer
