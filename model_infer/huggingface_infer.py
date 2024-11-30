from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



# def 

model_path = "/mnt/public/zhouyiqing/hf_models/edu/qwen2.5-7b/level1_naive_task_segmentation_withoutO_20241021_iter500"

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                device_map='auto',
                                                torch_dtype=torch.bfloat16)
    return model, tokenizer

system = '我爬取了一些碎片化的网页数据，数据内容鱼龙混杂，我希望过滤一些文章出来。我会给你爬取到的网页数据开头，如果你确信它是一篇文章的开头，请回复1；如果它像一篇文章，但是你不确定，请回复2；其他情况请回复0。'
user = '"<|extra_21|>除了挣钱，对什么兴趣都没兴趣怎么办？\n", "杂项\n", "感觉自己很失败，看到集思录都是几百万几千万几亿资金，再看到自己的本金和收益，可望不可即。", "再看周围拆迁户，比奋斗许多年都更有钱。\n", "<|extra_22|>147 个回复\n", "2\n", "凡先生\n", "赞同来自:", "wdwonderone、\n", "欧阳修\n", "最多跟周边的同学 朋友对比一下就好。", "还有起点就不同的没必要比，人家拆迁你不拆的也别比，把自己搞的心理失衡更赚不到钱。\n", "话说楼主多大？", "连性欲都没有了么？\n", "1\n", "梧桐雨\n", "赞同来自:\n", "cddw\n", "健身，深蹲，你会发现，能吃能喝能滚床单。\n", "0\n", "dubaby01\n", "赞同来自:\n", "@闭着眼呼吸\n", "他都对啥都不感兴趣了还要那腺有啥用果然是年轻人，只想着前列腺的爽，对老年人来说前列腺的不爽才是真的痛点。\n", "1\n", "deepocean\n", "赞同来自:\n", "米兰的螺丝钉\n", "至少你还对赚钱感兴趣，这已经很好了\n", '
output_format = '```json\n{"type":0/1/2}'
data = f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user} {output_format}<|im_end|>\n<|im_start|>assistant'

model, tokenizer = load_model_and_tokenizer(model_path)
model_inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=False).to(model.device)
with torch.no_grad():
    output_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True,
                                # stopping_criteria=[0]
                                )
# generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(data, output_ids)
#         ]
responses = [tokenizer.decode(gen, skip_special_tokens=False) for gen in output_ids]
print(responses)